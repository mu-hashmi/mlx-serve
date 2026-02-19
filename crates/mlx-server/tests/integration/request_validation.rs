//! Tests for request body deserialization and validation.
//!
//! These tests verify that the type system correctly accepts valid requests
//! and rejects malformed ones at the serde level, before any engine interaction.

#![allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]

use mlx_server::types::openai::{
    ChatCompletionMessage, ChatCompletionRequest, CompletionRequest, EmbeddingInput,
    EmbeddingRequest, StopSequence,
};

// ---------------------------------------------------------------------------
// OpenAI Chat Completions
// ---------------------------------------------------------------------------

#[test]
fn chat_request_minimal_valid() {
    let json = r#"{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "test-model");
    assert_eq!(req.messages.len(), 1);
    assert!(req.max_tokens.is_none());
    assert!(req.temperature.is_none());
    assert!(req.top_p.is_none());
    assert!(req.stream.is_none());
    assert!(req.stop.is_none());
    assert!(req.tools.is_none());
    assert!(req.response_format.is_none());
}

#[test]
fn chat_request_all_optional_fields() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": true,
        "stop": ["END", "\n"],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "response_format": {"type": "json_object"}
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.max_tokens, Some(256));
    assert!((req.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
    assert_eq!(req.stream, Some(true));
    assert!(req.tools.is_some());
    assert!(req.response_format.is_some());
}

#[test]
fn chat_request_missing_model_fails() {
    let json = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}

#[test]
fn chat_request_missing_messages_fails() {
    let json = r#"{"model": "m"}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}

#[test]
fn chat_request_empty_messages_deserializes() {
    // Empty messages array deserializes fine; the handler validates emptiness
    let json = r#"{"model": "m", "messages": []}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.messages.is_empty());
}

#[test]
fn chat_request_stop_single_string() {
    let json = r#"{"model": "m", "messages": [], "stop": "STOP"}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    match req.stop.unwrap() {
        StopSequence::Single(s) => assert_eq!(s, "STOP"),
        StopSequence::Multiple(_) => panic!("expected Single variant"),
    }
}

#[test]
fn chat_request_stop_array() {
    let json = r#"{"model": "m", "messages": [], "stop": ["a", "b", "c"]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    match req.stop.unwrap() {
        StopSequence::Multiple(v) => assert_eq!(v.len(), 3),
        StopSequence::Single(_) => panic!("expected Multiple variant"),
    }
}

#[test]
fn stop_sequence_extract_none() {
    let result = StopSequence::extract(None);
    assert!(result.is_empty());
}

#[test]
fn stop_sequence_extract_single() {
    let result = StopSequence::extract(Some(StopSequence::Single("end".to_owned())));
    assert_eq!(result, vec!["end"]);
}

#[test]
fn stop_sequence_extract_multiple() {
    let result = StopSequence::extract(Some(StopSequence::Multiple(vec![
        "a".to_owned(),
        "b".to_owned(),
    ])));
    assert_eq!(result, vec!["a", "b"]);
}

#[test]
fn chat_message_with_tool_calls() {
    let json = r#"{
        "role": "assistant",
        "tool_calls": [{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
        }]
    }"#;
    let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "assistant");
    assert!(msg.content.is_none());
    let calls = msg.tool_calls.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_abc");
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn chat_message_tool_result() {
    let json = r#"{
        "role": "tool",
        "content": "72 degrees Fahrenheit",
        "tool_call_id": "call_abc"
    }"#;
    let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "tool");
    assert_eq!(msg.tool_call_id, Some("call_abc".to_owned()));
}

#[test]
fn chat_request_wrong_type_for_max_tokens_fails() {
    let json = r#"{"model": "m", "messages": [], "max_tokens": "not_a_number"}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// OpenAI Completions
// ---------------------------------------------------------------------------

#[test]
fn completion_request_minimal() {
    let json = r#"{"model": "m", "prompt": "Once upon"}"#;
    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "Once upon");
    assert!(req.stream.is_none());
}

#[test]
fn completion_request_missing_prompt_fails() {
    let json = r#"{"model": "m"}"#;
    let result = serde_json::from_str::<CompletionRequest>(json);
    assert!(result.is_err());
}

#[test]
fn completion_request_missing_model_fails() {
    let json = r#"{"prompt": "test"}"#;
    let result = serde_json::from_str::<CompletionRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// OpenAI Embeddings
// ---------------------------------------------------------------------------

#[test]
fn embedding_request_single_input() {
    let json = r#"{"model": "m", "input": "hello world"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    assert!(matches!(req.input, EmbeddingInput::Single(ref s) if s == "hello world"));
}

#[test]
fn embedding_request_multiple_inputs() {
    let json = r#"{"model": "m", "input": ["hello", "world"]}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    match &req.input {
        EmbeddingInput::Multiple(v) => assert_eq!(v.len(), 2),
        _ => panic!("expected Multiple variant"),
    }
}

#[test]
fn embedding_request_with_encoding_format() {
    let json = r#"{"model": "m", "input": "hi", "encoding_format": "float"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.encoding_format, Some("float".to_owned()));
}

#[test]
fn embedding_request_missing_input_fails() {
    let json = r#"{"model": "m"}"#;
    let result = serde_json::from_str::<EmbeddingRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Extra unknown fields
// ---------------------------------------------------------------------------

#[test]
fn chat_request_with_extra_unknown_fields_accepted() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_field": 42,
        "another_unknown": {"nested": true},
        "vendor_specific_param": "value"
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "m");
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn completion_request_with_stop_as_array() {
    let json = r#"{
        "model": "m",
        "prompt": "Once upon",
        "stop": ["\n\n", "END", "---"]
    }"#;
    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    match req.stop.unwrap() {
        StopSequence::Multiple(v) => assert_eq!(v.len(), 3),
        StopSequence::Single(_) => panic!("expected Multiple variant"),
    }
}
