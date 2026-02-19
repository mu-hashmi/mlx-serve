#[path = "support/common.rs"]
mod common;

use mlx_tokenizers::{ChatMessage, TokenizerBundle};

#[test]
fn check_2_1_encode_round_trip() {
    let _guard = common::test_lock();
    let model_dir = common::resolve_model_dir();
    let tokenizer = TokenizerBundle::from_model_dir(&model_dir).expect("failed to load tokenizer");

    let text = "Hello, world!";
    let encoded = tokenizer
        .encode(text, false)
        .expect("failed to encode text");
    let decoded = tokenizer
        .decode(encoded.get_ids(), true)
        .expect("failed to decode token ids");

    assert!(
        decoded == text || decoded.trim() == text,
        "decoded text mismatch: expected '{text}', got '{decoded}'"
    );
}

#[test]
fn check_2_2_chat_template() {
    let _guard = common::test_lock();
    let model_dir = common::resolve_model_dir();
    let tokenizer = TokenizerBundle::from_model_dir(&model_dir).expect("failed to load tokenizer");

    let rendered = tokenizer
        .apply_chat_template(
            &[ChatMessage {
                role: "user".to_owned(),
                content: "Hi".to_owned(),
            }],
            true,
            None,
        )
        .expect("failed to render chat template");

    assert!(
        rendered.contains("<|start_header_id|>"),
        "missing <|start_header_id|> in rendered template: {rendered}"
    );
    assert!(
        rendered.contains("<|eot_id|>"),
        "missing <|eot_id|> in rendered template: {rendered}"
    );
}
