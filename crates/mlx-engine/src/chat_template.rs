use std::path::Path;

use minijinja::{
    Environment, Error as MiniJinjaError, ErrorKind as MiniJinjaErrorKind, Value, value::Kwargs,
};
use serde::Serialize;

use crate::error::EngineError;

/// A chat message for template rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    /// Chat role (`system`, `user`, `assistant`, `tool`).
    pub role: String,
    /// Message text content.
    pub content: String,
    /// Optional tool-call payloads for assistant tool messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Renders chat messages using a Jinja2 template (HuggingFace format).
pub struct ChatTemplateRenderer {
    env: Environment<'static>,
    special_tokens: SpecialTokens,
}

#[derive(Debug, Clone, Default)]
struct SpecialTokens {
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
    unk_token: Option<String>,
    sep_token: Option<String>,
}

impl ChatTemplateRenderer {
    /// Create a renderer from a Jinja2 template string.
    pub fn new(template_source: impl Into<String>) -> Result<Self, EngineError> {
        Self::new_with_tokens(template_source, SpecialTokens::default())
    }

    fn new_with_tokens(
        template_source: impl Into<String>,
        special_tokens: SpecialTokens,
    ) -> Result<Self, EngineError> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned("chat".to_owned(), template_source.into())
            .map_err(|e| EngineError::Template(e.to_string()))?;
        Ok(Self {
            env,
            special_tokens,
        })
    }

    /// Load template from a model directory (chat_template.jinja or tokenizer_config.json).
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, EngineError> {
        let special_tokens = load_special_tokens(model_dir)?;

        // Prefer standalone chat_template.jinja
        let jinja_path = model_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let template = std::fs::read_to_string(&jinja_path)
                .map_err(|e| EngineError::Template(format!("Failed to read template: {e}")))?;
            return Self::new_with_tokens(&template, special_tokens);
        }

        // Fall back to tokenizer_config.json
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Template(format!("Failed to read config: {e}")))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| EngineError::Template(format!("Invalid JSON: {e}")))?;
            if let Some(template) = config.get("chat_template").and_then(|v| v.as_str()) {
                return Self::new_with_tokens(template, special_tokens);
            }
        }

        Err(EngineError::Template(
            "No chat template found in model directory".to_owned(),
        ))
    }

    /// Apply the chat template to messages, returning the formatted prompt string.
    pub fn apply(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<String, EngineError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| EngineError::Template(e.to_string()))?;

        let mut context = serde_json::Map::new();
        context.insert(
            "messages".to_owned(),
            serde_json::to_value(messages)
                .map_err(|e| EngineError::Template(format!("Failed to serialize messages: {e}")))?,
        );
        context.insert(
            "add_generation_prompt".to_owned(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        if let Some(tool_list) = tools {
            context.insert(
                "tools".to_owned(),
                serde_json::Value::Array(tool_list.to_vec()),
            );
        }
        self.special_tokens.insert_into_context(&mut context);

        tmpl.render(context)
            .map_err(|e| EngineError::Template(e.to_string()))
    }
}

/// Custom tojson filter for minijinja (used by HF chat templates).
fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<String, MiniJinjaError> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;

    let serialized = if let Some(width) = indent {
        let mut output = Vec::new();
        let indentation = vec![b' '; width];
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&indentation);
        let mut serializer = serde_json::Serializer::with_formatter(&mut output, formatter);
        value.serialize(&mut serializer).map_err(|e| {
            MiniJinjaError::new(
                MiniJinjaErrorKind::InvalidOperation,
                "JSON serialization failed",
            )
            .with_source(e)
        })?;
        String::from_utf8(output).map_err(|e| {
            MiniJinjaError::new(MiniJinjaErrorKind::InvalidOperation, "JSON encoding failed")
                .with_source(e)
        })?
    } else {
        serde_json::to_string(&value).map_err(|e| {
            MiniJinjaError::new(
                MiniJinjaErrorKind::InvalidOperation,
                "JSON serialization failed",
            )
            .with_source(e)
        })?
    };

    Ok(serialized)
}

fn raise_exception(message: String) -> Result<String, MiniJinjaError> {
    Err(MiniJinjaError::new(
        MiniJinjaErrorKind::InvalidOperation,
        message,
    ))
}

impl SpecialTokens {
    fn from_json(value: &serde_json::Value) -> Self {
        Self {
            bos_token: token_value(value, "bos_token"),
            eos_token: token_value(value, "eos_token"),
            pad_token: token_value(value, "pad_token"),
            unk_token: token_value(value, "unk_token"),
            sep_token: token_value(value, "sep_token"),
        }
    }

    fn merge_missing(&mut self, other: Self) {
        if self.bos_token.is_none() {
            self.bos_token = other.bos_token;
        }
        if self.eos_token.is_none() {
            self.eos_token = other.eos_token;
        }
        if self.pad_token.is_none() {
            self.pad_token = other.pad_token;
        }
        if self.unk_token.is_none() {
            self.unk_token = other.unk_token;
        }
        if self.sep_token.is_none() {
            self.sep_token = other.sep_token;
        }
    }

    fn insert_into_context(&self, context: &mut serde_json::Map<String, serde_json::Value>) {
        if let Some(token) = &self.bos_token {
            context.insert(
                "bos_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
        if let Some(token) = &self.eos_token {
            context.insert(
                "eos_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
        if let Some(token) = &self.pad_token {
            context.insert(
                "pad_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
        if let Some(token) = &self.unk_token {
            context.insert(
                "unk_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
        if let Some(token) = &self.sep_token {
            context.insert(
                "sep_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
    }
}

fn token_value(container: &serde_json::Value, key: &str) -> Option<String> {
    match container.get(key) {
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(serde_json::Value::Object(map)) => map
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

fn load_special_tokens(model_dir: &Path) -> Result<SpecialTokens, EngineError> {
    let mut tokens = SpecialTokens::default();
    for file_name in ["tokenizer_config.json", "special_tokens_map.json"] {
        let path = model_dir.join(file_name);
        if !path.exists() {
            continue;
        }
        let raw = std::fs::read_to_string(&path).map_err(|e| {
            EngineError::Template(format!("Failed to read {}: {e}", path.display()))
        })?;
        let json: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            EngineError::Template(format!("Invalid JSON in {}: {e}", path.display()))
        })?;
        tokens.merge_missing(SpecialTokens::from_json(&json));
    }
    Ok(tokens)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
            tool_calls: None,
        }
    }

    /// Create a minijinja environment with the tojson filter and return the
    /// compiled template for `{{ value | tojson }}`.
    fn tojson_env(template_source: &str) -> minijinja::Environment<'static> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        env.add_template_owned("test".to_owned(), template_source.to_owned())
            .unwrap();
        env
    }

    const CHATML_TEMPLATE: &str = r#"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"#;

    const TOJSON_TEMPLATE: &str = r#"{{ value | tojson }}"#;

    #[test]
    fn test_simple_chatml_template() {
        let renderer = ChatTemplateRenderer::new(CHATML_TEMPLATE).unwrap();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello!")];

        let result = renderer.apply(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_tojson_filter() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "hello" })
            .unwrap();
        assert_eq!(result, r#""hello""#);
    }

    #[test]
    fn test_invalid_template_syntax_returns_error() {
        assert!(ChatTemplateRenderer::new("{%- invalid syntax %}}}").is_err());
    }

    #[test]
    fn test_apply_without_generation_prompt() {
        let renderer = ChatTemplateRenderer::new(CHATML_TEMPLATE).unwrap();
        let result = renderer
            .apply(&[msg("user", "Hello!")], None, false)
            .unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_empty_messages() {
        let template = r#"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}"#;
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let result = renderer.apply(&[], None, false).unwrap();
        assert!(!result.contains("<|im_start|>"));
    }

    #[test]
    fn test_apply_with_tools() {
        let template = r#"{%- for message in messages %}
{{ message.content }}
{%- endfor %}
{%- if tools %}
TOOLS:{{ tools | length }}
{%- endif %}"#;

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "test"}})];
        let result = renderer
            .apply(&[msg("user", "Hi")], Some(&tools), false)
            .unwrap();
        assert!(result.contains("TOOLS:1"));
    }

    #[test]
    fn test_from_model_dir_no_template_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_from_model_dir_tokenizer_config_no_chat_template_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_type": "qwen2"}"#,
        )
        .unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_from_model_dir_standalone_jinja_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            r#"{%- for message in messages %}{{ message.content }}{%- endfor %}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "hello")], None, false)
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_from_model_dir_jinja_takes_priority_over_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "JINJA:{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "CONFIG:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "test")], None, false).unwrap();
        assert!(result.starts_with("JINJA:"));
    }

    #[test]
    fn test_from_model_dir_fallback_to_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{%- for message in messages %}{{ message.content }}{%- endfor %}"}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "from_config")], None, false)
            .unwrap();
        assert_eq!(result, "from_config");
    }

    #[test]
    fn test_from_model_dir_injects_bos_token_from_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
                "chat_template": "{{- bos_token }}{%- for message in messages %}{{ message.content }}{%- endfor %}",
                "bos_token": "<s>"
            }"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "hello")], None, false)
            .unwrap();
        assert_eq!(result, "<s>hello");
    }

    #[test]
    fn test_from_model_dir_malformed_tokenizer_config_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            "this is not valid json {{{",
        )
        .unwrap();
        match ChatTemplateRenderer::from_model_dir(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Invalid JSON")),
            Ok(_) => panic!("Expected error for malformed JSON"),
        }
    }

    #[test]
    fn test_apply_with_assistant_role() {
        let template = r#"{%- for message in messages %}<|{{ message.role }}|>{{ message.content }}{%- endfor %}"#;
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![msg("user", "What is 2+2?"), msg("assistant", "4")];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("<|assistant|>4"));
    }

    #[test]
    fn test_apply_with_tool_calls_field() {
        let template = r#"{%- for message in messages %}{{ message.role }}:{{ message.content }}{%- if message.tool_calls %} [tools]{%- endif %}{%- endfor %}"#;
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![ChatMessage {
            role: "assistant".to_owned(),
            content: "calling tool".to_owned(),
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
            })]),
        }];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("[tools]"));
    }

    #[test]
    fn test_tojson_filter_with_nested_objects() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let nested = serde_json::json!({"a": {"b": [1, 2, 3]}});
        let result = tmpl
            .render(minijinja::context! { value => nested })
            .unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            reparsed.get("a").unwrap().get("b").unwrap(),
            &serde_json::json!([1, 2, 3])
        );
    }

    #[test]
    fn test_tojson_filter_with_arrays() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => vec![1, 2, 3] })
            .unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(reparsed, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_tojson_filter_with_special_characters() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "quotes: \"hello\" and backslash: \\" })
            .unwrap();
        let reparsed: String = serde_json::from_str(&result).unwrap();
        assert!(reparsed.contains("quotes: \"hello\""));
        assert!(reparsed.contains("backslash: \\"));
    }

    #[test]
    fn test_template_rendering_error_undefined_variable() {
        let renderer =
            ChatTemplateRenderer::new(r#"{{ undefined_variable.nested_field }}"#).unwrap();
        assert!(renderer.apply(&[msg("user", "hi")], None, false).is_err());
    }
}
