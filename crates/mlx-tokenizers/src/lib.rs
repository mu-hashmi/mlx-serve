#![warn(missing_docs)]

//! HuggingFace tokenizer and chat-template integration.

use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;
use tokenizers::{Encoding, Tokenizer};

/// Errors produced by tokenizer loading, encoding, or template rendering.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// Filesystem I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Tokenizer operation failed.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// Chat template is required but missing.
    #[error("tokenizer chat_template is missing")]
    MissingChatTemplate,
    /// Jinja template rendering failed.
    #[error("chat template rendering failed: {0}")]
    Template(#[from] minijinja::Error),
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role (for example `system`, `user`, or `assistant`).
    pub role: String,
    /// Message content.
    pub content: String,
}

/// Loaded tokenizer plus optional chat template metadata.
#[derive(Debug, Clone)]
pub struct TokenizerBundle {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    chat_template: Option<String>,
}

impl TokenizerBundle {
    /// Load tokenizer assets from a model directory.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))?;

        let config_path = model_dir.as_ref().join("tokenizer_config.json");
        let chat_template = if config_path.exists() {
            let raw = std::fs::read_to_string(&config_path)?;
            let cfg: TokenizerConfigFile = serde_json::from_str(&raw)?;
            cfg.chat_template
        } else {
            None
        };

        Ok(Self {
            tokenizer,
            chat_template,
        })
    }

    /// Return the underlying HuggingFace tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Return chat template content when available.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, TokenizerError> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Tokenizer(e.to_string()))
    }

    /// Render the model chat template with the provided messages.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&[Value]>,
    ) -> Result<String, TokenizerError> {
        let template_source = self
            .chat_template
            .as_deref()
            .ok_or(TokenizerError::MissingChatTemplate)?;

        let mut env = Environment::new();
        env.add_template("chat", template_source)?;
        let template = env.get_template("chat")?;

        let mut context = Map::new();
        context.insert("messages".to_owned(), serde_json::to_value(messages)?);
        context.insert(
            "add_generation_prompt".to_owned(),
            Value::Bool(add_generation_prompt),
        );
        if let Some(values) = tools {
            context.insert("tools".to_owned(), Value::Array(values.to_vec()));
        }

        template.render(context).map_err(TokenizerError::from)
    }
}
