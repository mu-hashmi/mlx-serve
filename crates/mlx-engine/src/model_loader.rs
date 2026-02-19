use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use mlx_models::{AnyModel, load_tokenizer as shared_load_tokenizer, registry, transformer};

use crate::error::EngineError;

/// Configuration for loading a model from a local directory.
#[derive(Debug)]
pub struct ModelConfig {
    /// Local path containing model files.
    pub model_dir: PathBuf,
    /// Model architecture string from config.json.
    pub model_type: String,
}

impl ModelConfig {
    /// Resolve a local path or HuggingFace repo ID and detect model type.
    pub fn from_source(source: impl AsRef<Path>) -> Result<Self, EngineError> {
        let model_dir = resolve_model_dir(source)?;
        let model_type = registry::detect_model_type(&model_dir)?;

        if !registry::is_supported(&model_type) {
            return Err(EngineError::Model(
                mlx_models::error::ModelError::UnsupportedModel(model_type),
            ));
        }

        Ok(Self {
            model_dir,
            model_type,
        })
    }
}

fn should_fetch_file(path: &str) -> bool {
    path.ends_with(".safetensors")
        || matches!(
            path,
            "config.json"
                | "generation_config.json"
                | "model.safetensors.index.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "tokenizer.model"
                | "vocab.json"
                | "merges.txt"
        )
}

fn resolve_remote_model_dir(repo_id: &str) -> Result<PathBuf, EngineError> {
    let api = Api::new()
        .map_err(|error| EngineError::Generation(format!("failed to initialize HF API: {error}")))?;
    let repo = api.model(repo_id.to_owned());

    let info = repo
        .info()
        .map_err(|error| EngineError::Generation(format!("failed to query HF repo '{repo_id}': {error}")))?;

    for sibling in &info.siblings {
        if should_fetch_file(&sibling.rfilename) {
            repo.get(&sibling.rfilename).map_err(|error| {
                EngineError::Generation(format!(
                    "failed to download '{}' from '{}': {error}",
                    sibling.rfilename, repo_id
                ))
            })?;
        }
    }

    let config_path = repo.get("config.json").map_err(|error| {
        EngineError::Generation(format!("failed to download config.json from '{repo_id}': {error}"))
    })?;

    let model_dir = config_path.parent().ok_or_else(|| {
        EngineError::Generation(format!("invalid config path returned for '{repo_id}'"))
    })?;

    Ok(model_dir.to_path_buf())
}

/// Resolve a model source into a local model directory.
///
/// If `source` exists as a filesystem path, it is used directly.
/// Otherwise `source` is interpreted as a HuggingFace repo ID and downloaded
/// into the local HF cache.
pub fn resolve_model_dir(source: impl AsRef<Path>) -> Result<PathBuf, EngineError> {
    let source = source.as_ref();

    if source.exists() {
        return Ok(source.to_path_buf());
    }

    let repo_id = source.to_string_lossy();
    resolve_remote_model_dir(repo_id.as_ref())
}

/// Load a model from a local path or HuggingFace repo ID.
pub fn load_model(source: impl AsRef<Path>) -> Result<AnyModel, EngineError> {
    let config = ModelConfig::from_source(source)?;

    match config.model_type.as_str() {
        "qwen2" | "qwen3" | "llama" | "mistral" => {
            let model = transformer::load_model(&config.model_dir).map_err(EngineError::Model)?;
            Ok(AnyModel::Transformer(model))
        }
        "qwen3_next" => {
            let model = mlx_models::qwen3_next::load_qwen3_next_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        other => Err(EngineError::Model(
            mlx_models::error::ModelError::UnsupportedModel(other.to_owned()),
        )),
    }
}

/// Load a tokenizer from a local path or HuggingFace repo ID.
pub fn load_tokenizer(source: impl AsRef<Path>) -> Result<tokenizers::Tokenizer, EngineError> {
    let model_dir = resolve_model_dir(source)?;
    shared_load_tokenizer(model_dir).map_err(|e| EngineError::Tokenization(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_models::error::ModelError;

    fn config_for_model(model_type: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
        let result = ModelConfig::from_source(dir.path());
        (dir, result)
    }

    fn config_from_raw(content: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), content).unwrap();
        let result = ModelConfig::from_source(dir.path());
        (dir, result)
    }

    #[test]
    fn model_config_from_dir_qwen2() {
        let (dir, result) = config_for_model("qwen2");
        let config = result.unwrap();
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.model_dir, dir.path());
    }

    #[test]
    fn model_config_from_dir_qwen3() {
        let (_dir, result) = config_for_model("qwen3");
        assert_eq!(result.unwrap().model_type, "qwen3");
    }

    #[test]
    fn model_config_from_dir_llama() {
        let (_dir, result) = config_for_model("llama");
        assert_eq!(result.unwrap().model_type, "llama");
    }

    #[test]
    fn model_config_from_dir_mistral() {
        let (_dir, result) = config_for_model("mistral");
        assert_eq!(result.unwrap().model_type, "mistral");
    }

    #[test]
    fn model_config_from_dir_qwen3_next() {
        let (_dir, result) = config_for_model("qwen3_next");
        assert_eq!(result.unwrap().model_type, "qwen3_next");
    }

    #[test]
    fn model_config_from_dir_unsupported_model_type() {
        let (_dir, result) = config_for_model("gpt2");
        match result {
            Err(e) => assert!(e.to_string().contains("gpt2")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn model_config_from_dir_missing_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let err = ModelConfig::from_source(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Io(_))));
    }

    #[test]
    fn model_config_from_dir_invalid_json() {
        let (_dir, result) = config_from_raw("not valid json {{{");
        let err = result.unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Json(_))));
    }

    #[test]
    fn model_config_from_dir_missing_model_type_field() {
        let (_dir, result) = config_from_raw(r#"{"vocab_size": 32000, "hidden_size": 4096}"#);
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            EngineError::Model(ModelError::UnsupportedModel(_))
        ));
    }

    #[test]
    fn load_tokenizer_missing_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        match load_tokenizer(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Tokenization error")),
            Ok(_) => panic!("Expected error for missing tokenizer.json"),
        }
    }

    #[test]
    fn should_fetch_file_includes_required_assets() {
        assert!(should_fetch_file("config.json"));
        assert!(should_fetch_file("model.safetensors"));
        assert!(should_fetch_file("model-00001-of-00002.safetensors"));
        assert!(should_fetch_file("tokenizer.json"));
        assert!(should_fetch_file("tokenizer_config.json"));
        assert!(!should_fetch_file("README.md"));
    }
}
