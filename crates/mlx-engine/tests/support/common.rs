use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

use mlx_engine::{
    model_loader,
    simple::{SimpleEngine, SimpleEngineOptions},
};

pub const TEST_MODEL_REPO: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn resolve_model_dir() -> PathBuf {
    model_loader::resolve_model_dir(TEST_MODEL_REPO).expect("failed to resolve test model")
}

pub fn load_engine(max_cache_mb: usize) -> SimpleEngine {
    let options = SimpleEngineOptions {
        max_cache_bytes: max_cache_mb * 1024 * 1024,
        prefix_cache_entries: 128,
        ..SimpleEngineOptions::default()
    };
    SimpleEngine::load_with_options(TEST_MODEL_REPO, options).expect("failed to load test engine")
}

pub fn long_prompt_tokens(engine: &SimpleEngine, min_tokens: usize) -> Vec<u32> {
    let mut text = String::from("Summarize the following points:\n");
    while engine
        .tokenizer()
        .encode(text.as_str(), false)
        .expect("tokenization failed")
        .get_ids()
        .len()
        < min_tokens
    {
        text.push_str("- Apple Silicon MLX inference is fast on Metal kernels.\n");
        text.push_str("- Rust server should isolate KV caches per request.\n");
        text.push_str("- Backpressure avoids OOM under overload.\n");
        text.push_str("- Prefix sharing should reuse system prompts safely.\n");
    }

    engine
        .tokenizer()
        .encode(text.as_str(), false)
        .expect("tokenization failed")
        .get_ids()
        .to_vec()
}
