//! Inference orchestration for MLX-backed language models.
//!
//! This crate provides model loading, prompt preparation, cache accounting,
//! and token generation utilities used by the HTTP server and CLI.

/// Chat template loading and rendering helpers.
pub mod chat_template;
/// Shared generation output types.
pub mod engine;
/// Engine-level error types.
pub mod error;
/// Model and tokenizer loading utilities.
pub mod model_loader;
/// Request-isolated KV cache accounting and prefix cache manager.
pub mod prompt_cache;
/// Main synchronous generation engine implementation.
pub mod simple;
/// Tool-call parsing for model text output.
pub mod tool_parser;
