//! CLI entrypoint for running mlx-serve.

use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::serve;
use clap::{Args, Parser, Subcommand};
use humansize::{BINARY, format_size};
use mlx_engine::{
    model_loader,
    simple::{SamplingConfig, SimpleEngine, SimpleEngineOptions},
};
use mlx_models::{WeightMapIndex, registry};
use mlx_server::{
    backpressure::BackpressureController,
    config::ServerConfig,
    state::{AppState, SharedState},
};
use safetensors::{SafeTensorError, tensor::SafeTensors};

const MEBIBYTE: usize = 1024 * 1024;

/// Command-line interface for mlx-serve.
#[derive(Debug, Parser)]
#[command(
    name = "mlx-serve",
    version,
    about = "Rust-native MLX inference server"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Supported CLI subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Start an OpenAI-compatible HTTP server.
    Serve(ServeArgs),
    /// Run one-shot generation from a prompt.
    Generate(GenerateArgs),
    /// Print model metadata and memory estimates.
    Info(InfoArgs),
}

/// Arguments for `mlx-serve serve`.
#[derive(Debug, Args)]
struct ServeArgs {
    /// Local model path or Hugging Face repo ID.
    #[arg(long)]
    model: String,

    /// Host interface to bind.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// TCP port to bind.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// KV cache budget in MiB.
    #[arg(long, default_value_t = 4096)]
    max_cache_mb: usize,

    /// Maximum requests admitted before overload (running + waiting for decode slot).
    ///
    /// Actual decode parallelism is fixed to one request at a time.
    #[arg(long, default_value_t = num_cpus::get().max(1))]
    max_admitted_requests: usize,

    /// Maximum queued requests waiting for an active slot.
    #[arg(long, default_value_t = 128)]
    max_queue_size: usize,

    /// Retry-After header value (seconds) for overload responses.
    #[arg(long, default_value_t = 1)]
    retry_after_seconds: u64,

    /// Clear MLX runtime cache after each non-streaming generation request.
    #[arg(long, default_value_t = false)]
    clear_runtime_cache_after_request: bool,

    /// Request timeout in seconds.
    #[arg(long, default_value_t = 300.0)]
    timeout: f64,

    /// Optional API key for bearer auth.
    #[arg(long)]
    api_key: Option<String>,

    /// Optional per-client requests-per-minute limiter.
    #[arg(long, default_value_t = 0)]
    rate_limit: u32,

    /// Default max_tokens used by server endpoints when request omits it.
    #[arg(long, default_value_t = 32768)]
    max_tokens: u32,

    /// Maximum number of cached prompt prefixes.
    #[arg(long, default_value_t = 128)]
    prefix_cache_entries: usize,

    /// Logical page size in KiB for cache accounting.
    #[arg(long, default_value_t = 1024)]
    cache_page_kb: usize,

    /// Minimum prefix length eligible for prefix sharing.
    #[arg(long, default_value_t = 16)]
    min_prefix_tokens: usize,
}

/// Arguments for `mlx-serve generate`.
#[derive(Debug, Args)]
struct GenerateArgs {
    /// Local model path or Hugging Face repo ID.
    #[arg(long)]
    model: String,

    /// Prompt text.
    #[arg(long)]
    prompt: String,

    /// Maximum generated tokens.
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,

    /// Sampling temperature.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Top-p nucleus sampling threshold.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Optional top-k sampling cutoff.
    #[arg(long)]
    top_k: Option<usize>,

    /// Repetition penalty (1.0 disables).
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,

    /// Optional deterministic sampling seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Optional stop sequence. May be repeated.
    #[arg(long = "stop")]
    stop_sequences: Vec<String>,

    /// KV cache budget in MiB.
    #[arg(long, default_value_t = 4096)]
    max_cache_mb: usize,

    /// Maximum number of cached prompt prefixes.
    #[arg(long, default_value_t = 128)]
    prefix_cache_entries: usize,
}

/// Arguments for `mlx-serve info`.
#[derive(Debug, Args)]
struct InfoArgs {
    /// Local model path or Hugging Face repo ID.
    #[arg(long)]
    model: String,
}

/// CLI error type.
#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Engine(#[from] mlx_engine::error::EngineError),

    #[error(transparent)]
    Model(#[from] mlx_models::error::ModelError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    SafeTensor(#[from] SafeTensorError),

    #[error("cache size overflow")]
    Overflow,

    #[error("{0}")]
    Message(String),
}

#[tokio::main]
async fn main() {
    init_tracing();

    if let Err(error) = run().await {
        tracing::error!(error = %error, "command failed");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), CliError> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve(args) => run_serve(args).await,
        Command::Generate(args) => run_generate(args),
        Command::Info(args) => run_info(args),
    }
}

async fn run_serve(args: ServeArgs) -> Result<(), CliError> {
    let model_id = model_id_from_source(&args.model);
    let engine_options = SimpleEngineOptions {
        max_cache_bytes: mib_to_bytes(args.max_cache_mb)?,
        prefix_cache_entries: args.prefix_cache_entries,
        cache_page_bytes: kib_to_bytes(args.cache_page_kb)?,
        min_prefix_tokens: args.min_prefix_tokens,
    };

    tracing::info!(model = %args.model, model_id = %model_id, "Loading model");
    let engine = SimpleEngine::load_with_options(&args.model, engine_options)?;

    let mut engines = HashMap::new();
    engines.insert(model_id.clone(), Arc::new(engine));

    let config = ServerConfig {
        models: vec![args.model.clone()],
        host: args.host.clone(),
        port: args.port,
        max_tokens: args.max_tokens,
        api_key: args.api_key.clone(),
        rate_limit: args.rate_limit,
        timeout: args.timeout,
        max_admitted_requests: args.max_admitted_requests.max(1),
        max_queue_size: args.max_queue_size,
        retry_after_seconds: args.retry_after_seconds,
        clear_runtime_cache_after_request: args.clear_runtime_cache_after_request,
    };

    let state: SharedState = Arc::new(AppState {
        engines,
        backpressure: BackpressureController::new(
            config.max_admitted_requests,
            config.max_queue_size,
            config.retry_after_seconds,
        ),
        config: config.clone(),
    });

    let app = mlx_server::build_router(
        state,
        config.timeout,
        config.api_key.clone(),
        config.rate_limit,
    );

    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port)).await?;
    let local_addr = listener.local_addr()?;

    tracing::info!(
        address = %local_addr,
        model_id = %model_id,
        max_cache_mb = args.max_cache_mb,
        max_admitted_requests = config.max_admitted_requests,
        max_queue_size = config.max_queue_size,
        "mlx-serve is listening"
    );

    serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(CliError::Io)?;

    tracing::info!("shutdown complete");
    Ok(())
}

fn run_generate(args: GenerateArgs) -> Result<(), CliError> {
    let engine_options = SimpleEngineOptions {
        max_cache_bytes: mib_to_bytes(args.max_cache_mb)?,
        prefix_cache_entries: args.prefix_cache_entries,
        ..SimpleEngineOptions::default()
    };

    let engine = SimpleEngine::load_with_options(&args.model, engine_options)?;
    let encoding = engine
        .tokenizer()
        .encode(args.prompt.as_str(), false)
        .map_err(|error| CliError::Message(format!("tokenization failed: {error}")))?;

    let sampling = SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };

    let output = engine.generate_with_sampling(
        encoding.get_ids(),
        args.max_tokens,
        sampling,
        &args.stop_sequences,
    )?;

    println!("{}", output.text);
    Ok(())
}

fn run_info(args: InfoArgs) -> Result<(), CliError> {
    let model_dir = model_loader::resolve_model_dir(&args.model)?;
    let architecture = registry::detect_model_type(&model_dir)?;
    let config = read_config_json(&model_dir)?;
    let stats = scan_safetensors(&model_dir)?;

    let kv_bytes_per_token = estimate_kv_bytes_per_token(&config).unwrap_or(0);
    let kv_4k = kv_bytes_per_token.saturating_mul(4096);
    let runtime_estimate = stats.total_tensor_bytes.saturating_add(kv_4k);

    println!("Model: {}", model_id_from_source(&args.model));
    println!("Resolved path: {}", model_dir.display());
    println!("Architecture: {}", architecture);
    println!("Tensor count: {}", stats.tensor_count);
    println!("Parameter count: {}", format_number(stats.parameter_count));
    println!(
        "Weights size: {}",
        format_size(stats.total_tensor_bytes, BINARY)
    );
    println!(
        "Estimated runtime memory (weights + 4k KV): {}",
        format_size(runtime_estimate, BINARY)
    );

    Ok(())
}

fn read_config_json(model_dir: &Path) -> Result<serde_json::Value, CliError> {
    let config_path = model_dir.join("config.json");
    let content = std::fs::read_to_string(config_path)?;
    Ok(serde_json::from_str(&content)?)
}

struct TensorStats {
    tensor_count: usize,
    parameter_count: u128,
    total_tensor_bytes: u64,
}

fn scan_safetensors(model_dir: &Path) -> Result<TensorStats, CliError> {
    let tensor_paths = collect_safetensor_paths(model_dir)?;

    let mut tensor_count = 0usize;
    let mut parameter_count = 0u128;
    let mut total_tensor_bytes = 0u64;

    for path in tensor_paths {
        let file_bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&file_bytes)?;

        for (_, tensor) in tensors.iter() {
            tensor_count = tensor_count.saturating_add(1);

            let count = tensor
                .shape()
                .iter()
                .try_fold(1u128, |acc, dim| acc.checked_mul(*dim as u128))
                .ok_or(CliError::Overflow)?;
            parameter_count = parameter_count
                .checked_add(count)
                .ok_or(CliError::Overflow)?;

            let bytes = u64::try_from(tensor.data().len()).map_err(|_| CliError::Overflow)?;
            total_tensor_bytes = total_tensor_bytes
                .checked_add(bytes)
                .ok_or(CliError::Overflow)?;
        }
    }

    Ok(TensorStats {
        tensor_count,
        parameter_count,
        total_tensor_bytes,
    })
}

fn collect_safetensor_paths(model_dir: &Path) -> Result<Vec<PathBuf>, CliError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)?;
        let index: WeightMapIndex = serde_json::from_str(&raw)?;
        let mut files = BTreeSet::new();
        for value in index.weight_map.values() {
            files.insert(model_dir.join(value));
        }
        return Ok(files.into_iter().collect());
    }

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut from_dir: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    from_dir.sort_unstable();

    if from_dir.is_empty() {
        return Err(CliError::Message(format!(
            "no safetensors files found in {}",
            model_dir.display()
        )));
    }

    Ok(from_dir)
}

fn estimate_kv_bytes_per_token(config: &serde_json::Value) -> Option<u64> {
    let layers = config.get("num_hidden_layers")?.as_u64()?;

    let kv_heads = config
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64)?;

    let head_dim = if let Some(value) = config.get("head_dim").and_then(serde_json::Value::as_u64) {
        value
    } else {
        let hidden = config.get("hidden_size")?.as_u64()?;
        let heads = config.get("num_attention_heads")?.as_u64()?;
        hidden.checked_div(heads)?
    };

    // 2 tensors (K,V) * fp16 bytes.
    let per_layer = kv_heads
        .checked_mul(head_dim)?
        .checked_mul(2)?
        .checked_mul(2)?;
    layers.checked_mul(per_layer)
}

fn model_id_from_source(source: &str) -> String {
    let path = Path::new(source);
    if path.exists() {
        return path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| source.to_owned());
    }

    source.to_owned()
}

fn mib_to_bytes(value: usize) -> Result<usize, CliError> {
    value.checked_mul(MEBIBYTE).ok_or(CliError::Overflow)
}

fn kib_to_bytes(value: usize) -> Result<usize, CliError> {
    value.checked_mul(1024).ok_or(CliError::Overflow)
}

fn format_number(value: u128) -> String {
    let raw = value.to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3);
    for (index, ch) in raw.chars().enumerate() {
        if index > 0 && (raw.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        let mut signal =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
        if let Some(ref mut term) = signal {
            let _ = term.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).compact().init();
}
