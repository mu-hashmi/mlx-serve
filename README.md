# mlx-serve

`mlx-serve` is a Rust-native LLM inference server for Apple Silicon that uses MLX for compute and serves models through OpenAI-compatible APIs.

## Quickstart

### 1. Build

```bash
cargo build --release
```

### 2. Serve a model

```bash
./target/release/mlx-serve serve \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --port 8080 \
  --max-cache-mb 4096
```

### 3. Run a one-shot generation

```bash
./target/release/mlx-serve generate \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "The capital of France is" \
  --max-tokens 20
```

### 4. Query model info

```bash
./target/release/mlx-serve info \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

## Supported Models

Current architecture support:
- `llama`
- `mistral`
- `qwen2`
- `qwen3`
- `qwen3_next`

Models are loaded from MLX HuggingFace layout (`config.json` + `*.safetensors` + `tokenizer.json`).

## API Documentation

See [`docs/API.md`](docs/API.md) for endpoint shapes, examples, and compatibility notes.

## Architecture

Workspace crate layout:

```text
mlx-sys -> mlx-core -> mlx-nn -> mlx-models -> mlx-engine -> mlx-server -> mlx-serve
                                       ^
                         mlx-tokenizers|
```

## Runtime Notes

- `--max-admitted-requests` is an admission setting, not decode parallelism; decode runs one request at a time.
- Prompt cache "pages" are logical accounting units for budget/eviction. KV tensors still use contiguous MLX cache arrays.
- `--clear-runtime-cache-after-request` clears MLX/runtime cache after each non-streaming request. This reduces steady-state footprint at the cost of throughput and cache reuse.
- `mlx-nn` is currently a thin re-export layer, while `mlx-tokenizers` owns tokenizer/chat-template integration logic.
- Known limitation: some Qwen3 chat templates rely on MiniJinja methods like `startswith` that are not available in the current renderer path, which can fail request rendering for affected templates.

## Contributing

1. Run the full verification gates before opening a PR:
   - `cargo test --workspace`
   - `cargo clippy --workspace -- -D warnings`
   - `cargo doc --workspace --no-deps`
2. Keep crate boundaries strict (`mlx-sys -> mlx-core -> mlx-nn -> mlx-models -> mlx-engine -> mlx-server -> mlx-serve`).
3. Add/extend integration tests for behavior changes, especially cache isolation and backpressure behavior.
