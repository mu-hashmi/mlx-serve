# API Reference

`mlx-serve` provides OpenAI-compatible endpoints.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

Anthropic-compatible endpoints:
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

## `POST /v1/chat/completions`

Request (non-streaming):

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "messages": [{"role":"user","content":"Say hello"}],
  "max_tokens": 32,
  "stream": false
}
```

Response includes `choices[0].message.content` and `usage`.

Streaming mode (`"stream": true`) uses SSE with `data:` chunks and a terminal `data: [DONE]` event.

## `POST /v1/completions`

Request:

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "prompt": "Once upon a",
  "max_tokens": 32
}
```

Response includes `choices[0].text` and `usage`.

## `GET /v1/models`

Returns OpenAI-style model list schema with loaded model IDs.

## Backpressure

When the bounded request queue is full, generation routes return:
- HTTP `503 Service Unavailable`
- `Retry-After` header

This behavior is intentional and prevents unbounded-memory overload.
