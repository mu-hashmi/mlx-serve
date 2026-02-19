#![allow(clippy::panic, clippy::unwrap_used)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use mlx_engine::simple::{SimpleEngine, SimpleEngineOptions};
use mlx_server::{
    backpressure::BackpressureController,
    build_router,
    config::ServerConfig,
    state::{AppState, SharedState},
};
use serde_json::{Value, json};

const TEST_MODEL_REPO: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("test lock poisoned")
}

struct RunningServer {
    base_url: String,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: tokio::task::JoinHandle<()>,
}

impl RunningServer {
    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let _ = self.join_handle.await;
    }
}

async fn start_server(
    engine: Arc<SimpleEngine>,
    model_id: String,
    max_concurrent_requests: usize,
    max_queue_size: usize,
) -> RunningServer {
    let mut engines = HashMap::new();
    engines.insert(model_id.clone(), engine);

    let config = ServerConfig {
        models: vec![model_id.clone()],
        host: "127.0.0.1".to_owned(),
        port: 0,
        max_tokens: 256,
        api_key: None,
        rate_limit: 0,
        timeout: 300.0,
        max_concurrent_requests,
        max_queue_size,
        retry_after_seconds: 2,
    };

    let state: SharedState = Arc::new(AppState {
        engines,
        backpressure: BackpressureController::new(
            config.max_concurrent_requests,
            config.max_queue_size,
            config.retry_after_seconds,
        ),
        config: config.clone(),
    });

    let app = build_router(state, config.timeout, config.api_key.clone(), config.rate_limit);

    let listener = tokio::net::TcpListener::bind((config.host.as_str(), 0))
        .await
        .expect("failed to bind test server");
    let addr = listener.local_addr().expect("failed to get local address");

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let join_handle = tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await;
    });

    RunningServer {
        base_url: format!("http://{}", addr),
        shutdown_tx: Some(shutdown_tx),
        join_handle,
    }
}

fn chat_payload(model: &str, prompt: &str, max_tokens: u32, stream: bool) -> Value {
    json!({
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": max_tokens,
        "stream": stream
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn check_stage_5_http_endpoints_and_backpressure() {
    let _guard = test_lock();

    let engine_options = SimpleEngineOptions {
        max_cache_bytes: 4 * 1024 * 1024 * 1024,
        prefix_cache_entries: 128,
        ..SimpleEngineOptions::default()
    };
    let engine = Arc::new(
        SimpleEngine::load_with_options(TEST_MODEL_REPO, engine_options)
            .expect("failed to load engine"),
    );
    let model_id = TEST_MODEL_REPO.to_owned();

    // Server for checks 5.1 - 5.6.
    let server = start_server(Arc::clone(&engine), model_id.clone(), 5, 16).await;
    let client = reqwest::Client::new();

    // Check 5.1 — Health endpoint
    let health = client
        .get(format!("{}/health", server.base_url))
        .send()
        .await
        .expect("health request failed");
    assert_eq!(health.status(), reqwest::StatusCode::OK);
    let health_json: Value = health.json().await.expect("health json parse failed");
    let models = health_json
        .get("models")
        .and_then(Value::as_array)
        .expect("health response missing models array");
    assert!(
        models.iter().any(|item| item.as_str() == Some(model_id.as_str())),
        "health response does not contain loaded model id"
    );

    // Check 5.2 — Models endpoint
    let models_resp = client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .await
        .expect("models request failed");
    assert_eq!(models_resp.status(), reqwest::StatusCode::OK);
    let models_json: Value = models_resp.json().await.expect("models json parse failed");
    let model_entries = models_json
        .get("data")
        .and_then(Value::as_array)
        .expect("models response missing data array");
    assert!(
        model_entries
            .iter()
            .any(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id.as_str())),
        "models endpoint missing loaded model id"
    );

    // Check 5.3 — Non-streaming chat completion
    let chat_resp = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&chat_payload(&model_id, "Say hello", 10, false))
        .send()
        .await
        .expect("chat completion request failed");
    assert_eq!(chat_resp.status(), reqwest::StatusCode::OK);
    let chat_json: Value = chat_resp
        .json()
        .await
        .expect("chat completion json parse failed");

    let chat_text = chat_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(!chat_text.is_empty(), "chat completion content is empty");

    let prompt_tokens = chat_json["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = chat_json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(prompt_tokens > 0, "expected prompt_tokens > 0");
    assert!(
        completion_tokens > 0 && completion_tokens <= 10,
        "completion_tokens out of range: {completion_tokens}"
    );

    // Check 5.4 — Streaming chat completion SSE
    let stream_resp = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&chat_payload(&model_id, "Write a short greeting.", 20, true))
        .send()
        .await
        .expect("streaming chat request failed");
    assert_eq!(stream_resp.status(), reqwest::StatusCode::OK);
    let content_type = stream_resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(
        content_type.contains("text/event-stream"),
        "unexpected streaming content-type: {content_type}"
    );

    let stream_body = stream_resp.text().await.expect("failed to read SSE body");
    let mut chunks = Vec::new();
    let mut saw_done = false;
    for line in stream_body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                saw_done = true;
                break;
            }
            chunks.push(data.to_owned());
        }
    }

    assert!(saw_done, "did not receive [DONE] sentinel");
    assert!(
        chunks.len() >= 2,
        "expected at least 2 SSE data chunks, got {}",
        chunks.len()
    );

    let mut combined_text = String::new();
    for chunk in &chunks {
        let json_chunk: Value =
            serde_json::from_str(chunk).expect("failed to parse SSE JSON chunk");
        assert!(
            json_chunk["choices"][0].get("delta").is_some(),
            "SSE chunk missing choices[0].delta"
        );
        if let Some(fragment) = json_chunk["choices"][0]["delta"]["content"].as_str() {
            combined_text.push_str(fragment);
        }
    }
    assert!(
        combined_text.chars().any(char::is_alphabetic),
        "streamed text is not coherent: '{combined_text}'"
    );

    // Check 5.5 — Completions endpoint
    let completion_resp = client
        .post(format!("{}/v1/completions", server.base_url))
        .json(&json!({
            "model": model_id,
            "prompt": "Once upon a",
            "max_tokens": 10
        }))
        .send()
        .await
        .expect("completions request failed");
    assert_eq!(completion_resp.status(), reqwest::StatusCode::OK);
    let completion_json: Value = completion_resp
        .json()
        .await
        .expect("completions json parse failed");
    let completion_text = completion_json["choices"][0]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(!completion_text.is_empty(), "completion text is empty");

    // Check 5.6 — Concurrent requests
    let mut tasks = Vec::new();
    for i in 0..5 {
        let client_clone = client.clone();
        let base_url = server.base_url.clone();
        let model_id_clone = TEST_MODEL_REPO.to_owned();
        tasks.push(tokio::spawn(async move {
            client_clone
                .post(format!("{base_url}/v1/chat/completions"))
                .json(&chat_payload(
                    &model_id_clone,
                    &format!("Reply with word number {i}"),
                    12,
                    false,
                ))
                .send()
                .await
        }));
    }

    for task in tasks {
        let response = task
            .await
            .expect("concurrent task join failed")
            .expect("concurrent request failed");
        assert_eq!(
            response.status(),
            reqwest::StatusCode::OK,
            "expected HTTP 200 for concurrent request"
        );
        let body: Value = response
            .json()
            .await
            .expect("failed to parse concurrent response body");
        assert!(
            body["choices"][0]["message"]["content"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "concurrent response content is empty"
        );
    }

    server.shutdown().await;

    // Separate server instance for strict overload check (5.7).
    let overload_server = start_server(Arc::clone(&engine), TEST_MODEL_REPO.to_owned(), 2, 0).await;

    let mut overload_tasks = Vec::new();
    for _ in 0..5 {
        let client_clone = client.clone();
        let base_url = overload_server.base_url.clone();
        let model_id_clone = TEST_MODEL_REPO.to_owned();
        overload_tasks.push(tokio::spawn(async move {
            client_clone
                .post(format!("{base_url}/v1/chat/completions"))
                .json(&chat_payload(
                    &model_id_clone,
                    "Write a detailed paragraph about Rust ownership.",
                    128,
                    false,
                ))
                .send()
                .await
        }));
    }

    let mut success_count = 0usize;
    let mut overloaded_count = 0usize;
    let mut saw_retry_after = false;

    for task in overload_tasks {
        let response = task
            .await
            .expect("overload task join failed")
            .expect("overload request failed");

        if response.status() == reqwest::StatusCode::OK {
            success_count = success_count.saturating_add(1);
        } else if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            overloaded_count = overloaded_count.saturating_add(1);
            if response.headers().get(reqwest::header::RETRY_AFTER).is_some() {
                saw_retry_after = true;
            }
        }
    }

    assert!(
        success_count >= 2,
        "expected at least 2 successful requests, got {success_count}"
    );
    assert!(
        overloaded_count >= 1,
        "expected at least 1 overloaded request, got {overloaded_count}"
    );
    assert!(saw_retry_after, "expected Retry-After header on 503 response");

    overload_server.shutdown().await;
}
