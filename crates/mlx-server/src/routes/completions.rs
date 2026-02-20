use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use tokio_stream::Stream;

use crate::{
    backpressure::RunPermit,
    error::ServerError,
    state::SharedState,
    types::openai::{
        CompletionChoice, CompletionChunk, CompletionChunkChoice, CompletionRequest,
        CompletionResponse, CompletionUsage, StopSequence,
    },
};

pub async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.prompt.is_empty() {
        return Err(ServerError::BadRequest(
            "prompt must not be empty".to_owned(),
        ));
    }

    let queue_ticket = state.queue_request()?;
    let permit = queue_ticket.wait_for_run().await?;

    if req.stream == Some(true) {
        let stream = completions_stream(state, req, permit)?;
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        let response = completions_non_streaming(state, req, permit).await?;
        Ok(Json(response).into_response())
    }
}

async fn completions_non_streaming(
    state: SharedState,
    req: CompletionRequest,
    _permit: RunPermit,
) -> Result<CompletionResponse, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(1.0);
    let stop_sequences = StopSequence::extract(req.stop);

    let encoding = engine
        .tokenizer()
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    let engine_for_generate = std::sync::Arc::clone(&engine);
    let output = tokio::task::spawn_blocking(move || {
        engine_for_generate.generate(
            &prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            &stop_sequences,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(|error| {
        ServerError::from_engine_with_retry(error, state.config.retry_after_seconds)
    })?;

    if state.config.clear_runtime_cache_after_request {
        engine.clear_runtime_cache().map_err(|error| {
            ServerError::from_engine_with_retry(error, state.config.retry_after_seconds)
        })?;
    }

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    Ok(CompletionResponse {
        id: request_id,
        object: "text_completion",
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: output.text,
            finish_reason: output.finish_reason,
        }],
        usage: CompletionUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
}

fn completions_stream(
    state: SharedState,
    req: CompletionRequest,
    permit: RunPermit,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(1.0);
    let stop_sequences = StopSequence::extract(req.stop);

    let encoding = engine
        .tokenizer()
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model = req.model;

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming(
            &prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            &stop_sequences,
            tx,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let stream = async_stream::stream! {
        let _permit = permit;
        while let Some(output) = rx.recv().await {
            let chunk = CompletionChunk {
                id: request_id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: output.new_text,
                    finish_reason: output.finish_reason,
                }],
            };
            match serde_json::to_string(&chunk) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(stream)
}
