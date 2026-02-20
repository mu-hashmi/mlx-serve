use axum::{Json, extract::State};
use serde::Serialize;

use crate::{error::ServerError, state::SharedState};

#[derive(Debug, Serialize)]
pub struct MemoryDebugResponse {
    pub active_bytes: usize,
    pub cache_bytes: usize,
    pub baseline_bytes: usize,
    pub prefix_cache_stats: PrefixCacheStats,
    pub engines: Vec<EngineMemoryStats>,
}

#[derive(Debug, Serialize)]
pub struct EngineMemoryStats {
    pub model: String,
    pub baseline_bytes: usize,
    pub cache_bytes: usize,
    pub max_cache_bytes: usize,
    pub prefix_cache: PrefixCacheStats,
}

#[derive(Debug, Serialize)]
pub struct PrefixCacheStats {
    pub active_request_bytes: usize,
    pub prefix_bytes: usize,
    pub total_used_bytes: usize,
    pub total_capacity_bytes: usize,
    pub free_pages: usize,
    pub active_requests: usize,
    pub prefix_entries: usize,
}

#[derive(Debug, Serialize)]
pub struct CacheClearResponse {
    pub cleared_engines: usize,
}

pub async fn memory(
    State(state): State<SharedState>,
) -> Result<Json<MemoryDebugResponse>, ServerError> {
    let mut engines: Vec<_> = state.engines.iter().collect();
    engines.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

    let active_bytes = if let Some((_, engine)) = engines.first() {
        engine.active_memory_bytes().map_err(|error| {
            ServerError::from_engine_with_retry(error, state.config.retry_after_seconds)
        })?
    } else {
        0
    };

    let mut cache_bytes = 0usize;
    let mut baseline_bytes = 0usize;
    let mut aggregate_prefix_stats = PrefixCacheStats {
        active_request_bytes: 0,
        prefix_bytes: 0,
        total_used_bytes: 0,
        total_capacity_bytes: 0,
        free_pages: 0,
        active_requests: 0,
        prefix_entries: 0,
    };
    let mut per_engine = Vec::with_capacity(engines.len());

    for (model, engine) in engines {
        let stats = engine.prefix_cache_stats().map_err(|error| {
            ServerError::from_engine_with_retry(error, state.config.retry_after_seconds)
        })?;
        let model_baseline = engine.memory_baseline_bytes();
        let model_cache_bytes = stats.total_used_bytes;

        baseline_bytes = baseline_bytes.saturating_add(model_baseline);
        cache_bytes = cache_bytes.saturating_add(model_cache_bytes);
        aggregate_prefix_stats.active_request_bytes = aggregate_prefix_stats
            .active_request_bytes
            .saturating_add(stats.active_request_bytes);
        aggregate_prefix_stats.prefix_bytes = aggregate_prefix_stats
            .prefix_bytes
            .saturating_add(stats.prefix_bytes);
        aggregate_prefix_stats.total_used_bytes = aggregate_prefix_stats
            .total_used_bytes
            .saturating_add(stats.total_used_bytes);
        aggregate_prefix_stats.total_capacity_bytes = aggregate_prefix_stats
            .total_capacity_bytes
            .saturating_add(stats.total_capacity_bytes);
        aggregate_prefix_stats.free_pages = aggregate_prefix_stats
            .free_pages
            .saturating_add(stats.free_pages);
        aggregate_prefix_stats.active_requests = aggregate_prefix_stats
            .active_requests
            .saturating_add(stats.active_requests);
        aggregate_prefix_stats.prefix_entries = aggregate_prefix_stats
            .prefix_entries
            .saturating_add(stats.prefix_entries);

        per_engine.push(EngineMemoryStats {
            model: model.clone(),
            baseline_bytes: model_baseline,
            cache_bytes: model_cache_bytes,
            max_cache_bytes: engine.max_cache_bytes(),
            prefix_cache: PrefixCacheStats {
                active_request_bytes: stats.active_request_bytes,
                prefix_bytes: stats.prefix_bytes,
                total_used_bytes: stats.total_used_bytes,
                total_capacity_bytes: stats.total_capacity_bytes,
                free_pages: stats.free_pages,
                active_requests: stats.active_requests,
                prefix_entries: stats.prefix_entries,
            },
        });
    }

    Ok(Json(MemoryDebugResponse {
        active_bytes,
        cache_bytes,
        baseline_bytes,
        prefix_cache_stats: aggregate_prefix_stats,
        engines: per_engine,
    }))
}

pub async fn clear_cache(
    State(state): State<SharedState>,
) -> Result<Json<CacheClearResponse>, ServerError> {
    for engine in state.engines.values() {
        engine.clear_runtime_cache().map_err(|error| {
            ServerError::from_engine_with_retry(error, state.config.retry_after_seconds)
        })?;
    }

    Ok(Json(CacheClearResponse {
        cleared_engines: state.engines.len(),
    }))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    use super::*;

    fn empty_state() -> SharedState {
        std::sync::Arc::new(crate::state::AppState {
            engines: std::collections::HashMap::new(),
            config: crate::config::ServerConfig::default(),
            backpressure: crate::backpressure::BackpressureController::new(1, 1, 1),
        })
    }

    #[tokio::test]
    async fn memory_endpoint_empty_state_returns_zero_totals() {
        let response = memory(State(empty_state())).await.unwrap().into_response();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["active_bytes"], 0);
        assert_eq!(json["cache_bytes"], 0);
        assert_eq!(json["baseline_bytes"], 0);
        assert_eq!(
            json["prefix_cache_stats"],
            serde_json::json!({
                "active_request_bytes": 0,
                "prefix_bytes": 0,
                "total_used_bytes": 0,
                "total_capacity_bytes": 0,
                "free_pages": 0,
                "active_requests": 0,
                "prefix_entries": 0
            })
        );
        assert_eq!(json["engines"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn clear_cache_endpoint_empty_state_is_successful() {
        let response = clear_cache(State(empty_state()))
            .await
            .unwrap()
            .into_response();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["cleared_engines"], 0);
    }
}
