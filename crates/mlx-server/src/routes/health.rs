use axum::{Json, extract::State};
use serde::Serialize;

use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub models: Vec<String>,
}

pub async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let mut models: Vec<String> = state.engines.keys().cloned().collect();
    models.sort_unstable();

    Json(HealthResponse {
        status: "ok",
        models,
    })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_returns_ok() {
        let state = std::sync::Arc::new(crate::state::AppState {
            engines: std::collections::HashMap::new(),
            config: crate::config::ServerConfig::default(),
            backpressure: crate::backpressure::BackpressureController::new(1, 1, 1),
        });
        let Json(resp) = health(State(state)).await;
        assert_eq!(resp.status, "ok");
        assert!(resp.models.is_empty());
    }
}
