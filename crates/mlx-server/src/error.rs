use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// JSON error response body (OpenAI-compatible).
#[derive(Debug, Serialize)]
#[allow(missing_docs)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Individual error detail within an [`ErrorResponse`].
#[derive(Debug, Serialize)]
#[allow(missing_docs)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

/// Server error types.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum ServerError {
    #[error("Engine error: {0}")]
    Engine(#[from] mlx_engine::error::EngineError),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Server overloaded")]
    Overloaded { retry_after_seconds: u64 },
}

const DEFAULT_RETRY_AFTER_SECONDS: u64 = 1;

fn is_engine_cache_admission_error(error: &mlx_engine::error::EngineError) -> bool {
    match error {
        mlx_engine::error::EngineError::Generation(message) => {
            let message = message.to_ascii_lowercase();
            message.contains("cache request requires")
                || message.contains("cache pressure")
                || message.contains("insufficient free pages")
        }
        _ => false,
    }
}

impl ServerError {
    /// Convert an engine error while preserving backpressure semantics.
    pub fn from_engine_with_retry(
        error: mlx_engine::error::EngineError,
        retry_after_seconds: u64,
    ) -> Self {
        if is_engine_cache_admission_error(&error) {
            Self::Overloaded {
                retry_after_seconds,
            }
        } else {
            Self::Engine(error)
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, message, retry_after) = match &self {
            ServerError::Engine(e) if is_engine_cache_admission_error(e) => {
                tracing::warn!(error = %e, "Engine cache admission overloaded");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_overloaded",
                    "Server is overloaded, retry later".to_owned(),
                    Some(DEFAULT_RETRY_AFTER_SECONDS),
                )
            }
            ServerError::Engine(e) => {
                tracing::error!(error = %e, "Engine error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "Internal server error".to_owned(),
                    None,
                )
            }
            ServerError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
                None,
            ),
            ServerError::ModelNotFound(model) => (
                StatusCode::NOT_FOUND,
                "model_not_found",
                format!("Model '{model}' is not loaded"),
                None,
            ),
            ServerError::InternalError(msg) => {
                tracing::error!(error = %msg, "Internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "Internal server error".to_owned(),
                    None,
                )
            }
            ServerError::Overloaded {
                retry_after_seconds,
            } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_overloaded",
                "Server is overloaded, retry later".to_owned(),
                Some(*retry_after_seconds),
            ),
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message,
                r#type: error_type.to_owned(),
                code: None,
            },
        });

        let mut response = (status, body).into_response();
        if let Some(value) = retry_after
            && let Ok(header_value) = HeaderValue::from_str(&value.to_string())
        {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header_value);
        }
        response
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    async fn response_status_and_body(resp: Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        (status, body)
    }

    fn retry_after_value(resp: &Response) -> Option<String> {
        resp.headers()
            .get(header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    }

    /// Asserts that the given error produces a 500 with a masked message
    /// that does not contain `leaked_detail`.
    async fn assert_masked_500(error: ServerError, leaked_detail: &str) {
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap();
        assert_eq!(message, "Internal server error");
        assert!(
            !message.contains(leaked_detail),
            "Internal error detail leaked: {leaked_detail}"
        );
        assert_eq!(body["error"]["type"].as_str().unwrap(), "server_error");
    }

    #[tokio::test]
    async fn test_engine_error_returns_500_with_masked_message() {
        let engine_err =
            mlx_engine::error::EngineError::Generation("sensitive internal details".to_owned());
        assert_masked_500(
            ServerError::Engine(engine_err),
            "sensitive internal details",
        )
        .await;
    }

    async fn assert_bad_request(msg: &str) {
        let error = ServerError::BadRequest(msg.to_owned());
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["message"].as_str().unwrap(), msg);
        assert_eq!(
            body["error"]["type"].as_str().unwrap(),
            "invalid_request_error"
        );
    }

    #[tokio::test]
    async fn test_bad_request_returns_400_with_actual_message() {
        assert_bad_request("missing field: model").await;
    }

    #[tokio::test]
    async fn test_internal_error_returns_500_with_masked_message() {
        assert_masked_500(
            ServerError::InternalError("disk full".to_owned()),
            "disk full",
        )
        .await;
    }

    #[tokio::test]
    async fn test_error_code_field_is_null() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();
        let (_, body) = response_status_and_body(resp).await;

        assert!(body["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn test_engine_tokenization_error_masked() {
        let engine_err = mlx_engine::error::EngineError::Tokenization(
            "tokenizer failed on byte 0xFF".to_owned(),
        );
        assert_masked_500(ServerError::Engine(engine_err), "0xFF").await;
    }

    #[tokio::test]
    async fn test_engine_template_error_masked() {
        let engine_err =
            mlx_engine::error::EngineError::Template("template parse failed".to_owned());
        assert_masked_500(ServerError::Engine(engine_err), "template parse failed").await;
    }

    #[tokio::test]
    async fn test_engine_cache_admission_error_maps_to_503_with_retry_after() {
        let engine_err = mlx_engine::error::EngineError::Generation(
            "cache pressure too high: need 10 pages and have 2 free pages".to_owned(),
        );
        let resp = ServerError::Engine(engine_err).into_response();
        let retry_after = retry_after_value(&resp);
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            body["error"]["message"].as_str().unwrap(),
            "Server is overloaded, retry later"
        );
        assert_eq!(body["error"]["type"].as_str().unwrap(), "server_overloaded");
        assert_eq!(retry_after.as_deref(), Some("1"));
    }

    #[tokio::test]
    async fn test_from_engine_with_retry_uses_configured_retry_after_seconds() {
        let engine_err = mlx_engine::error::EngineError::Generation(
            "cache request requires 123 bytes, but cache budget is 100 bytes".to_owned(),
        );
        let resp = ServerError::from_engine_with_retry(engine_err, 7).into_response();
        let retry_after = retry_after_value(&resp);
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            body["error"]["message"].as_str().unwrap(),
            "Server is overloaded, retry later"
        );
        assert_eq!(body["error"]["type"].as_str().unwrap(), "server_overloaded");
        assert_eq!(retry_after.as_deref(), Some("7"));
    }

    #[tokio::test]
    async fn test_bad_request_with_empty_message() {
        assert_bad_request("").await;
    }

    #[tokio::test]
    async fn test_bad_request_with_very_long_message() {
        let long_msg = "x".repeat(2000);
        assert_bad_request(&long_msg).await;
    }

    #[tokio::test]
    async fn test_internal_error_with_empty_message_still_masked() {
        let resp = ServerError::InternalError(String::new()).into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body["error"]["message"].as_str().unwrap(),
            "Internal server error"
        );
    }

    #[tokio::test]
    async fn test_error_response_json_structure() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();
        let (_, body) = response_status_and_body(resp).await;

        assert!(body.get("error").is_some());
        let error_obj = body.get("error").unwrap();
        assert!(error_obj.get("message").is_some());
        assert!(error_obj.get("type").is_some());
        assert!(error_obj.get("code").is_some());
    }

    #[tokio::test]
    async fn test_error_response_content_type_is_json() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();

        let content_type = resp
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap().to_owned())
            .unwrap_or_default();
        assert!(
            content_type.contains("application/json"),
            "Expected application/json, got: {content_type}"
        );
    }
}
