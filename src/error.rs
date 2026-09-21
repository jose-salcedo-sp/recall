use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::types::Stage;

#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("{stage} timed out after {ms}ms")]
    Timeout { stage: Stage, ms: u64 },

    #[error("{stage} upstream failed: {source}")]
    Upstream {
        stage: Stage,
        #[source]
        source: reqwest::Error,
    },

    /// An upstream answered, but with a status or body we cannot use.
    #[error("{stage} returned {status}: {body}")]
    UpstreamStatus {
        stage: Stage,
        status: u16,
        body: String,
    },

    #[error("index query failed: {0}")]
    Index(#[from] sqlx::Error),

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error("at capacity")]
    AtCapacity,

    #[error("ask not found")]
    NotFound,
}

impl RecallError {
    /// Transient errors are worth one retry. Permanent ones are not.
    ///
    /// This single classification feeds both the retry predicate and the decision of
    /// what to report, so the two can never disagree.
    pub fn is_transient(&self) -> bool {
        match self {
            RecallError::Timeout { .. } => true,
            RecallError::Upstream { source, .. } => {
                source.is_timeout() || source.is_connect() || source.is_request()
            }
            RecallError::UpstreamStatus { status, .. } => *status >= 500 || *status == 429,
            RecallError::Index(e) => matches!(
                e,
                sqlx::Error::PoolTimedOut | sqlx::Error::Io(_) | sqlx::Error::PoolClosed
            ),
            RecallError::BadRequest(_) | RecallError::AtCapacity | RecallError::NotFound => false,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            RecallError::Timeout { .. } => "timeout",
            RecallError::Upstream { .. } | RecallError::UpstreamStatus { .. } => "upstream_error",
            RecallError::Index(_) => "index_error",
            RecallError::BadRequest(_) => "bad_request",
            RecallError::AtCapacity => "at_capacity",
            RecallError::NotFound => "not_found",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            RecallError::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            RecallError::Upstream { .. } | RecallError::UpstreamStatus { .. } => {
                StatusCode::BAD_GATEWAY
            }
            RecallError::Index(_) => StatusCode::INTERNAL_SERVER_ERROR,
            RecallError::BadRequest(_) => StatusCode::BAD_REQUEST,
            RecallError::AtCapacity => StatusCode::SERVICE_UNAVAILABLE,
            RecallError::NotFound => StatusCode::NOT_FOUND,
        }
    }
}

impl IntoResponse for RecallError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = json!({ "code": self.code(), "message": self.to_string() });

        // Backpressure has to tell the caller when to come back, or clients hammer.
        if matches!(self, RecallError::AtCapacity) {
            return (status, [("Retry-After", "2")], axum::Json(body)).into_response();
        }
        (status, axum::Json(body)).into_response()
    }
}

pub type Result<T> = std::result::Result<T, RecallError>;
