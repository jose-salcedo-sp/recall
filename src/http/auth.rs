use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::ctx::Ctx;

/// Require the service token issued to Nexus.
///
/// This is a service-to-service boundary, not a user one: Recall never sees an
/// end-user JWT and has no notion of a user or an org. Which brain a caller may read
/// is decided by Nexus and expressed as the `brain_id` in the request body. That is
/// precisely why the browser must not reach this service — anyone who can call it can
/// name any brain.
pub async fn require_service_token(
    State(ctx): State<Arc<Ctx>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = ctx.cfg.service_token.as_deref() else {
        // Startup validation rejects a missing token, so reaching here means the
        // config was mutated out from under us. Fail closed.
        return unauthorized("service token not configured");
    };

    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);

    match presented {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => next.run(req).await,
        Some(_) => unauthorized("invalid service token"),
        None => unauthorized("missing Bearer service token"),
    }
}

fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        axum::Json(json!({ "code": "unauthorized", "message": message })),
    )
        .into_response()
}

/// Compare without leaking the match length through timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::constant_time_eq;

    #[test]
    fn compares_exactly() {
        assert!(constant_time_eq(b"tok", b"tok"));
        assert!(!constant_time_eq(b"tok", b"tox"));
        assert!(!constant_time_eq(b"tok", b"token"), "length must not match");
        assert!(!constant_time_eq(b"", b"x"));
    }
}
