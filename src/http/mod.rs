pub mod ask;
pub mod health;
pub mod index;

use std::sync::Arc;

use axum::routing::{delete, get, post, put};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::ctx::Ctx;

pub fn router(ctx: Arc<Ctx>) -> Router {
    Router::new()
        .route("/healthz", get(health::healthz))
        .route("/readyz", get(health::readyz))
        .route("/v1/ask", post(ask::stream))
        .route("/v1/ask/sync", post(ask::sync))
        .route("/v1/asks/{ask_id}", get(index::get_ask))
        .route("/v1/index/chunks/{id}", put(index::put_chunk))
        .route("/v1/index/chunks/{id}", delete(index::delete_chunk))
        // No global timeout layer: it would cut the SSE stream mid-answer. Timeouts
        // are per stage in `Ctx::with_retry`, plus `ask_timeout` around the pipeline.
        .layer(TraceLayer::new_for_http())
        .with_state(ctx)
}
