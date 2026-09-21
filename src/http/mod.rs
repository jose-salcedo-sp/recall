pub mod ask;
pub mod auth;
pub mod health;
pub mod index;

use std::sync::Arc;

use axum::routing::{delete, get, post, put};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::ctx::Ctx;

pub fn router(ctx: Arc<Ctx>) -> Router {
    // Everything that can read the corpus sits behind the service token. Health does
    // not, so a load balancer can probe without holding a credential.
    let mut protected = Router::new()
        .route("/v1/ask", post(ask::stream))
        .route("/v1/ask/sync", post(ask::sync))
        .route("/v1/asks/{ask_id}", get(index::get_ask));

    // The corpus belongs to Nexus on this deployment; Recall reads it through two
    // functions and must never write it. These routes exist only for a
    // Recall-owned index, so they are absent unless that is explicitly enabled.
    if ctx.cfg.index_writes_enabled {
        protected = protected
            .route("/v1/index/chunks/{id}", put(index::put_chunk))
            .route("/v1/index/chunks/{id}", delete(index::delete_chunk));
    }

    let protected = protected.layer(axum::middleware::from_fn_with_state(
        ctx.clone(),
        auth::require_service_token,
    ));

    Router::new()
        .route("/healthz", get(health::healthz))
        .route("/readyz", get(health::readyz))
        .merge(protected)
        // No global timeout layer: it would cut the SSE stream mid-answer. Timeouts
        // are per stage in `Ctx::with_retry`, plus `ask_timeout` around the pipeline.
        .layer(TraceLayer::new_for_http())
        .with_state(ctx)
}
