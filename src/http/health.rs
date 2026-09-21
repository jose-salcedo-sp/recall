use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::json;

use crate::ctx::Ctx;

/// Liveness. No dependency checks, so a slow upstream never gets the process killed.
pub async fn healthz() -> Json<serde_json::Value> {
    Json(json!({ "ok": true }))
}

/// Readiness. Checks every dependency with short timeouts and reports per-dependency
/// latency, so a 503 says which one is down rather than just that something is.
pub async fn readyz(State(ctx): State<Arc<Ctx>>) -> (StatusCode, Json<serde_json::Value>) {
    let (index_ms, index_ok) = timed(check_index(&ctx)).await;
    let (embed_ms, embed_ok) = timed(async { ctx.embedder.healthy().await }).await;
    let (sys1_ms, sys1_ok) = timed(async { ctx.system_one.healthy().await }).await;
    let (gen_ms, gen_ok) = timed(async { ctx.generator.healthy().await }).await;

    let ok = index_ok && embed_ok && sys1_ok && gen_ok;
    let body = json!({
        "ok": ok,
        "index_ms": index_ms, "index": index_ok,
        "embed_ms": embed_ms, "embedder": embed_ok,
        "system_one_ms": sys1_ms, "system_one": sys1_ok,
        "generator_ms": gen_ms, "generator": gen_ok,
        "admit_threshold": ctx.cfg.admit_threshold,
        "admit_threshold_calibrated": ctx.cfg.admit_threshold_is_calibrated,
    });

    let status = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}

async fn timed<F: std::future::Future<Output = bool>>(fut: F) -> (u64, bool) {
    let started = Instant::now();
    let ok = fut.await;
    (started.elapsed().as_millis() as u64, ok)
}

async fn check_index(ctx: &Arc<Ctx>) -> bool {
    sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&ctx.pg)
        .await
        .is_ok()
}
