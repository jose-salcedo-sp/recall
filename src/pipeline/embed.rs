use std::sync::Arc;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::Stage;

/// Embed the question. One vector, one call.
///
/// plan.md deletes `embed_batch_wait_ms` for v0: micro-batching concurrent asks into a
/// shared forward pass buys nothing at single-user concurrency and costs a component.
/// Add it when measured load justifies it.
pub async fn run(ctx: &Arc<Ctx>, question: &str) -> Result<Vec<f32>> {
    ctx.with_retry(Stage::Embed, ctx.cfg.embed_timeout, || {
        ctx.embedder.embed_one(question)
    })
    .await
}
