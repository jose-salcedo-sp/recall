use std::sync::Arc;

use futures::stream::BoxStream;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::AdmittedCitations;

/// Non-streaming generation for `/v1/ask/sync`.
///
/// Takes `AdmittedCitations`, which cannot be empty — that is the type-level guarantee
/// that this stage is unreachable without admitted evidence.
pub async fn run(
    ctx: &Arc<Ctx>,
    question: &str,
    admitted: &AdmittedCitations,
) -> Result<(String, serde_json::Value)> {
    tracing::info!(citations = admitted.len(), "generating");
    ctx.generator.complete(question, admitted).await
}

pub async fn run_chitchat(ctx: &Arc<Ctx>, question: &str) -> Result<(String, serde_json::Value)> {
    tracing::info!("generating chitchat");
    ctx.generator.complete_chitchat(question).await
}

/// Streaming generation for `/v1/ask`.
///
/// No retry wrapper: once a token has reached the client, a retry would duplicate
/// visible output. architecture.md's rule is an `error` event after any tokens sent.
pub async fn stream(
    ctx: &Arc<Ctx>,
    question: &str,
    admitted: &AdmittedCitations,
) -> Result<BoxStream<'static, Result<String>>> {
    tracing::info!(citations = admitted.len(), "streaming generation");
    ctx.generator.stream(question, admitted).await
}

pub async fn stream_chitchat(
    ctx: &Arc<Ctx>,
    question: &str,
) -> Result<BoxStream<'static, Result<String>>> {
    tracing::info!("streaming chitchat");
    ctx.generator.stream_chitchat(question).await
}
