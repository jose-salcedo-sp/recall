use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::{Candidate, Stage};

/// Hybrid retrieval: vector and lexical fused in one round trip.
///
/// The SQL lives in `sql/hybrid_retrieve.sql` and does the reciprocal rank fusion
/// server-side, which is what keeps architecture.md's "one ask, one retrieve" rule
/// true. Doing fusion here would mean two queries and a join in application code.
pub async fn run(
    ctx: &Arc<Ctx>,
    brain_id: Uuid,
    embedding: &[f32],
    question: &str,
    as_of: Option<DateTime<Utc>>,
) -> Result<Vec<Candidate>> {
    let vector = pgvector::Vector::from(embedding.to_vec());

    let rows = ctx
        .with_retry(Stage::Retrieve, ctx.cfg.index_timeout, || {
            let vector = vector.clone();
            async move {
                sqlx::query_as::<_, RetrievedRow>(crate::ctx::HYBRID_RETRIEVE_SQL)
                    .bind(vector)
                    .bind(question)
                    .bind(brain_id)
                    .bind(as_of)
                    .bind(ctx.cfg.retrieve_k)
                    .fetch_all(&ctx.pg)
                    .await
                    .map_err(Into::into)
            }
        })
        .await?;

    tracing::info!(count = rows.len(), k = ctx.cfg.retrieve_k, "retrieved");

    Ok(rows
        .into_iter()
        .map(|r| Candidate {
            id: r.id,
            statement: r.statement,
            text: r.text,
            origin: r.origin,
            grantor_name: r.grantor_name,
            rrf_score: r.rrf_score,
            noul: None,
        })
        .collect())
}

#[derive(sqlx::FromRow)]
struct RetrievedRow {
    id: Uuid,
    statement: String,
    text: String,
    origin: String,
    grantor_name: Option<String>,
    rrf_score: f64,
}
