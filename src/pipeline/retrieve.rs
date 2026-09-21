use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::{Candidate, Stage};

/// Retrieval against the Nexus corpus.
///
/// Recall does not query tables. It connects as `recall_search`, which may execute
/// exactly two functions and read nothing directly, and those functions own the parts
/// of this that are easy to get subtly wrong:
///
/// - `hybrid_search_brain` — the brain's own memories, returned as `origin = personal`
/// - `search_mounted_for_brain` — memories granted to this brain by others, returned
///   as `origin = granted`
///
/// Grant semantics, rank fusion and slice rules all live behind those functions.
/// Reimplementing any of them here would mean two definitions of who may see what,
/// and the copy in Recall would be the one that drifts. `sql/hybrid_retrieve.sql` is
/// retained only for the local Compose stack and is not used on this path.
pub async fn run(
    ctx: &Arc<Ctx>,
    brain_id: Uuid,
    embedding: &[f32],
    question: &str,
    as_of: Option<DateTime<Utc>>,
) -> Result<Vec<Candidate>> {
    let vector = pgvector::Vector::from(embedding.to_vec());
    let k = ctx.cfg.retrieve_k;

    // Both searches are independent reads, so overlap them rather than paying two
    // round trips in series.
    let (personal, granted) = tokio::try_join!(
        search(ctx, crate::ctx::SQL_SEARCH_PERSONAL, brain_id, &vector, question, as_of, k, "personal"),
        search(ctx, crate::ctx::SQL_SEARCH_MOUNTED, brain_id, &vector, question, as_of, k, "granted"),
    )?;

    tracing::info!(
        personal = personal.len(),
        granted = granted.len(),
        k,
        "retrieved from nexus"
    );

    // Concatenate and cap. The two functions rank within their own result sets and
    // their scores are not on a comparable scale, so there is nothing meaningful to
    // sort across them here — admission is what orders the merged set, by noul.
    let mut all = personal;
    all.extend(granted);
    all.truncate(k as usize);
    Ok(all)
}

#[allow(clippy::too_many_arguments)]
async fn search(
    ctx: &Arc<Ctx>,
    sql: &'static str,
    brain_id: Uuid,
    vector: &pgvector::Vector,
    question: &str,
    as_of: Option<DateTime<Utc>>,
    k: i32,
    origin: &'static str,
) -> Result<Vec<Candidate>> {
    let rows = ctx
        .with_retry(Stage::Retrieve, ctx.cfg.index_timeout, || {
            let vector = vector.clone();
            async move {
                sqlx::query_as::<_, SearchRow>(sql)
                    .bind(brain_id)
                    .bind(question)
                    .bind(vector)
                    .bind(as_of)
                    .bind(k)
                    .fetch_all(&ctx.pg)
                    .await
                    .map_err(Into::into)
            }
        })
        .await?;

    Ok(rows
        .into_iter()
        .map(|r| Candidate {
            id: r.id,
            statement: r.memory_statement,
            text: r.text,
            origin: origin.to_string(),
            grantor_name: r.grantor_name,
            source: r.source,
            occurred_at: r.occurred_at,
            rrf_score: r.score.unwrap_or(0.0),
            noul: None,
        })
        .collect())
}

/// Shared shape of both search functions. Optional columns are tolerated as NULL so
/// a difference between the personal and mounted result sets is not a hard failure.
#[derive(sqlx::FromRow)]
struct SearchRow {
    id: Uuid,
    memory_statement: String,
    text: String,
    #[sqlx(default)]
    grantor_name: Option<String>,
    #[sqlx(default)]
    source: Option<String>,
    #[sqlx(default)]
    occurred_at: Option<DateTime<Utc>>,
    #[sqlx(default)]
    score: Option<f64>,
}
