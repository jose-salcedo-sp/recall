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
/// and the copy in Recall would be the one that drifts.
///
/// The local Compose stack defines the same two functions over its own corpus
/// (`migrations/0002_local_search_shims.sql`), so this is the only retrieval path
/// and it is exercised identically in both environments.
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

    let mut out = Vec::with_capacity(rows.len());
    let mut withheld = 0usize;
    for r in rows {
        // Fail closed on `secret`. This is not a reimplementation of the grant
        // rules — it cannot widen access, only narrow it — but architecture.md
        // states secret is never indexed or cited, and forwarding one to a
        // generator is not a mistake worth risking on an assumption about what
        // the mounted function already filters.
        if r.sensitivity.as_deref() == Some("secret") {
            withheld += 1;
            continue;
        }

        out.push(Candidate {
            // `memory_id` is the memory being cited; `id` is the segment within it.
            id: r.memory_id.unwrap_or(r.id),
            statement: r.memory_statement.unwrap_or_else(|| r.text.clone()),
            text: r.text,
            // The mounted function returns its own origin; trust it over our label.
            origin: r.origin.unwrap_or_else(|| origin.to_string()),
            grantor_name: r.grantor_name,
            source: r.source_name.or(r.source_channel),
            occurred_at: r.occurred_at,
            rrf_score: r.score.unwrap_or(0.0),
            noul: None,
        });
    }

    if withheld > 0 {
        tracing::warn!(withheld, origin, "withheld secret-sensitivity rows");
    }
    Ok(out)
}

/// Shared shape of both search functions.
///
/// The two result sets differ: the personal search returns no origin, grantor,
/// source or sensitivity columns at all. Those are `#[sqlx(default)]` so an absent
/// column is None rather than a decode error, which lets one struct serve both.
#[derive(sqlx::FromRow)]
struct SearchRow {
    id: Uuid,
    text: String,
    #[sqlx(default)]
    memory_id: Option<Uuid>,
    #[sqlx(default)]
    memory_statement: Option<String>,
    #[sqlx(default)]
    score: Option<f64>,
    #[sqlx(default)]
    origin: Option<String>,
    #[sqlx(default)]
    grantor_name: Option<String>,
    #[sqlx(default)]
    source_name: Option<String>,
    #[sqlx(default)]
    source_channel: Option<String>,
    #[sqlx(default)]
    occurred_at: Option<DateTime<Utc>>,
    #[sqlx(default)]
    sensitivity: Option<String>,
}
