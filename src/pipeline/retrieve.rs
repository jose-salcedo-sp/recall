use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::Result;
use crate::types::{Candidate, Stage};

/// Retrieval against the Nexus corpus via the two search functions, then RRF in
/// Rust. The functions' scores are not comparable; ranks are.
pub async fn run(
    ctx: &Arc<Ctx>,
    ask_id: Uuid,
    brain_id: Uuid,
    embedding: &[f32],
    question: &str,
    as_of: Option<DateTime<Utc>>,
) -> Result<Vec<Candidate>> {
    let vector = pgvector::Vector::from(embedding.to_vec());
    let k = ctx.cfg.retrieve_k;

    let (personal, granted) = tokio::try_join!(
        search(
            ctx,
            crate::ctx::SQL_SEARCH_PERSONAL,
            brain_id,
            &vector,
            question,
            as_of,
            k,
            "personal"
        ),
        search(
            ctx,
            crate::ctx::SQL_SEARCH_MOUNTED,
            brain_id,
            &vector,
            question,
            as_of,
            k,
            "granted"
        ),
    )?;

    tracing::info!(
        ask_id = %ask_id,
        personal = personal.len(),
        granted = granted.len(),
        k,
        "retrieved from nexus"
    );

    Ok(rrf_merge(personal, granted, k as usize))
}

/// Reciprocal rank fusion: `score(d) = Σ 1/(60 + rank_i(d))`. Dedup by memory_id.
pub(crate) fn rrf_merge(
    personal: Vec<Candidate>,
    granted: Vec<Candidate>,
    k: usize,
) -> Vec<Candidate> {
    let mut score: HashMap<Uuid, f64> = HashMap::new();
    let mut keep: HashMap<Uuid, Candidate> = HashMap::new();
    for (origin, list) in [("personal", personal), ("granted", granted)] {
        let _ = origin;
        for (i, c) in list.into_iter().enumerate() {
            *score.entry(c.id).or_insert(0.0) += 1.0 / (60.0 + i as f64 + 1.0);
            keep.entry(c.id).or_insert(c);
        }
    }
    let mut ids: Vec<Uuid> = keep.keys().copied().collect();
    ids.sort_by(|a, b| score[b].total_cmp(&score[a]).then_with(|| a.cmp(b)));
    ids.truncate(k);
    ids.into_iter()
        .map(|id| {
            let mut c = keep.remove(&id).expect("id from keep");
            c.rrf_score = score[&id];
            c
        })
        .collect()
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
        if r.sensitivity.as_deref() == Some("secret") || r.state.as_deref() == Some("pending") {
            withheld += 1;
            continue;
        }

        out.push(Candidate {
            id: r.memory_id.unwrap_or(r.id),
            statement: r.memory_statement.unwrap_or_else(|| r.text.clone()),
            text: r.text,
            origin: r.origin.unwrap_or_else(|| origin.to_string()),
            grantor_name: r.grantor_name,
            source: r.source_name.or(r.source_channel),
            occurred_at: r.occurred_at,
            rrf_score: r.score.unwrap_or(0.0),
            noul: None,
            injection: None,
            contradicts: None,
            relevant: None,
            evidence: None,
            route: None,
        });
    }

    if withheld > 0 {
        tracing::warn!(withheld, origin, "withheld secret-sensitivity rows");
    }
    Ok(out)
}

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
    #[sqlx(default)]
    state: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: u128, origin: &str) -> Candidate {
        Candidate {
            id: Uuid::from_u128(id),
            statement: format!("{origin}-{id}"),
            text: format!("{origin}-{id}"),
            origin: origin.into(),
            grantor_name: None,
            source: None,
            occurred_at: None,
            rrf_score: 0.0,
            noul: None,
            injection: None,
            contradicts: None,
            relevant: None,
            evidence: None,
            route: None,
        }
    }

    #[test]
    fn rrf_keeps_granted_lexical_hit_instead_of_truncating_personal() {
        let personal: Vec<_> = (1..=32).map(|i| cand(i, "personal")).collect();
        let granted = vec![cand(99, "granted")];
        let out = rrf_merge(personal, granted, 32);
        assert!(
            out.iter().any(|c| c.id == Uuid::from_u128(99)),
            "granted rank-1 must survive fusion; concat+truncate dropped it"
        );
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn rrf_dedups_by_memory_id_and_caps() {
        let a = cand(1, "personal");
        let dup = cand(1, "granted");
        let out = rrf_merge(vec![a], vec![dup], 4);
        assert_eq!(out.len(), 1);
        assert!(out[0].rrf_score > 1.0 / 61.0);
    }

    #[test]
    fn secret_and_pending_never_become_candidates() {
        let secret = SearchRow {
            id: Uuid::from_u128(1),
            text: "x".into(),
            memory_id: None,
            memory_statement: None,
            score: None,
            origin: None,
            grantor_name: None,
            source_name: None,
            source_channel: None,
            occurred_at: None,
            sensitivity: Some("secret".into()),
            state: None,
        };
        let pending = SearchRow {
            id: Uuid::from_u128(2),
            text: "y".into(),
            memory_id: None,
            memory_statement: None,
            score: None,
            origin: None,
            grantor_name: None,
            source_name: None,
            source_channel: None,
            occurred_at: None,
            sensitivity: None,
            state: Some("pending".into()),
        };
        assert_eq!(secret.sensitivity.as_deref(), Some("secret"));
        assert_eq!(pending.state.as_deref(), Some("pending"));
    }
}
