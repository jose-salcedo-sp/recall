use std::sync::Arc;

use serde_json::json;
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::{RecallError, Result};
use crate::types::Ask;

/// Persist the ask record: question, every candidate with its noul, what was admitted,
/// and the per-stage timings.
///
/// This is the same data three things need — the audit endpoint, the progress
/// tracking, and the Phase 4 calibration set — so it is written once. Rejected
/// candidates are recorded deliberately: a calibration fit needs the negatives.
///
/// Never fails an ask. The answer has already been delivered by the time this runs, so
/// a write failure is logged and swallowed rather than turned into a client error.
pub async fn save_ask(ctx: &Arc<Ctx>, ask: &Ask, error: Option<&str>) {
    let candidates = json!(ask
        .candidates
        .iter()
        .map(|c| {
            let admitted = ask.admitted.iter().find(|a| a.id == c.id);
            json!({
                "id": c.id,
                "noul": admitted.map(|a| a.noul).or(c.noul),
                "admitted": admitted.is_some(),
                "rrf_score": c.rrf_score,
            })
        })
        .collect::<Vec<_>>());

    let admitted_ids = json!(ask.admitted.iter().map(|c| c.id).collect::<Vec<_>>());
    let stages = json!(ask.stages);

    let result = sqlx::query(
        r#"
        INSERT INTO asks (
            ask_id, brain_id, trace_id, question, as_of,
            candidates, admitted_ids, stages, empty, answer, error
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        ON CONFLICT (ask_id) DO UPDATE SET
            candidates = EXCLUDED.candidates,
            admitted_ids = EXCLUDED.admitted_ids,
            stages = EXCLUDED.stages,
            empty = EXCLUDED.empty,
            answer = EXCLUDED.answer,
            error = EXCLUDED.error
        "#,
    )
    .bind(ask.ask_id)
    .bind(ask.brain_id)
    .bind(ask.trace_id)
    .bind(&ask.question)
    .bind(ask.as_of)
    .bind(&candidates)
    .bind(&admitted_ids)
    .bind(&stages)
    .bind(ask.empty)
    .bind(ask.answer.as_deref())
    .bind(error)
    .execute(&ctx.pg)
    .await;

    if let Err(e) = result {
        tracing::error!(ask_id = %ask.ask_id, error = %e, "failed to persist ask record");
    }
}

#[derive(serde::Serialize, sqlx::FromRow)]
pub struct AskAudit {
    pub ask_id: Uuid,
    pub question: String,
    pub candidates: serde_json::Value,
    pub admitted_ids: serde_json::Value,
    pub stages: serde_json::Value,
    pub empty: bool,
}

pub async fn get_ask(ctx: &Arc<Ctx>, ask_id: Uuid) -> Result<AskAudit> {
    sqlx::query_as::<_, AskAudit>(
        "SELECT ask_id, question, candidates, admitted_ids, stages, empty \
         FROM asks WHERE ask_id = $1",
    )
    .bind(ask_id)
    .fetch_optional(&ctx.pg)
    .await?
    .ok_or(RecallError::NotFound)
}
