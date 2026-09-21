use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::clients::embedder::EMBEDDING_DIM;
use crate::ctx::Ctx;
use crate::error::{RecallError, Result};
use crate::store::AskAudit;

/// `GET /v1/asks/{ask_id}` — the admission log. Not a user surface; this is what the
/// calibration work in Phase 4 is fitted on.
pub async fn get_ask(
    State(ctx): State<Arc<Ctx>>,
    Path(ask_id): Path<Uuid>,
) -> Result<Json<AskAudit>> {
    crate::store::get_ask(&ctx, ask_id).await.map(Json)
}

#[derive(Deserialize)]
pub struct UpsertChunk {
    pub brain_id: Uuid,
    pub text: String,
    pub statement: String,
    /// Optional: when absent, Recall embeds the text itself.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub origin: String,
    #[serde(default)]
    pub grantor_brain_id: Option<Uuid>,
    #[serde(default)]
    pub grantor_name: Option<String>,
    #[serde(default = "default_sensitivity")]
    pub sensitivity: String,
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default)]
    pub valid_to: Option<DateTime<Utc>>,
    #[serde(default = "default_state")]
    pub state: String,
}

fn default_sensitivity() -> String {
    "normal".into()
}
fn default_state() -> String {
    "active".into()
}

impl UpsertChunk {
    /// The mount invariants from architecture.md, enforced here rather than only by
    /// the DB check constraints so the caller gets a useful message.
    fn validate(&self) -> Result<()> {
        if self.sensitivity == "secret" {
            return Err(RecallError::BadRequest(
                "sensitivity 'secret' is never indexed".into(),
            ));
        }
        if self.state == "pending" {
            return Err(RecallError::BadRequest(
                "state 'pending' is never indexed".into(),
            ));
        }
        if !matches!(self.sensitivity.as_str(), "normal" | "sensitive") {
            return Err(RecallError::BadRequest(format!(
                "unknown sensitivity '{}'",
                self.sensitivity
            )));
        }
        if !matches!(self.origin.as_str(), "personal" | "granted") {
            return Err(RecallError::BadRequest(format!(
                "unknown origin '{}'",
                self.origin
            )));
        }
        if self.origin == "granted" && self.grantor_name.is_none() {
            return Err(RecallError::BadRequest(
                "granted chunks must carry a grantor_name".into(),
            ));
        }
        if self.text.trim().is_empty() || self.statement.trim().is_empty() {
            return Err(RecallError::BadRequest(
                "text and statement must be non-empty".into(),
            ));
        }
        if let Some(e) = &self.embedding {
            if e.len() != EMBEDDING_DIM {
                return Err(RecallError::BadRequest(format!(
                    "embedding must have {EMBEDDING_DIM} dimensions, got {}",
                    e.len()
                )));
            }
        }
        Ok(())
    }
}

/// `PUT /v1/index/chunks/{id}` — upsert a chunk.
pub async fn put_chunk(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<Uuid>,
    Json(body): Json<UpsertChunk>,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    body.validate()?;

    let embedding = match body.embedding {
        Some(e) => e,
        None => {
            let combined = format!("{} {}", body.statement, body.text);
            ctx.embedder.embed_one(&combined).await?
        }
    };

    sqlx::query(
        r#"
        INSERT INTO chunks (
            id, brain_id, text, statement, embedding, origin,
            grantor_brain_id, grantor_name, sensitivity, valid_from, valid_to, state
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        ON CONFLICT (id) DO UPDATE SET
            brain_id = EXCLUDED.brain_id,
            text = EXCLUDED.text,
            statement = EXCLUDED.statement,
            embedding = EXCLUDED.embedding,
            origin = EXCLUDED.origin,
            grantor_brain_id = EXCLUDED.grantor_brain_id,
            grantor_name = EXCLUDED.grantor_name,
            sensitivity = EXCLUDED.sensitivity,
            valid_from = EXCLUDED.valid_from,
            valid_to = EXCLUDED.valid_to,
            state = EXCLUDED.state,
            updated_at = now()
        "#,
    )
    .bind(id)
    .bind(body.brain_id)
    .bind(&body.text)
    .bind(&body.statement)
    .bind(pgvector::Vector::from(embedding))
    .bind(&body.origin)
    .bind(body.grantor_brain_id)
    .bind(body.grantor_name.as_deref())
    .bind(&body.sensitivity)
    .bind(body.valid_from)
    .bind(body.valid_to)
    .bind(&body.state)
    .execute(&ctx.pg)
    .await?;

    Ok((StatusCode::OK, Json(json!({ "id": id, "ok": true }))))
}

/// `DELETE /v1/index/chunks/{id}`
pub async fn delete_chunk(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<Uuid>,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    let done = sqlx::query("DELETE FROM chunks WHERE id = $1")
        .bind(id)
        .execute(&ctx.pg)
        .await?;

    Ok((
        StatusCode::OK,
        Json(json!({ "id": id, "deleted": done.rows_affected() })),
    ))
}
