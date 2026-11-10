//! `RunStore` over the `pipeline_runs` table.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use eros::Context;
use sqlx::{PgPool, Row, types::Json};

use crate::{
    domain::{CategoryKind, Question, RunId, UserId},
    pipeline::{
        report::{Outcome, RunReport},
        trace::Trace,
    },
};

use super::{RunStore, StoredRun};

/// The production [`RunStore`].
pub struct PgRunStore {
    pool: PgPool,
}

impl PgRunStore {
    /// Wraps a pool whose migrations have run.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// The `pipeline_runs.status` value for an outcome.
pub fn status(outcome: &Outcome) -> &'static str {
    match outcome {
        Outcome::Answered(_) => "answered",
        Outcome::Declined { .. } => "declined",
        Outcome::Failed { .. } => "failed",
    }
}

#[async_trait]
impl RunStore for PgRunStore {
    async fn save(&self, user: &UserId, report: &RunReport) -> eros::Result<()> {
        sqlx::query(
            "INSERT INTO pipeline_runs (id, user_id, question, category, status, outcome, trace)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(report.id.as_uuid())
        .bind(&user.0)
        .bind(report.question.as_str())
        .bind(report.category.as_ref().map(CategoryKind::as_str))
        .bind(status(&report.outcome))
        .bind(Json(&report.outcome))
        .bind(Json(&report.trace))
        .execute(&self.pool)
        .await
        .with_context(|| format!("saving run {}", report.id))?;
        Ok(())
    }

    async fn get(&self, id: RunId) -> eros::Result<Option<StoredRun>> {
        let row = sqlx::query(
            "SELECT user_id, question, category, outcome, trace, created_at
             FROM pipeline_runs WHERE id = $1",
        )
        .bind(id.as_uuid())
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("loading run {id}"))?;
        let Some(row) = row else {
            return Ok(None);
        };

        let category: Option<String> = row.try_get("category")?;
        let Json(outcome): Json<Outcome> = row.try_get("outcome")?;
        let Json(trace): Json<Trace> = row.try_get("trace")?;
        let created_at: DateTime<Utc> = row.try_get("created_at")?;
        Ok(Some(StoredRun {
            user: UserId(row.try_get("user_id")?),
            created_at,
            report: RunReport {
                id,
                question: Question::new(row.try_get::<String, _>("question")?),
                category: category
                    .as_deref()
                    .map(CategoryKind::try_from)
                    .transpose()?,
                outcome,
                trace,
            },
        }))
    }
}
