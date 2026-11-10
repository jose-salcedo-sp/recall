//! Postgres + pgvector persistence: chunk versions ([`chunks`]) and pipeline runs ([`RunStore`], [`runs`]). Schema
//! lives in `migrations/`; queries use runtime-checked `sqlx::query`, not the macros. Retrieval queries live in
//! [`crate::retrieval::hybrid`], not here.

pub mod chunks;
/// In-memory [`RunStore`]s for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes;
pub mod runs;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, migrate::Migrator, postgres::PgPoolOptions};

use crate::{
    domain::{RunId, UserId},
    pipeline::report::RunReport,
};

/// Every migration in `migrations/`, embedded at compile time. [`connect`] runs it; database tests pass it to
/// `#[sqlx::test]`.
pub static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

/// Database connection options.
#[derive(Clone, Debug, clap::Args)]
pub struct DbArgs {
    /// Postgres connection URL.
    #[arg(long, env = "DATABASE_URL")]
    pub database_url: String,
}

/// Connects and applies pending migrations.
///
/// # Errors
///
/// When the database is unreachable or a migration fails.
pub async fn connect(args: &DbArgs) -> eros::Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&args.database_url)
        .await?;
    MIGRATOR.run(&pool).await?;
    Ok(pool)
}

/// A persisted run, as `GET /v1/runs/{id}` returns it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRun {
    /// Who asked.
    pub user: UserId,
    /// When it was saved, set by the store.
    pub created_at: DateTime<Utc>,
    /// The outcome and full trace.
    pub report: RunReport,
}

/// Every run is persisted: answered, declined, and failed.
#[async_trait]
pub trait RunStore: Send + Sync {
    /// Persists `report` under its run id, once.
    ///
    /// # Errors
    ///
    /// When a run with the same id is already saved, or the store fails.
    async fn save(&self, user: &UserId, report: &RunReport) -> eros::Result<()>;

    /// `None` when no run has this id.
    ///
    /// # Errors
    ///
    /// When the store fails or the stored row no longer parses.
    async fn get(&self, id: RunId) -> eros::Result<Option<StoredRun>>;
}
