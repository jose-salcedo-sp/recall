use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use tokio::sync::Semaphore;

use crate::clients::embedder::EmbedderClient;
use crate::clients::generator::GeneratorClient;
use crate::clients::systemone::SystemOneClient;
use crate::config::Config;
use crate::error::{RecallError, Result};
use crate::types::Stage;

// Compiled in so the binary cannot drift from the checked-in SQL.
//
// Recall reads the Nexus corpus only through these two functions; it has no table
// privileges. `hybrid_retrieve.sql` is kept for the local Compose stack and is
// deliberately not referenced on the Nexus path.
pub const SQL_SEARCH_PERSONAL: &str = include_str!("../sql/search_personal.sql");
pub const SQL_SEARCH_MOUNTED: &str = include_str!("../sql/search_mounted.sql");

pub struct Ctx {
    pub cfg: Config,
    pub pg: PgPool,
    pub embedder: EmbedderClient,
    pub system_one: SystemOneClient,
    pub generator: GeneratorClient,
    /// Backpressure. architecture.md returns 503 with Retry-After rather than queuing
    /// streams for seconds; on a CPU classifier the scorer is the real bottleneck.
    inflight: Arc<Semaphore>,
}

impl Ctx {
    pub async fn build(cfg: Config) -> anyhow::Result<Arc<Self>> {
        let pg = PgPoolOptions::new()
            // The pool must not be the thing that limits concurrency before
            // max_inflight does, or backpressure shows up as pool timeouts.
            .max_connections((cfg.max_inflight as u32 * 2).max(5))
            .acquire_timeout(cfg.index_timeout)
            .connect(&cfg.database_url)
            .await?;

        let http = reqwest::Client::builder()
            .pool_max_idle_per_host(cfg.max_inflight.max(4))
            .build()?;

        let inflight = Arc::new(Semaphore::new(cfg.max_inflight));

        Ok(Arc::new(Self {
            embedder: EmbedderClient::new(
                http.clone(),
                cfg.embedder_url.clone(),
                cfg.embedding_model.clone(),
                cfg.embedding_dim,
                std::env::var("EMBEDDER_API_KEY")
                    .or_else(|_| std::env::var("OPENAI_API_KEY"))
                    .ok()
                    .filter(|s| !s.trim().is_empty()),
            ),
            system_one: SystemOneClient::new(http.clone(), cfg.system_one_url.clone()),
            generator: GeneratorClient::new(
                http.clone(),
                cfg.generator_url.clone(),
                cfg.generator_model.clone(),
            ),
            pg,
            inflight,
            cfg,
        }))
    }

    /// Acquire an in-flight slot or refuse fast.
    pub fn try_admit_request(&self) -> Result<tokio::sync::OwnedSemaphorePermit> {
        self.inflight
            .clone()
            .try_acquire_owned()
            .map_err(|_| RecallError::AtCapacity)
    }

    /// One timeout, and one retry only if the failure was transient.
    ///
    /// Deliberately narrow. The queue owns redelivery for worker-backed stages, so an
    /// in-process backoff loop here would only duplicate that, and sleeping through a
    /// long backoff while holding an in-flight slot makes tail latency worse.
    pub async fn with_retry<T, F, Fut>(
        &self,
        stage: Stage,
        timeout: Duration,
        mut op: F,
    ) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        match self.attempt(stage, timeout, op()).await {
            Ok(v) => Ok(v),
            Err(e) if e.is_transient() => {
                tracing::warn!(stage = stage.as_str(), error = %e, "retrying once");
                self.attempt(stage, timeout, op()).await
            }
            Err(e) => Err(e),
        }
    }

    async fn attempt<T, Fut>(&self, stage: Stage, timeout: Duration, fut: Fut) -> Result<T>
    where
        Fut: Future<Output = Result<T>>,
    {
        match tokio::time::timeout(timeout, fut).await {
            Ok(r) => r,
            Err(_) => Err(RecallError::Timeout {
                stage,
                ms: timeout.as_millis() as u64,
            }),
        }
    }
}
