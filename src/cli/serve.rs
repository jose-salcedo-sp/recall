//! `recall serve`: the OpenAI-compatible API over Postgres and the configured models, until SIGINT or SIGTERM.

use std::{net::SocketAddr, sync::Arc};

use eros::Context;

use crate::{
    clock::{Clock, SystemClock},
    server::{AppState, auth::ApiKeys, router},
    store::runs::PgRunStore,
};

use super::{args::DbArgs, pipeline::PipelineArgs};

/// `recall serve` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Database connection.
    #[command(flatten)]
    pub db: DbArgs,
    /// Socket address to listen on.
    #[arg(long, env = "RECALL_BIND", default_value = "0.0.0.0:8080")]
    pub bind: SocketAddr,
    /// Per-user bearer keys: `key:user_id` pairs, comma-separated.
    #[arg(long, env = "RECALL_API_KEYS", hide_env_values = true)]
    pub api_keys: ApiKeys,
    /// Models, tunables and the access policy, which also decides who reads persisted runs.
    #[command(flatten)]
    pub pipeline: PipelineArgs,
}

/// Runs `recall serve` until SIGINT or SIGTERM, then shuts the HTTP server down gracefully.
///
/// # Errors
///
/// When the database or pipeline cannot be set up, the address cannot be bound, or the server fails.
pub async fn run(args: Args) -> eros::Result<()> {
    let pool = args.db.connect().await?;
    let answerer = args.pipeline.answerer(pool.clone()).await?;
    tracing::info!(plan = ?answerer.plan(), "pipeline assembled");
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let state = AppState {
        authorizer: answerer.authorizer(),
        answerer: Arc::new(answerer),
        runs: Arc::new(PgRunStore::new(pool.clone(), Arc::clone(&clock))),
        keys: Arc::new(args.api_keys),
        clock,
    };
    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    tracing::info!(addr = %listener.local_addr()?, "recall serve listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// As PID 1 in a container the process gets no default SIGTERM handler, so it has to exit on SIGTERM itself.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::warn!(%error, "no SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}
