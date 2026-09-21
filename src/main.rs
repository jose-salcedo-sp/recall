mod clients;
mod config;
mod ctx;
mod error;
mod http;
mod pipeline;
mod store;
mod types;

use anyhow::Context;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cfg = config::Config::from_env();
    cfg.warn_if_uncalibrated();

    tracing::info!(
        bind = %cfg.bind_addr,
        retrieve_k = cfg.retrieve_k,
        admit_batch_size = cfg.admit_batch_size,
        max_citations = cfg.max_citations,
        max_inflight = cfg.max_inflight,
        "recall starting"
    );

    let bind_addr = cfg.bind_addr.clone();
    let ctx = ctx::Ctx::build(cfg)
        .await
        .context("failed to build service context")?;

    sqlx::migrate!("./migrations")
        .run(&ctx.pg)
        .await
        .context("migrations failed")?;
    tracing::info!("migrations applied");

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;
    tracing::info!(addr = %bind_addr, "listening");

    axum::serve(listener, http::router(ctx))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
