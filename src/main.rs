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
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let json = std::env::var("LOG_FORMAT").as_deref() == Ok("json");
    let reg = tracing_subscriber::registry().with(filter);
    // JSON is what `recall-dash` reads; human format stays the default.
    if json {
        reg.with(tracing_subscriber::fmt::layer().json().flatten_event(true))
            .init();
    } else {
        reg.with(tracing_subscriber::fmt::layer()).init();
    }

    let cfg = config::Config::from_env();
    cfg.validate()?;
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

    // Migrations only where Recall owns the database. Against Nexus this would try
    // to create the corpus tables in someone else's project, which is exactly the
    // thing that must not happen — and `recall_search` could not do it anyway.
    if ctx.cfg.index_writes_enabled {
        sqlx::migrate!("./migrations")
            .run(&ctx.pg)
            .await
            .context("migrations failed")?;
        tracing::info!("migrations applied");
    } else {
        tracing::info!(
            "read-only corpus: skipping migrations, index writes and ask persistence"
        );
    }

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
