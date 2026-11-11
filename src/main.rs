//! The `recall` binary: parses the command line and runs `serve`, `ingest`, `eval` or `models` from the library.

use std::process::ExitCode;

use clap::Parser;
use recall::cli::Cli;
use tracing_subscriber::EnvFilter;

/// Reports a failure once, with its causes and context; set `RUST_BACKTRACE=1` for the backtrace too.
#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    match Cli::parse().run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if std::env::var_os("RUST_BACKTRACE").is_some() => {
            eprintln!("error: {error:?}");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("error: {error:#?}");
            ExitCode::FAILURE
        }
    }
}
