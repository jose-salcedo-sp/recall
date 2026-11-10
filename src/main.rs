//! The `drew` binary: parses the command line and runs `serve`, `ingest`, `eval` or `models` from the library.

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use drew::{eval, ingest, models, server};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "drew",
    about = "Drew AI: answers about SAP and ARM internals, with provenance"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the OpenAI-compatible API.
    Serve(server::Args),
    /// Ingest the markdown corpus snapshot from DREW_DOCS_DIR.
    Ingest(ingest::Args),
    /// Run the golden set and report per-step metrics.
    Eval(eval::Args),
    /// Check each model backend with one call.
    Models(models::Args),
}

/// Reports a failure once, with its causes and context; set `RUST_BACKTRACE=1` for the backtrace too.
#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let result = match Cli::parse().command {
        Command::Serve(args) => server::run(args).await,
        Command::Ingest(args) => ingest::run(args).await,
        Command::Eval(args) => eval::run(args).await,
        Command::Models(args) => models::run(args).await,
    };
    match result {
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

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_is_well_formed() {
        super::Cli::command().debug_assert();
    }
}
