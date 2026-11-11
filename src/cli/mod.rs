//! The `recall` command line: clap options read from flags and the environment, and the adapters they select. It
//! is one caller of the library, not part of its interface; a domain library builds [`crate::Ports`] and a
//! [`crate::PipelineConfig`] itself. A domain binary can reuse these options, e.g. flatten [`ingest::IngestArgs`]
//! beside its own connector's and call [`ingest::run_with`]. Behind the `cli` feature, on by default.
//!
//! The environment is read here and nowhere else in the crate.

pub mod args;
mod aws;
pub mod backend;
pub mod eval;
pub mod ingest;
pub mod models;
pub mod pipeline;
pub mod serve;

use clap::{Parser, Subcommand};

/// `recall`'s arguments.
#[derive(Debug, Parser)]
#[command(
    name = "recall",
    about = "recall: answers from a corpus, with every claim's provenance checked and recorded"
)]
pub struct Cli {
    /// What to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The `recall` subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve the OpenAI-compatible API.
    Serve(serve::Args),
    /// Ingest the markdown corpus snapshot from RECALL_DOCS_DIR.
    Ingest(ingest::Args),
    /// Run the golden set and report per-step metrics.
    Eval(eval::Args),
    /// Check each model backend with one call.
    Models(models::Args),
}

impl Cli {
    /// Runs the subcommand.
    ///
    /// # Errors
    ///
    /// As the subcommand's `run`.
    pub async fn run(self) -> eros::Result<()> {
        match self.command {
            Command::Serve(args) => serve::run(args).await,
            Command::Ingest(args) => ingest::run(args).await,
            Command::Eval(args) => eval::run(args).await,
            Command::Models(args) => models::run(args).await,
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
