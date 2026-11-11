//! `recall ingest`: the markdown corpus into Postgres, embedded by the configured backend and labelled by the
//! access policy's rules.

use std::path::PathBuf;

use crate::{
    ingest::{
        Sources, ingest,
        markdown::{Connector, MarkdownConfig},
    },
    store::chunks::PgChunkStore,
};

use super::{
    args::{AccessArgs, DbArgs},
    backend::{self, EmbedArgs},
};

/// Where the markdown corpus is and how its paths split into layers.
#[derive(Debug, clap::Args)]
pub struct MarkdownArgs {
    /// Root directory of the markdown corpus.
    #[arg(long, env = "RECALL_DOCS_DIR")]
    pub docs: Option<PathBuf>,
    /// Optional base URL for citations (e.g. a GitHub blob URL ending in `/`).
    #[arg(long, env = "RECALL_DOCS_BASE_URL")]
    pub docs_base_url: Option<String>,
    /// Comma-separated paths under the docs root holding settled, authoritative answers. Empty means the whole root.
    #[arg(long, env = "RECALL_DOCS_AUTHORITATIVE", value_delimiter = ',')]
    pub authoritative: Vec<PathBuf>,
    /// Comma-separated paths under the docs root ingested as supporting evidence, not authoritative.
    /// The longest listed path containing a file decides its layer; files under no listed path are skipped.
    #[arg(long, env = "RECALL_DOCS_EVIDENCE", value_delimiter = ',')]
    pub evidence: Vec<PathBuf>,
    /// Comma-separated paths under the docs root to leave out, even inside an authoritative or evidence path.
    #[arg(long, env = "RECALL_DOCS_SKIP", value_delimiter = ',')]
    pub skip: Vec<PathBuf>,
}

impl MarkdownArgs {
    /// `None` when `RECALL_DOCS_DIR` is unset, so the source is not configured.
    pub fn config(&self) -> Option<MarkdownConfig> {
        Some(MarkdownConfig {
            docs: self.docs.clone()?,
            base_url: self.docs_base_url.clone(),
            authoritative: self.authoritative.clone(),
            evidence: self.evidence.clone(),
            skip: self.skip.clone(),
        })
    }
}

/// What every ingestion needs, whichever connectors it runs. A domain binary flattens this beside its own
/// connector's options and calls [`run_with`]; a domain library calls [`crate::ingest::ingest`] instead.
#[derive(Debug, clap::Args)]
pub struct IngestArgs {
    /// Database connection.
    #[command(flatten)]
    pub db: DbArgs,
    /// Embedding backend; its model is recorded on every chunk.
    #[command(flatten)]
    pub embed: EmbedArgs,
    /// Re-ingest every source unit from scratch with the configured embedding model, replacing chunks embedded with any other model.
    #[arg(long)]
    pub reembed: bool,
    /// The access policy, whose label rules set each document's labels.
    #[command(flatten)]
    pub access: AccessArgs,
}

/// `recall ingest` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Database, embedder and `--reembed`.
    #[command(flatten)]
    pub ingest: IngestArgs,
    /// The markdown corpus.
    #[command(flatten)]
    pub markdown: MarkdownArgs,
}

/// Runs `recall ingest` with the shipped markdown connector; see [`run_with`].
///
/// # Errors
///
/// When the markdown options are invalid, `RECALL_DOCS_DIR` is unset, or [`run_with`] fails.
pub async fn run(args: Args) -> eros::Result<()> {
    let Some(markdown) = args.markdown.config() else {
        eros::bail!("no sources are configured (set RECALL_DOCS_DIR)");
    };
    let sources = Sources::new(vec![Box::new(Connector::new(&markdown)?)])?;
    run_with(&args.ingest, &sources).await
}

/// Connects, ingests `sources`, and fails if any source failed.
///
/// # Errors
///
/// When `sources` is empty, the access policy does not load, the database or embedder cannot be set up, [`ingest`]
/// fails, or any source's report carries an error.
#[eros::context("ingesting sources {:?}", sources)]
pub async fn run_with(args: &IngestArgs, sources: &Sources) -> eros::Result<()> {
    eros::ensure!(!sources.is_empty(), "no sources are configured");
    let labeler = args.access.load()?.labels().clone();
    let store = PgChunkStore::new(args.db.connect().await?);
    let embedder = backend::embedder(&args.embed).await?;

    let reports = ingest(&store, embedder.as_ref(), &labeler, sources, args.reembed).await?;
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| r.error.is_some())
        .map(|r| r.source.as_str())
        .collect();
    eros::ensure!(
        failed.is_empty(),
        "ingest failed for: {}",
        failed.join(", ")
    );
    Ok(())
}
