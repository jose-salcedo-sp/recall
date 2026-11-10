//! Batch ingestion (`drew ingest`): a [`Source`] returns a full snapshot of documents; [`ingest`] versions, embeds
//! and stores their chunks through [`crate::store::chunks::ChunkStore`], and supersedes what the snapshot no longer
//! contains. The pre-audited markdown directory ([`markdown`]) is the only source. Ingestion never reads or writes
//! pipeline runs.

pub mod markdown;

use async_trait::async_trait;
use eros::{IntoUnion, ReshapeUnion};
use sqlx::PgPool;

use crate::{
    domain::{Document, SourceKind},
    llm::{self, Embedder, ModelRef, backend::EmbedArgs},
    store::{
        self, DbArgs,
        chunks::{ChunkStore, Stored},
    },
};

/// A connector that lists every document it currently holds. Each call is a full snapshot: a document missing from
/// it has its chunks superseded.
#[async_trait]
pub trait Source: Send + Sync {
    /// Every document this source returns has this [`Document::source`].
    fn kind(&self) -> SourceKind;

    /// Every document in the source's current snapshot.
    async fn fetch_all(&self) -> eros::Result<Vec<Document>>;
}

/// `drew ingest` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Database connection.
    #[command(flatten)]
    pub db: DbArgs,
    /// Embedding backend; its model is recorded on every chunk.
    #[command(flatten)]
    pub embed: EmbedArgs,
    /// Re-ingest every source unit from scratch with the configured embedding model, replacing chunks embedded with any other model.
    #[arg(long)]
    pub reembed: bool,
    /// The markdown corpus.
    #[command(flatten)]
    pub markdown: markdown::MarkdownArgs,
}

/// Runs `drew ingest`: connects, ingests every configured source, and fails if any source failed.
///
/// # Errors
///
/// When no source is configured, the database or embedder cannot be set up, [`ingest`] fails, or any source's
/// report carries an error.
pub async fn run(args: Args) -> eros::Result<()> {
    let pool = store::connect(&args.db).await?;
    let embedder = llm::embedder(&args.embed).await?;

    let mut sources: Vec<Box<dyn Source>> = Vec::new();
    if let Some(connector) = markdown::Connector::from_args(&args.markdown)? {
        sources.push(Box::new(connector));
    }
    eros::ensure!(
        !sources.is_empty(),
        "no sources are configured (set DREW_DOCS_DIR)"
    );

    let reports = ingest(&pool, embedder.as_ref(), &sources, args.reembed).await?;
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

/// Current chunks use a different embedding model than the configured one, and `--reembed` is off.
#[derive(Debug, thiserror::Error)]
#[error(
    "current chunks were embedded with {current:?} but the configured embedding model is {configured:?}; \
     mixing embedding models is not allowed, so pass --reembed to re-ingest everything with the configured model"
)]
pub struct EmbeddingModelMismatch {
    /// Every model among current chunks.
    pub current: Vec<ModelRef>,
    /// The embedder's model.
    pub configured: ModelRef,
}

/// What [`ingest`] did for one source.
#[derive(Debug)]
pub struct SourceReport {
    /// The source this report covers.
    pub source: SourceKind,
    /// Documents stored, whether or not any chunk changed.
    pub documents: u64,
    /// Chunk versions inserted and superseded.
    pub stored: Stored,
    /// Set when the source failed; counts cover what was stored before the failure.
    pub error: Option<eros::ErrorUnion>,
}

/// Ingests every source independently: a failing source does not stop the others; its error is in its report.
///
/// With `reembed`, and only when every source succeeded, chunks embedded with any other model are superseded too.
/// Last, the full-text statistics that BM25 ranking reads are recomputed from the current chunks, in their own
/// transaction: each document commits separately, so there is no snapshot transaction to join, and until the refresh
/// commits retrieval ranks with the previous statistics.
///
/// # Errors
///
/// [`EmbeddingModelMismatch`], before touching any source, when current chunks use another model and `reembed` is
/// off; [`sqlx::Error`] when the model check, the final supersede, or the statistics refresh fails.
pub async fn ingest(
    pool: &PgPool,
    embedder: &dyn Embedder,
    sources: &[Box<dyn Source>],
    reembed: bool,
) -> eros::Result<Vec<SourceReport>, (EmbeddingModelMismatch, sqlx::Error)> {
    let store = ChunkStore::new(pool.clone());
    let model = embedder.model();

    let current = store.current_embedding_models().await.widen()?;
    if !reembed && current.iter().any(|m| *m != model) {
        return Err(EmbeddingModelMismatch {
            current,
            configured: model,
        })
        .union();
    }

    let mut reports = Vec::new();
    for source in sources {
        let kind = source.kind();
        let mut report = SourceReport {
            source: kind,
            documents: 0,
            stored: Stored::default(),
            error: None,
        };
        let result = ingest_source(&store, embedder, source.as_ref(), &mut report).await;
        match result {
            Ok(()) => tracing::info!(
                source = kind.as_str(),
                documents = report.documents,
                new_versions = report.stored.new_versions,
                superseded = report.stored.superseded,
                "source ingested"
            ),
            Err(e) => {
                tracing::error!(
                    source = kind.as_str(),
                    documents = report.documents,
                    new_versions = report.stored.new_versions,
                    superseded = report.stored.superseded,
                    error = format!("{e:#?}"),
                    "source failed"
                );
                report.error = Some(e);
            }
        }
        reports.push(report);
    }

    if reembed && reports.iter().all(|r| r.error.is_none()) {
        let superseded = store.supersede_other_models(&model).await.widen()?;
        if superseded > 0 {
            tracing::warn!(
                superseded,
                "superseded chunks embedded with other models that this run did not re-ingest"
            );
        }
    }
    // A failed source may have stored some documents before failing, so the statistics are refreshed regardless.
    store.refresh_term_stats().await.widen()?;
    Ok(reports)
}

async fn ingest_source(
    store: &ChunkStore,
    embedder: &dyn Embedder,
    source: &dyn Source,
    report: &mut SourceReport,
) -> eros::Result<()> {
    let kind = source.kind();
    let documents = source.fetch_all().await?;
    let mut seen = Vec::with_capacity(documents.len());
    for document in &documents {
        eros::ensure!(
            document.source == kind,
            "{} source returned document {}",
            kind.as_str(),
            document.id().0
        );
        report.stored += store.store(document, embedder).await?;
        report.documents += 1;
        seen.push(document.id());
    }
    report.stored.superseded += store.supersede_missing_documents(kind, &seen).await?;
    Ok(())
}

/// In-memory [`Source`]s for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes {
    use async_trait::async_trait;

    use super::Source;
    use crate::domain::{Document, SourceKind};

    /// Returns all of its documents on every fetch.
    pub struct StaticSource {
        /// Returned by [`Source::kind`]; documents of another kind fail ingestion.
        pub kind: SourceKind,
        /// The snapshot.
        pub documents: Vec<Document>,
    }

    #[async_trait]
    impl Source for StaticSource {
        fn kind(&self) -> SourceKind {
            self.kind
        }

        async fn fetch_all(&self) -> eros::Result<Vec<Document>> {
            Ok(self.documents.clone())
        }
    }
}
