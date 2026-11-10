//! Batch ingestion (`recall ingest`): a [`Source`] returns a full snapshot of documents; [`ingest`] labels them through
//! the [`Labeler`] port, versions, embeds and stores their chunks through the [`ChunkStore`] port, and
//! supersedes what the snapshot no longer contains. Ingestion never reads or writes pipeline runs.
//!
//! A connector is any [`Source`]: it names itself with a [`SourceName`], and nothing in the core or the schema lists
//! the names. [`markdown`] is the shipped connector; a domain application passes its own to [`run_with`] or
//! [`ingest`], and checks it with the port's contract suite, `ingest::contract` and `recall::source_contract!` (both
//! behind the `testing` feature).

#[cfg(any(test, feature = "testing"))]
pub mod contract;
pub mod markdown;

use std::{collections::HashSet, fmt};

use async_trait::async_trait;
use eros::{Context, IntoUnion, ReshapeUnion};

use crate::{
    access::{
        Labeler,
        rules::{AccessArgs, LabelRules},
    },
    domain::{Document, SourceName},
    llm::{self, Embedder, ModelRef, backend::EmbedArgs},
    store::{
        self, DbArgs,
        chunks::{ChunkStore, ChunkStoreFailed, PgChunkStore, Stored},
    },
};

/// A connector that lists every document it currently holds. Each call is a full snapshot: a document missing from
/// it has its chunks superseded. `ingest::contract` (`testing` feature) holds the rules every connector keeps.
#[async_trait]
pub trait Source: Send + Sync {
    /// The [`Document::source`] of every document it returns. The same on every call: ingestion supersedes the
    /// documents stored under this name that a snapshot leaves out.
    fn name(&self) -> &SourceName;

    /// Every document in the source's current snapshot.
    async fn fetch_all(&self) -> eros::Result<Vec<Document>>;
}

/// The connectors one ingestion runs, in order, with distinct [`Source::name`]s.
pub struct Sources(Vec<Box<dyn Source>>);

/// Two connectors given to one ingestion share a name, so each one's snapshot would supersede the other's documents.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("more than one source is named {0:?}")]
pub struct DuplicateSourceName(pub SourceName);

impl Sources {
    /// # Errors
    ///
    /// [`DuplicateSourceName`] when two of `sources` have the same name.
    pub fn new(sources: Vec<Box<dyn Source>>) -> Result<Self, DuplicateSourceName> {
        let mut names = HashSet::new();
        if let Some(duplicate) = sources.iter().find(|s| !names.insert(s.name())) {
            return Err(DuplicateSourceName(duplicate.name().clone()));
        }
        Ok(Self(sources))
    }

    /// `true` when there is no connector to ingest.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = &dyn Source> {
        self.0.iter().map(Box::as_ref)
    }
}

impl TryFrom<Vec<Box<dyn Source>>> for Sources {
    type Error = DuplicateSourceName;

    fn try_from(sources: Vec<Box<dyn Source>>) -> Result<Self, DuplicateSourceName> {
        Self::new(sources)
    }
}

impl fmt::Debug for Sources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter().map(Source::name))
            .finish()
    }
}

/// What every ingestion needs, whichever connectors it runs. A domain binary flattens this beside its own
/// connector's options and calls [`run_with`].
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
    pub markdown: markdown::MarkdownArgs,
}

/// Runs `recall ingest` with the shipped markdown connector; see [`run_with`].
///
/// # Errors
///
/// When the markdown options are invalid, `RECALL_DOCS_DIR` is unset, or [`run_with`] fails.
pub async fn run(args: Args) -> eros::Result<()> {
    let Some(markdown) = markdown::Connector::from_args(&args.markdown)? else {
        eros::bail!("no sources are configured (set RECALL_DOCS_DIR)");
    };
    let sources = Sources::new(vec![Box::new(markdown)])?;
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
    let labeler = LabelRules::new(args.access.load()?.labels());
    let store = PgChunkStore::new(store::connect(&args.db).await?);
    let embedder = llm::embedder(&args.embed).await?;

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
    pub source: SourceName,
    /// Documents stored, whether or not any chunk changed.
    pub documents: u64,
    /// Chunk versions inserted and superseded.
    pub stored: Stored,
    /// Set when the source failed; counts cover what was stored before the failure.
    pub error: Option<eros::ErrorUnion>,
}

/// Ingests every source independently: a failing source does not stop the others; its error is in its report.
/// Each source's snapshot supersedes only documents stored under its own name. Each document is stored with the
/// labels `labeler` gives it; a document it cannot label is not stored, and its source fails.
///
/// With `reembed`, and only when every source succeeded, chunks embedded with any other model are superseded too.
/// Last, the full-text statistics that BM25 ranking reads are recomputed from the current chunks, in their own
/// transaction: each document commits separately, so there is no snapshot transaction to join, and until the refresh
/// commits retrieval ranks with the previous statistics.
///
/// # Errors
///
/// [`EmbeddingModelMismatch`], before touching any source, when current chunks use another model and `reembed` is
/// off; [`ChunkStoreFailed`] when the model check, the final supersede, or the statistics refresh fails.
pub async fn ingest(
    store: &dyn ChunkStore,
    embedder: &dyn Embedder,
    labeler: &dyn Labeler,
    sources: &Sources,
    reembed: bool,
) -> eros::Result<Vec<SourceReport>, (EmbeddingModelMismatch, ChunkStoreFailed)> {
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
    for source in sources.iter() {
        let name = source.name();
        let mut report = SourceReport {
            source: name.clone(),
            documents: 0,
            stored: Stored::default(),
            error: None,
        };
        let result = ingest_source(store, embedder, labeler, source, &mut report).await;
        match result {
            Ok(()) => tracing::info!(
                source = name.as_str(),
                documents = report.documents,
                new_versions = report.stored.new_versions,
                superseded = report.stored.superseded,
                "source ingested"
            ),
            Err(e) => {
                tracing::error!(
                    source = name.as_str(),
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

#[eros::context("ingesting source {}", source.name())]
async fn ingest_source(
    store: &dyn ChunkStore,
    embedder: &dyn Embedder,
    labeler: &dyn Labeler,
    source: &dyn Source,
    report: &mut SourceReport,
) -> eros::Result<()> {
    let name = source.name();
    let documents = source.fetch_all().await?;
    let mut seen = Vec::with_capacity(documents.len());
    for document in documents {
        eros::ensure!(
            document.source == *name,
            "{name} source returned document {}",
            document.id().0
        );
        let labels = labeler
            .label(&document)
            .await
            .with_context(|| format!("labelling document {}", document.id().0))?;
        let document = Document { labels, ..document };
        report.stored += store.store(&document, embedder).await?;
        report.documents += 1;
        seen.push(document.id());
    }
    report.stored.superseded += store.supersede_missing_documents(name, &seen).await?;
    Ok(())
}

/// In-memory [`Source`]s for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes {
    use async_trait::async_trait;

    use super::Source;
    use crate::domain::{Document, SourceName};

    /// Returns all of its documents on every fetch.
    pub struct StaticSource {
        /// Returned by [`Source::name`]; documents under another name fail ingestion.
        pub name: SourceName,
        /// The snapshot.
        pub documents: Vec<Document>,
    }

    #[async_trait]
    impl Source for StaticSource {
        fn name(&self) -> &SourceName {
            &self.name
        }

        async fn fetch_all(&self) -> eros::Result<Vec<Document>> {
            Ok(self.documents.clone())
        }
    }

    /// Fails every fetch, for ingestion's error paths.
    pub struct FailingSource {
        /// Returned by [`Source::name`].
        pub name: SourceName,
    }

    #[async_trait]
    impl Source for FailingSource {
        fn name(&self) -> &SourceName {
            &self.name
        }

        async fn fetch_all(&self) -> eros::Result<Vec<Document>> {
            eros::bail!("source {} is down", self.name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DuplicateSourceName, Source, Sources, fakes::FailingSource};
    use crate::domain::SourceName;

    #[test]
    fn sources_reject_two_connectors_of_one_name() {
        let named = |name: &'static str| -> Box<dyn Source> {
            Box::new(FailingSource {
                name: SourceName::from_static(name),
            })
        };

        assert!(Sources::new(vec![named("wiki"), named("markdown")]).is_ok());
        assert_eq!(
            Sources::new(vec![named("wiki"), named("markdown"), named("wiki")]).unwrap_err(),
            DuplicateSourceName(SourceName::from_static("wiki"))
        );
    }
}
