use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use eros::IntoUnion;

use crate::{
    clock::{Clock, fakes::FixedClock},
    domain::{Chunk, ChunkVersion, Document, DocumentId, RunId, SourceName, UserId},
    llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef},
    pipeline::report::RunReport,
};

use super::{
    RunStore, StoredRun,
    chunks::{ChunkStore, ChunkStoreFailed, Stored, embedding_input},
};

/// Keeps runs in a map; like the Postgres store, rejects a second save of the same run id.
pub struct MemoryRunStore {
    runs: Mutex<HashMap<RunId, StoredRun>>,
    clock: Arc<dyn Clock>,
}

/// Stamps runs with a [`FixedClock`] at its default time.
impl Default for MemoryRunStore {
    fn default() -> Self {
        Self::new(Arc::new(FixedClock::default()))
    }
}

#[async_trait]
impl RunStore for MemoryRunStore {
    async fn save(&self, user: &UserId, report: &RunReport) -> eros::Result<()> {
        let mut runs = self
            .runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it");
        let Entry::Vacant(slot) = runs.entry(report.id) else {
            eros::bail!("run {} already saved", report.id);
        };
        slot.insert(StoredRun {
            user: user.clone(),
            created_at: self.clock.now(),
            report: report.clone(),
        });
        Ok(())
    }

    async fn get(&self, id: RunId) -> eros::Result<Option<StoredRun>> {
        Ok(self
            .runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .get(&id)
            .cloned())
    }
}

/// Fails every call, with a message like a database error's, for error-path tests.
pub struct FailingRunStore;

#[async_trait]
impl RunStore for FailingRunStore {
    async fn save(&self, _: &UserId, _: &RunReport) -> eros::Result<()> {
        eros::bail!("pool timed out connecting to postgres://recall@db.internal:5432/recall")
    }

    async fn get(&self, _: RunId) -> eros::Result<Option<StoredRun>> {
        eros::bail!("pool timed out connecting to postgres://recall@db.internal:5432/recall")
    }
}

impl MemoryRunStore {
    /// `clock` stamps [`StoredRun::created_at`].
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
            clock,
        }
    }

    /// Every saved run, in no particular order.
    pub fn stored(&self) -> Vec<StoredRun> {
        self.runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .values()
            .cloned()
            .collect()
    }
}

/// One stored chunk version, with whether it is current.
#[derive(Clone)]
struct Version {
    chunk: Chunk,
    ordinal: usize,
    model: ModelRef,
    current: bool,
}

/// Keeps chunk versions in memory, with the Postgres store's rules: content-addressed versions, a revived version is
/// not re-embedded, and every supersede counts what it superseded. It keeps no search statistics.
#[derive(Default)]
pub struct MemoryChunkStore {
    versions: Mutex<Vec<Version>>,
}

impl MemoryChunkStore {
    /// The current chunks, ordered by document then ordinal, for a retriever over what was stored.
    pub fn current(&self) -> Vec<Chunk> {
        let mut current: Vec<_> = self
            .lock()
            .iter()
            .filter(|version| version.current)
            .map(|version| {
                (
                    version.chunk.document_id.clone(),
                    version.ordinal,
                    version.chunk.clone(),
                )
            })
            .collect();
        current.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        current.into_iter().map(|(_, _, chunk)| chunk).collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Version>> {
        self.versions
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
    }

    fn supersede(&self, mut stale: impl FnMut(&Version) -> bool) -> u64 {
        let mut superseded = 0;
        for version in self.lock().iter_mut().filter(|version| version.current) {
            if stale(version) {
                version.current = false;
                superseded += 1;
            }
        }
        superseded
    }
}

#[async_trait]
impl ChunkStore for MemoryChunkStore {
    async fn store(&self, document: &Document, embedder: &dyn Embedder) -> eros::Result<Stored> {
        use std::collections::HashSet;

        let document_id = document.id();
        let model = embedder.model();
        let mut locators = HashSet::new();
        for draft in &document.chunks {
            eros::ensure!(
                locators.insert(draft.locator.as_str()),
                "document {} has duplicate chunk locator {:?}",
                document_id.0,
                draft.locator
            );
        }
        let drafts: Vec<(Chunk, usize)> = document
            .chunks
            .iter()
            .enumerate()
            .map(|(ordinal, draft)| {
                let chunk = Chunk {
                    id: document.chunk_id(draft),
                    version: ChunkVersion::compute(document, draft, &model),
                    document_id: document_id.clone(),
                    source: document.source.clone(),
                    title: document.title.clone(),
                    heading: draft.heading.clone(),
                    url: draft.url.clone(),
                    text: draft.text.clone(),
                    authoritative: draft.authoritative,
                    labels: document.labels.clone(),
                };
                (chunk, ordinal)
            })
            .collect();
        let fresh: Vec<usize> = {
            let versions = self.lock();
            drafts
                .iter()
                .enumerate()
                .filter(|(_, (draft, _))| {
                    !versions.iter().any(|stored| {
                        stored.chunk.id == draft.id && stored.chunk.version == draft.version
                    })
                })
                .map(|(index, _)| index)
                .collect()
        };
        if !fresh.is_empty() {
            let texts: Vec<String> = fresh
                .iter()
                .map(|&index| embedding_input(document, &document.chunks[index]))
                .collect();
            let embeddings = embedder
                .embed(&texts, EmbedPurpose::Document)
                .await
                .map_err(|error| error.context(format!("embedding chunks of {}", document_id.0)))?;
            eros::ensure!(
                embeddings.len() == texts.len(),
                "embedder returned {} vectors for {} texts",
                embeddings.len(),
                texts.len()
            );
            if let Some(bad) = embeddings.iter().find(|e| e.len() != EMBEDDING_DIM) {
                eros::bail!(
                    "embedder returned a {}-dimensional vector, expected {EMBEDDING_DIM}",
                    bad.len()
                );
            }
        }

        let mut versions = self.lock();
        let mut superseded = 0;
        for stored in versions
            .iter_mut()
            .filter(|stored| stored.current && stored.chunk.document_id == document_id)
        {
            if !drafts.iter().any(|(draft, _)| {
                draft.id == stored.chunk.id && draft.version == stored.chunk.version
            }) {
                stored.current = false;
                superseded += 1;
            }
        }
        let mut new_versions = 0;
        for (draft, ordinal) in drafts {
            match versions
                .iter_mut()
                .find(|stored| stored.chunk.id == draft.id && stored.chunk.version == draft.version)
            {
                Some(stored) => {
                    if !stored.current {
                        new_versions += 1;
                    }
                    stored.current = true;
                    stored.ordinal = ordinal;
                    stored.chunk = draft;
                }
                None => {
                    new_versions += 1;
                    versions.push(Version {
                        chunk: draft,
                        ordinal,
                        model: model.clone(),
                        current: true,
                    });
                }
            }
        }
        Ok(Stored {
            new_versions,
            superseded,
        })
    }

    async fn supersede_missing_documents(
        &self,
        source: &SourceName,
        present: &[DocumentId],
    ) -> eros::Result<u64> {
        Ok(self.supersede(|version| {
            version.chunk.source == *source && !present.contains(&version.chunk.document_id)
        }))
    }

    async fn current_embedding_models(&self) -> eros::Result<Vec<ModelRef>, (ChunkStoreFailed,)> {
        let mut models: Vec<_> = self
            .lock()
            .iter()
            .filter(|version| version.current)
            .map(|version| version.model.clone())
            .collect();
        models.sort_by(|a, b| (&a.id, &a.version).cmp(&(&b.id, &b.version)));
        models.dedup();
        Ok(models)
    }

    async fn supersede_other_models(
        &self,
        model: &ModelRef,
    ) -> eros::Result<u64, (ChunkStoreFailed,)> {
        Ok(self.supersede(|version| version.model != *model))
    }

    async fn refresh_term_stats(&self) -> eros::Result<(), (ChunkStoreFailed,)> {
        Ok(())
    }
}

/// Fails every call, with a message like a database error's, for error-path tests.
pub struct FailingChunkStore;

const DOWN: &str = "pool timed out connecting to postgres://recall@db.internal:5432/recall";

#[async_trait]
impl ChunkStore for FailingChunkStore {
    async fn store(&self, _: &Document, _: &dyn Embedder) -> eros::Result<Stored> {
        eros::bail!("{DOWN}")
    }

    async fn supersede_missing_documents(
        &self,
        _: &SourceName,
        _: &[DocumentId],
    ) -> eros::Result<u64> {
        eros::bail!("{DOWN}")
    }

    async fn current_embedding_models(&self) -> eros::Result<Vec<ModelRef>, (ChunkStoreFailed,)> {
        Err(ChunkStoreFailed::new(DOWN)).union()
    }

    async fn supersede_other_models(&self, _: &ModelRef) -> eros::Result<u64, (ChunkStoreFailed,)> {
        Err(ChunkStoreFailed::new(DOWN)).union()
    }

    async fn refresh_term_stats(&self) -> eros::Result<(), (ChunkStoreFailed,)> {
        Err(ChunkStoreFailed::new(DOWN)).union()
    }
}
