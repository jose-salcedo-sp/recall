//! Retrieval behind the [`Retriever`] port: given the question text and the asker's [`ChunkScope`], return the current
//! chunk versions the scope admits, best first.
//! [`hybrid::HybridRetriever`] is the Postgres adapter (full-text + vector, merged with reciprocal rank fusion).
//! Judging relevance is not retrieval's job; [`crate::pipeline::steps::Rerank`] does that.

/// An in-memory and a failing [`Retriever`] for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes;
pub mod glossary;
pub mod hybrid;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    access::ChunkScope,
    domain::{Chunk, ChunkId},
};

use glossary::Expansion;

/// One chunk in a retrieval result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Retrieved {
    /// The current version of the chunk.
    pub chunk: Chunk,
    /// A context chunk scores just below the match it sits next to.
    pub score: Relevance,
    /// Set when the chunk is not a match itself but context next to one: the id of that match.
    #[serde(default)]
    pub context_for: Option<ChunkId>,
}

/// Fused retrieval score; higher is better. Comparable only within one retrieval, and not a probability. Finite, so
/// scores sort without a `NaN` case.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Relevance(f32);

/// A score rejected by [`Relevance::new`]: `NaN` or infinite.
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("expected a finite relevance score, got {0}")]
pub struct NotFinite(pub f32);

impl Relevance {
    /// # Errors
    ///
    /// [`NotFinite`] when `score` is `NaN` or infinite.
    pub fn new(score: f32) -> Result<Self, NotFinite> {
        if score.is_finite() {
            Ok(Self(score))
        } else {
            Err(NotFinite(score))
        }
    }

    /// The score.
    pub fn get(self) -> f32 {
        self.0
    }

    /// The next score below, for a context chunk placed just under its match.
    pub fn next_down(self) -> Self {
        Self(self.0.next_down())
    }

    /// Higher first.
    pub fn descending(a: &Self, b: &Self) -> std::cmp::Ordering {
        b.0.total_cmp(&a.0)
    }
}

impl TryFrom<f32> for Relevance {
    type Error = NotFinite;

    fn try_from(score: f32) -> Result<Self, NotFinite> {
        Self::new(score)
    }
}

impl From<Relevance> for f32 {
    fn from(score: Relevance) -> f32 {
        score.0
    }
}

/// Finds the chunks a question is most likely about, among those the asker may read.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Top hybrid matches, expanded to their parent unit and deduplicated, best first, ties by chunk id.
    ///
    /// Only chunks `scope` admits are read: matches, the context added next to them, and the text questions are
    /// expanded from. Filtering happens before ranking, so a chunk outside the scope neither appears nor takes a
    /// place. There is no relevance floor: a non-blank query over a corpus with a readable chunk returns its nearest
    /// readable chunks even when none is about the question, and Rerank judges relevance. Only a blank query or no
    /// readable chunk returns nothing.
    ///
    /// # Errors
    ///
    /// When the backend fails, or the corpus was embedded with another model than the query embedder.
    async fn retrieve(&self, query: &str, scope: &ChunkScope) -> eros::Result<Vec<Retrieved>>;

    /// [`Retriever::retrieve`], with the glossary expansion it searched with. A retriever that does not expand
    /// questions returns an empty expansion.
    ///
    /// # Errors
    ///
    /// As [`Retriever::retrieve`].
    async fn retrieve_expanded(&self, query: &str, scope: &ChunkScope) -> eros::Result<Retrieval> {
        Ok(Retrieval {
            hits: self.retrieve(query, scope).await?,
            expansion: Expansion::default(),
        })
    }
}

/// What [`Retriever::retrieve_expanded`] found and searched with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Retrieval {
    /// As [`Retriever::retrieve`] returns them.
    pub hits: Vec<Retrieved>,
    /// The register entries the question matched and the terms they added.
    pub expansion: Expansion,
}
