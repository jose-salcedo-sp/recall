//! Retrieval behind the [`Retriever`] port: given the question text, return current chunk versions best first.
//! [`hybrid::HybridRetriever`] is the Postgres adapter (full-text + vector, merged with reciprocal rank fusion).
//! Judging relevance is not retrieval's job; [`crate::pipeline::steps::Rerank`] does that.

/// An in-memory [`Retriever`] for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes;
pub mod glossary;
pub mod hybrid;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{Chunk, ChunkId};

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

/// Fused retrieval score; higher is better. Comparable only within one retrieval, and not a probability.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Relevance(pub f32);

/// Finds the chunks a question is most likely about.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Top hybrid matches, expanded to their parent unit and deduplicated, best first, ties by chunk id.
    ///
    /// There is no relevance floor: a non-blank query over a non-empty corpus returns its nearest chunks even when
    /// none is about the question, and Rerank judges relevance. Only a blank query or an empty corpus returns nothing.
    ///
    /// # Errors
    ///
    /// When the backend fails, or the corpus was embedded with another model than the query embedder.
    async fn retrieve(&self, query: &str) -> eros::Result<Vec<Retrieved>>;

    /// [`Retriever::retrieve`], with the glossary expansion it searched with. A retriever that does not expand
    /// questions returns an empty expansion.
    ///
    /// # Errors
    ///
    /// As [`Retriever::retrieve`].
    async fn retrieve_expanded(&self, query: &str) -> eros::Result<Retrieval> {
        Ok(Retrieval {
            hits: self.retrieve(query).await?,
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
