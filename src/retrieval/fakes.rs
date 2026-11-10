use std::collections::HashSet;

use async_trait::async_trait;

use crate::{access::ChunkScope, domain::Chunk};

use super::{Relevance, Retrieved, Retriever};

/// Ranks every readable chunk by the fraction of query words it contains, ties by chunk id. Like the vector side of
/// `HybridRetriever`, it keeps chunks that share no word: judging relevance is Rerank's job.
pub struct KeywordRetriever {
    /// The corpus; every chunk the scope admits is returned on each retrieval.
    pub chunks: Vec<Chunk>,
}

#[async_trait]
impl Retriever for KeywordRetriever {
    async fn retrieve(&self, query: &str, scope: &ChunkScope) -> eros::Result<Vec<Retrieved>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let query = words(query);
        let mut hits: Vec<_> = self
            .chunks
            .iter()
            .filter(|chunk| scope.admits(&chunk.labels))
            .map(|chunk| {
                let shared = query.intersection(&words(&chunk.text)).count();
                Retrieved {
                    chunk: chunk.clone(),
                    score: Relevance(shared as f32 / query.len().max(1) as f32),
                    context_for: None,
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .0
                .total_cmp(&a.score.0)
                .then_with(|| a.chunk.id.cmp(&b.chunk.id))
        });
        Ok(hits)
    }
}

fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Ignores the scope and returns every chunk, as a retriever with a broken pushdown would, so tests can prove that
/// the pipeline's own checks keep what it returns out of answers.
pub struct LeakyRetriever {
    /// Returned on every retrieval, in order, whatever the scope.
    pub chunks: Vec<Chunk>,
}

#[async_trait]
impl Retriever for LeakyRetriever {
    async fn retrieve(&self, _query: &str, _scope: &ChunkScope) -> eros::Result<Vec<Retrieved>> {
        Ok(self
            .chunks
            .iter()
            .map(|chunk| Retrieved {
                chunk: chunk.clone(),
                score: Relevance(1.0),
                context_for: None,
            })
            .collect())
    }
}

/// Fails every retrieval, with a message like a database error's, for error-path tests.
pub struct FailingRetriever;

#[async_trait]
impl Retriever for FailingRetriever {
    async fn retrieve(&self, _query: &str, _scope: &ChunkScope) -> eros::Result<Vec<Retrieved>> {
        eros::bail!("pool timed out connecting to postgres://recall@db.internal:5432/recall")
    }
}
