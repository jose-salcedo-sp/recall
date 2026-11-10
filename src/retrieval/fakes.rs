use std::collections::HashSet;

use async_trait::async_trait;

use crate::domain::Chunk;

use super::{Relevance, Retrieved, Retriever};

/// Ranks every chunk by the fraction of query words it contains, ties by chunk id. Like the vector side of
/// `HybridRetriever`, it keeps chunks that share no word: judging relevance is Rerank's job.
pub struct KeywordRetriever {
    /// The corpus; every chunk is returned on each retrieval.
    pub chunks: Vec<Chunk>,
}

#[async_trait]
impl Retriever for KeywordRetriever {
    async fn retrieve(&self, query: &str) -> eros::Result<Vec<Retrieved>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let query = words(query);
        let mut hits: Vec<_> = self
            .chunks
            .iter()
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
