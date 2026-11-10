//! The steps every category's pipeline is built from. See `docs/design.md` sections 5.3 to 5.5.

mod answer;
mod classify;
mod explain;
mod rerank;
mod retrieve;
mod verify;

pub use self::{
    answer::{ANSWER_PROMPT_VERSION, Answer, Decided},
    classify::{CLASSIFY_PROMPT_VERSION, Classify},
    explain::{
        BOOLEAN_PROMPT_VERSION, Claim, Explain, Explained, ExplainedVerdict, Explanation,
        InvalidExplanation, PROMPT_VERSION,
    },
    rerank::{RERANK_PROMPT_VERSION, Rerank},
    retrieve::Retrieve,
    verify::{Verified, VerifiedAnswer, VerifiedVerdict, Verify},
};

use crate::{domain::AtLeastOne, retrieval::Retrieved};

use super::{Open, Seq, trace::RankedChunk};

/// Kev's smaller models are validated to 8,192 tokens; longer text is judged on its beginning.
const MAX_STATE_CHARS: usize = 16_000;

/// The LLM explains from the reranked sources, Verify checks the citations. The `open` pipeline after `Rerank`.
pub type OpenEnded = Seq<Explain<Open>, Verify<Open>>;
/// System One decides, the LLM explains, Verify checks the citations. Only `Decisive<Boolean>` is a `Pipeline` so far.
pub type Decisive<TCategory> = Seq<Answer<TCategory>, Seq<Explain<TCategory>, Verify<TCategory>>>;

/// Output of `Retrieve`: hybrid matches, best first.
#[derive(Clone, Debug)]
pub struct Candidates(AtLeastOne<Retrieved>);

/// Output of `Rerank`: the sources `Answer` and `Explain` see, best first.
#[derive(Clone, Debug)]
pub struct Reranked(Vec<Retrieved>);

impl Reranked {
    /// Each kept match followed by its context chunks, matches best first. May be empty.
    pub fn sources(&self) -> &[Retrieved] {
        &self.0
    }
}

fn ranked(hits: &[Retrieved]) -> Vec<RankedChunk> {
    hits.iter()
        .map(|hit| RankedChunk {
            chunk: hit.chunk.reference(),
            score: hit.score,
            probability: None,
            context_for: hit.context_for.clone(),
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod fixtures {
    use crate::domain::{
        Chunk, ChunkId, ChunkVersion, DocumentId, Labels, Sensitivity, SourceKind,
    };

    pub fn chunk(locator: &str, title: &str, text: &str) -> Chunk {
        let document_id = DocumentId::new(SourceKind::Markdown, title);
        Chunk {
            id: ChunkId::new(&document_id, locator),
            version: ChunkVersion(format!("v-{locator}")),
            document_id,
            source: SourceKind::Markdown,
            title: title.into(),
            heading: Some(locator.into()),
            url: format!("https://example.test/docs/{title}#{locator}"),
            text: text.into(),
            authoritative: false,
            labels: Labels {
                acl: vec![],
                customer: None,
                sensitivity: Sensitivity::Internal,
            },
        }
    }
}
