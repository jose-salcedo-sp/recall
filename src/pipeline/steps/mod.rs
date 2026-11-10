//! The steps every category's pipeline is built from. See `docs/design.md` sections 5.3 to 5.5.

mod admit;
mod answer;
mod classify;
mod explain;
mod rerank;
mod retrieve;
mod verify;

pub use self::{
    admit::{ADMIT_PROMPT_VERSION, Admit},
    answer::{ANSWER_PROMPT_VERSION, Answer, Decided},
    classify::{CLASSIFY_PROMPT_VERSION, Classify},
    explain::{
        BOOLEAN_PROMPT_VERSION, Claim, Explain, Explained, ExplainedVerdict, Explanation,
        InvalidExplanation, PROMPT_VERSION,
    },
    rerank::{RERANK_PROMPT_VERSION, Rerank, RerankBy},
    retrieve::Retrieve,
    verify::{SUPPORT_PROMPT_VERSION, Support, Verified, VerifiedAnswer, VerifiedVerdict, Verify},
};

use crate::{
    domain::{AtLeastOne, Chunk},
    retrieval::Retrieved,
};

use super::{Open, Seq, trace::RankedChunk};

/// Kev's smaller models are validated to 8,192 tokens; longer text is judged on its beginning.
const MAX_STATE_CHARS: usize = 16_000;

/// Requests in flight at once when a step judges several texts; the server batches them.
const CONCURRENT_REQUESTS: usize = 8;

/// A chunk as a judge reads it: its title and heading, then the beginning of its text.
fn judged_text(chunk: &Chunk) -> String {
    let heading = chunk
        .heading
        .as_deref()
        .map(|heading| format!(" > {heading}"))
        .unwrap_or_default();
    let text = match chunk.text.char_indices().nth(MAX_STATE_CHARS) {
        Some((end, _)) => &chunk.text[..end],
        None => &chunk.text,
    };
    format!("{}{heading}\n\n{text}", chunk.title)
}

/// The LLM explains from the reranked sources, Verify checks the claims. The `open` pipeline after `Rerank`, or after
/// `Admit` when `admit.judge` names a judge.
pub type OpenEnded = Seq<Explain<Open>, Verify<Open>>;
/// The judge decides, the LLM explains, Verify checks the claims. Only `Decisive<Boolean>` is a `Pipeline` so far.
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
        Chunk, ChunkId, ChunkVersion, DocumentId, Labels, Sensitivity, SourceName,
    };

    pub fn chunk(locator: &str, title: &str, text: &str) -> Chunk {
        let source = SourceName::from_static("markdown");
        let document_id = DocumentId::new(&source, title);
        Chunk {
            id: ChunkId::new(&document_id, locator),
            version: ChunkVersion(format!("v-{locator}")),
            document_id,
            source,
            title: title.into(),
            heading: Some(locator.into()),
            url: format!("https://example.test/docs/{title}#{locator}"),
            text: text.into(),
            authoritative: false,
            labels: Labels {
                acl: vec![],
                tenant: None,
                sensitivity: Sensitivity::Internal,
            },
        }
    }
}
