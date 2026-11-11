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
    access::{Access, AccessDecision},
    domain::{AtLeastOne, Chunk},
    retrieval::Retrieved,
};

use super::{Open, Scored, Seq, StepError, trace::RankedChunk};

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

    /// The sources `access` admits, in order, and one decision per source.
    fn enforce(self, access: &Access) -> (Self, Vec<AccessDecision>) {
        let (kept, decisions) = enforce(access, self.0);
        (Self(kept), decisions)
    }
}

/// Keeps the hits `access` admits, in order, with one decision per hit. Every step that reads chunks calls it on its
/// input, so a chunk outside the run's scope never reaches a judge, the LLM, a citation or the report.
fn enforce(access: &Access, hits: Vec<Retrieved>) -> (Vec<Retrieved>, Vec<AccessDecision>) {
    let decisions = access.chunks().check(hits.iter().map(|hit| &hit.chunk));
    let kept = hits
        .into_iter()
        .zip(&decisions)
        .filter(|(_, check)| check.decision.is_allowed())
        .map(|(hit, _)| hit)
        .collect();
    (kept, decisions)
}

/// Attaches the step's access checks to its result, whether it passed or halted.
fn with_access<TOutput, TRecord>(
    result: Result<Scored<TOutput, TRecord>, StepError>,
    access: Vec<AccessDecision>,
) -> Result<Scored<TOutput, TRecord>, StepError> {
    match result {
        Ok(scored) => Ok(scored.with_access(access)),
        Err(error) => Err(error.with_access(access)),
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
            version: ChunkVersion::try_from(format!("v-{locator}")).unwrap(),
            document_id,
            source,
            title: title.into(),
            heading: Some(locator.into()),
            url: format!("https://example.test/docs/{title}#{locator}"),
            text: text.into(),
            standing: crate::domain::Standing::Evidence,
            labels: Labels {
                acl: crate::access::Acl::anyone(),
                sensitivity: Sensitivity::Internal,
            },
        }
    }
}
