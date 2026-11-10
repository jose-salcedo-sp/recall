//! The audit trail of a run: one [`TraceEntry`] per step, appended by [`super::Run::then`]. Records reference chunks
//! by id and version and never copy their text.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, ChunkId, ChunkRef, Probability},
    llm::ModelRef,
    retrieval::{Relevance, glossary::Expansion},
};

use super::DeclineReason;

/// Flat list of step entries, ordered by start time. Persisted as JSONB in `pipeline_runs.trace`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Trace(Vec<TraceEntry>);

impl Trace {
    /// Entries in the order they were appended: by step, with a parallel group's left step first.
    pub fn entries(&self) -> &[TraceEntry] {
        &self.0
    }

    pub(super) fn push(&mut self, entry: TraceEntry) {
        self.0.push(entry);
    }
}

/// One step's execution, whatever its outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceEntry {
    /// The step's [`super::Step::NAME`].
    pub step: String,
    /// Set when the step ran concurrently with others, e.g. `classify_retrieve`.
    pub group: Option<String>,
    /// Wall-clock start.
    pub started_at: DateTime<Utc>,
    /// Elapsed time, in milliseconds.
    pub duration_ms: u64,
    /// The model the step called, if any.
    pub model: Option<ModelRef>,
    /// Version of the prompt sent to `model`.
    pub prompt_version: Option<String>,
    /// Passed, declined or failed, with the score and threshold compared.
    pub outcome: StepOutcome,
    /// The step's own data; `None` when it failed before producing any.
    pub record: Option<TraceRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
/// How one step ended. Mirrors [`super::Halt`], plus `Passed`.
pub enum StepOutcome {
    /// The run continued; `score` met `threshold` when both are set.
    Passed {
        /// The step's score, if it is scored.
        score: Option<Probability>,
        /// The step's threshold, if it has one.
        threshold: Option<Probability>,
    },
    /// The step declined, or its score missed its threshold.
    Declined {
        /// Why.
        reason: DeclineReason,
        /// The step's score, when the decline was a threshold miss.
        score: Option<Probability>,
        /// The step's threshold, if it has one.
        threshold: Option<Probability>,
    },
    /// The step errored.
    Failed {
        /// The error rendered with `{:#?}`.
        error: String,
    },
}

/// One variant per step. Steps own their record type; the records here are the fields eval relies on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceRecord {
    /// From `Classify`.
    Classify(ClassifyRecord),
    /// From `Retrieve`.
    Retrieve(RetrieveRecord),
    /// From `Rerank`.
    Rerank(RerankRecord),
    /// From `Answer`.
    Answer(AnswerRecord),
    /// From `Explain`.
    Explain(ExplainRecord),
    /// From `Verify`.
    Verify(VerifyRecord),
}

/// A chunk in a retrieve or rerank result, in result order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedChunk {
    /// The exact version used.
    pub chunk: ChunkRef,
    /// Retrieval's fused score. Rerank records written before `probability` existed hold System One's
    /// probability here instead.
    pub score: Relevance,
    /// System One's probability that the chunk helps answer the question, when `Rerank` judged it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability: Option<Probability>,
    /// Set when retrieval added the chunk as context next to a match rather than ranking it: the id of that match.
    /// Records written before it existed list every chunk as a match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_for: Option<ChunkId>,
}

/// What `Classify` asked for and got.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifyRecord {
    /// System One's answer as JSON; `None` when no System One is configured or it gave no answer.
    pub raw: Option<String>,
    /// The category the run continued with; `None` when the answer was invalid.
    pub category: Option<CategoryKind>,
    /// What System One chose when its probability was below `classify.min_confidence` and the run fell back to `open`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_from: Option<CategoryKind>,
}

/// Recall@50 is computed from the matches in `chunks`; context chunks neither count nor take a place.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrieveRecord {
    /// Every retrieved chunk, best first.
    pub chunks: Vec<RankedChunk>,
    /// The glossary entries the question matched and the terms they added to the search. Records written before it
    /// existed have none.
    #[serde(default, skip_serializing_if = "Expansion::is_empty")]
    pub expansion: Expansion,
}

/// Hit@5 is computed from `chunks`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RerankRecord {
    /// System One's probability of yes per judged chunk id, as a JSON object; `None` without System One.
    pub raw: Option<String>,
    /// The kept matches best first, then the context chunks attached to them.
    pub chunks: Vec<RankedChunk>,
}

/// What `Answer` asked for and got.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnswerRecord {
    /// System One's response as JSON.
    pub raw: Option<String>,
    /// The verdict the step parsed from `raw`; `None` when parsing failed.
    pub parsed: Option<serde_json::Value>,
}

/// What `Explain` got back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplainRecord {
    /// The LLM's reply, verbatim.
    pub raw: Option<String>,
    /// The LLM flagged that cited sources contradict the System One verdict.
    pub contradicts_verdict: bool,
}

/// What `Verify` rejected; both empty when the answer passed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyRecord {
    /// Claims with no citation.
    pub uncited_claims: Vec<String>,
    /// Cited chunks that are not among this run's reranked sources.
    pub unknown_citations: Vec<ChunkId>,
}

macro_rules! into_trace_record {
    ($($record:ident => $variant:ident),* $(,)?) => {
        $(impl From<$record> for TraceRecord {
            fn from(record: $record) -> Self {
                Self::$variant(record)
            }
        })*
    };
}

into_trace_record! {
    ClassifyRecord => Classify,
    RetrieveRecord => Retrieve,
    RerankRecord => Rerank,
    AnswerRecord => Answer,
    ExplainRecord => Explain,
    VerifyRecord => Verify,
}
