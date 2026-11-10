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
    /// From `Admit`.
    Admit(AdmitRecord),
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

/// How `Admit` screened each reranked source, in `Rerank`'s order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdmitRecord {
    /// Every source `Rerank` kept, matches and their context, with its probabilities and where it was routed.
    pub sources: Vec<AdmittedSource>,
}

/// One source's admission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdmittedSource {
    /// The exact version screened.
    pub chunk: ChunkRef,
    /// What the judge answered about it.
    pub screen: Screen,
    /// Where it went.
    pub route: AdmissionRoute,
}

/// The judge's probabilities for one source. A match is asked all four questions; context, which rides with its
/// match, is only screened for injection.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Screen {
    /// A match `Rerank` ranked.
    Match {
        /// That the text tries to change the assistant's instructions.
        injection: Probability,
        /// That it addresses the question's subject.
        relevant: Probability,
        /// That it contradicts a premise of the question.
        contradicts: Probability,
        /// That it states something usable in a direct answer.
        evidence: Probability,
    },
    /// Context retrieval attached to a match.
    Context {
        /// That the text tries to change the assistant's instructions.
        injection: Probability,
    },
}

/// Where `Admit` routed a source. `Admitted` and `Conflict` reach `Explain`; the rest are excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionRoute {
    /// Kept: a match with evidence, or context of a kept match.
    Admitted,
    /// Kept: a relevant match that contradicts the question's premise, so the answer can say so.
    Conflict,
    /// Excluded: the text tries to change the assistant's instructions.
    Injection,
    /// Excluded: the match does not address the question's subject.
    Irrelevant,
    /// Excluded: the match is on topic but states nothing usable in an answer.
    NoEvidence,
    /// Excluded: context of an excluded match.
    MatchExcluded,
}

impl AdmissionRoute {
    /// Whether the source reaches `Explain`.
    pub fn is_kept(self) -> bool {
        matches!(self, Self::Admitted | Self::Conflict)
    }
}

/// What `Verify` checked and rejected. `uncited_claims` and `unknown_citations` are both empty when every claim
/// passed the membership check.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VerifyRecord {
    /// Claims with no citation.
    pub uncited_claims: Vec<String>,
    /// Cited chunks that are not among this run's reranked sources.
    pub unknown_citations: Vec<ChunkId>,
    /// The check each cited claim got beyond membership. Records written before it existed checked membership only.
    #[serde(default)]
    pub support: SupportMode,
    /// One verdict per claim that passed the membership check, in claim order; empty under `membership`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claims: Vec<ClaimCheck>,
    /// How many claims were left out of the published answer under the `strip` policy.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub stripped: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// The check `Verify` ran on each cited claim, after resolving the config against the models available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportMode {
    /// Citations only.
    #[default]
    Membership,
    /// Citations and quoted spans, with no model.
    Quotes,
    /// Citations, quoted spans, and a judge asked whether a cited chunk supports the claim.
    Judged,
}

/// One claim's verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaimCheck {
    /// The claim as `Explain` wrote it.
    pub claim: String,
    /// What the checks found.
    pub verdict: ClaimVerdict,
}

/// Whether a cited claim is backed by the chunks it cites.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum ClaimVerdict {
    /// Every quoted span is in a cited chunk; under `quotes`, no judge was asked.
    QuotesFound,
    /// A quoted span appears in none of the cited chunks; no judge was asked.
    Fabricated {
        /// The span, without its quotation marks.
        span: String,
    },
    /// A cited chunk supports the claim; chunks after it were not judged.
    Supported {
        /// The supporting judgement.
        by: Relation,
    },
    /// No cited chunk supports the claim, and one contradicts it.
    Contradicted {
        /// The contradicting judgement.
        by: Relation,
    },
    /// No cited chunk supports or contradicts the claim with enough probability.
    Unsupported {
        /// One judgement per cited chunk, in citation order.
        relations: Vec<Relation>,
    },
}

impl ClaimVerdict {
    /// Whether the claim may be published.
    pub fn passed(&self) -> bool {
        matches!(self, Self::QuotesFound | Self::Supported { .. })
    }
}

/// The judge's reading of how one cited chunk relates to a claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// The chunk judged.
    pub chunk: ChunkId,
    /// The most probable relation.
    pub relation: RelationKind,
    /// Its probability.
    pub probability: Probability,
}

/// How a chunk relates to a claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// The chunk supports the claim.
    Supports,
    /// The chunk contradicts the claim.
    Contradicts,
    /// The chunk is silent on the claim.
    SaysNothing,
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
    AdmitRecord => Admit,
    AnswerRecord => Answer,
    ExplainRecord => Explain,
    VerifyRecord => Verify,
}
