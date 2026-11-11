//! The audit trail of a run: one [`TraceEntry`] per step, appended by [`super::Run::then`]. Records reference chunks
//! by id and version and never copy their text.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    access::{Access, AccessDecision, PolicyRef, Principal},
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
    /// One access check per chunk the step touched, in order; empty for a step that reads no chunk. Entries written
    /// before access was enforced have none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub access: Vec<AccessDecision>,
}

impl Trace {
    /// The trace without denied access checks, for a reader who may only see its own runs: a denied check names a
    /// chunk the asker may not know exists.
    pub fn without_denials(mut self) -> Self {
        for entry in &mut self.0 {
            entry.access.retain(|check| check.decision.is_allowed());
        }
        self
    }
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
    /// From [`super::Run::authorize`], the first entry of every run.
    Authorize(AuthorizeRecord),
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

/// Who asked, which policy decided, and what it gave them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorizeRecord {
    /// Who asked, with the roles their authenticator asserted.
    pub principal: Principal,
    /// The policy that decided.
    pub policy: PolicyRef,
    /// What the policy gave the principal; `None` when it could not answer.
    pub access: Option<Access>,
}

/// A chunk in a retrieve or rerank result, in result order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedChunk {
    /// The exact version used.
    pub chunk: ChunkRef,
    /// Retrieval's fused score. Rerank records written before `probability` existed hold the judge's
    /// probability here instead.
    pub score: Relevance,
    /// The judge's probability that the chunk helps answer the question, when `Rerank` judged it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability: Option<Probability>,
    /// Set when retrieval added the chunk as context next to a match rather than ranking it: the id of that match.
    /// Records written before it existed list every chunk as a match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_for: Option<ChunkId>,
}

/// How `Classify` chose the category, as the pipeline was assembled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifyMode {
    /// A judge chose the category.
    #[default]
    Judged,
    /// No judge is configured, so every question is `open`, unscored.
    AllOpen,
}

/// How `Rerank` ordered the matches, as the pipeline was assembled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankMode {
    /// Retrieval rank and the evidence penalty only; unscored.
    #[default]
    RetrievalOrder,
    /// A judge's probability per match, fused with retrieval rank; scored.
    Judged,
}

/// What `Classify` asked for and got. Stored as `{mode, raw, category, fallback_from}`; a record written before
/// `mode` existed is `all_open` when its `raw` is null, else judged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "RawClassifyRecord", into = "RawClassifyRecord")]
pub enum ClassifyRecord {
    /// No judge is configured: the run continued as `open`, unscored.
    AllOpen,
    /// A judge chose the category.
    Judged {
        /// The judge's answer as JSON.
        raw: String,
        /// The category the run continued with; `None` when the answer was invalid.
        category: Option<CategoryKind>,
        /// What the judge chose when its probability was below `classify.min_confidence` and the run fell back to
        /// `open`.
        fallback_from: Option<CategoryKind>,
    },
}

impl ClassifyRecord {
    /// How the category was chosen.
    pub fn mode(&self) -> ClassifyMode {
        match self {
            Self::AllOpen => ClassifyMode::AllOpen,
            Self::Judged { .. } => ClassifyMode::Judged,
        }
    }

    /// The category the run continued with; `None` when the judge's answer was invalid.
    pub fn category(&self) -> Option<CategoryKind> {
        match self {
            Self::AllOpen => Some(CategoryKind::Open),
            Self::Judged { category, .. } => *category,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RawClassifyRecord {
    #[serde(default)]
    mode: ClassifyMode,
    raw: Option<String>,
    category: Option<CategoryKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback_from: Option<CategoryKind>,
}

impl From<RawClassifyRecord> for ClassifyRecord {
    fn from(record: RawClassifyRecord) -> Self {
        match (record.mode, record.raw) {
            (ClassifyMode::AllOpen, _) | (ClassifyMode::Judged, None) => Self::AllOpen,
            (ClassifyMode::Judged, Some(raw)) => Self::Judged {
                raw,
                category: record.category,
                fallback_from: record.fallback_from,
            },
        }
    }
}

impl From<ClassifyRecord> for RawClassifyRecord {
    fn from(record: ClassifyRecord) -> Self {
        match record {
            ClassifyRecord::AllOpen => Self {
                mode: ClassifyMode::AllOpen,
                raw: None,
                category: Some(CategoryKind::Open),
                fallback_from: None,
            },
            ClassifyRecord::Judged {
                raw,
                category,
                fallback_from,
            } => Self {
                mode: ClassifyMode::Judged,
                raw: Some(raw),
                category,
                fallback_from,
            },
        }
    }
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

/// Hit@5 is computed from `chunks`. Stored as `{mode, raw, chunks}`; a record written before `mode` existed was
/// judged exactly when its `raw` is set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "RawRerankRecord", into = "RawRerankRecord")]
pub struct RerankRecord {
    /// How the matches were ordered.
    pub order: RerankOrder,
    /// The kept matches best first, then the context chunks attached to them.
    pub chunks: Vec<RankedChunk>,
}

/// How `Rerank` ordered one run's matches.
#[derive(Clone, Debug, PartialEq)]
pub enum RerankOrder {
    /// Retrieval rank and the evidence penalty only.
    RetrievalOrder,
    /// Fused with a judge's probabilities.
    Judged {
        /// The judge's probability of yes per judged chunk id, as a JSON object.
        raw: String,
    },
}

impl RerankOrder {
    /// The mode, as the pipeline was assembled.
    pub fn mode(&self) -> RerankMode {
        match self {
            Self::RetrievalOrder => RerankMode::RetrievalOrder,
            Self::Judged { .. } => RerankMode::Judged,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RawRerankRecord {
    #[serde(default)]
    mode: RerankMode,
    raw: Option<String>,
    chunks: Vec<RankedChunk>,
}

impl From<RawRerankRecord> for RerankRecord {
    fn from(record: RawRerankRecord) -> Self {
        Self {
            order: match record.raw {
                Some(raw) => RerankOrder::Judged { raw },
                None => RerankOrder::RetrievalOrder,
            },
            chunks: record.chunks,
        }
    }
}

impl From<RerankRecord> for RawRerankRecord {
    fn from(record: RerankRecord) -> Self {
        let mode = record.order.mode();
        let raw = match record.order {
            RerankOrder::RetrievalOrder => None,
            RerankOrder::Judged { raw } => Some(raw),
        };
        Self {
            mode,
            raw,
            chunks: record.chunks,
        }
    }
}

/// What `Answer` asked for and got.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnswerRecord {
    /// The judge's answer as JSON.
    pub raw: Option<String>,
    /// The verdict the step parsed from `raw`; `None` when parsing failed.
    pub parsed: Option<serde_json::Value>,
}

/// What `Explain` got back, by the path it ran. Stored as `{raw}` on the open path and
/// `{raw, contradicts_verdict}` when it explained a verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "RawExplainRecord", into = "RawExplainRecord")]
pub enum ExplainRecord {
    /// The whole answer to an `open` question.
    Open {
        /// The LLM's reply, verbatim.
        raw: String,
    },
    /// The reason for the judge's verdict.
    Verdict {
        /// The LLM's reply, verbatim.
        raw: String,
        /// The LLM flagged that cited sources contradict the verdict.
        contradicts_verdict: bool,
    },
}

impl ExplainRecord {
    /// The LLM's reply, verbatim.
    pub fn raw(&self) -> &str {
        match self {
            Self::Open { raw } | Self::Verdict { raw, .. } => raw,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RawExplainRecord {
    raw: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    contradicts_verdict: Option<bool>,
}

impl From<RawExplainRecord> for ExplainRecord {
    fn from(record: RawExplainRecord) -> Self {
        match record.contradicts_verdict {
            None => Self::Open { raw: record.raw },
            Some(contradicts_verdict) => Self::Verdict {
                raw: record.raw,
                contradicts_verdict,
            },
        }
    }
}

impl From<ExplainRecord> for RawExplainRecord {
    fn from(record: ExplainRecord) -> Self {
        match record {
            ExplainRecord::Open { raw } => Self {
                raw,
                contradicts_verdict: None,
            },
            ExplainRecord::Verdict {
                raw,
                contradicts_verdict,
            } => Self {
                raw,
                contradicts_verdict: Some(contradicts_verdict),
            },
        }
    }
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

/// The check `Verify` ran on each cited claim: `verify.support` as configured, since a pipeline whose check needs a
/// judge does not assemble without one.
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
    AuthorizeRecord => Authorize,
    ClassifyRecord => Classify,
    RetrieveRecord => Retrieve,
    RerankRecord => Rerank,
    AdmitRecord => Admit,
    AnswerRecord => Answer,
    ExplainRecord => Explain,
    VerifyRecord => Verify,
}
