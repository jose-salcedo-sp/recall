//! What a finished run looks like to the server, the run store, and eval.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{CategoryKind, ChunkRef, Cited, Probability, Question, RunId};

use super::{DeclineReason, trace::Trace};

/// A finished run, whatever its outcome: what [`crate::store::RunStore`] persists and the server renders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    /// The run's id.
    pub id: RunId,
    /// The question asked.
    pub question: Question,
    /// What `Classify` chose, even when the run took the `open` path; `None` when it halted before classifying.
    pub category: Option<CategoryKind>,
    /// Answered, declined or failed.
    pub outcome: Outcome,
    /// Every step entry; audit data, never in the chat response.
    pub trace: Trace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
/// How a run ended; stored as `pipeline_runs.status` and `outcome`.
pub enum Outcome {
    /// A verified answer.
    Answered(Answer),
    /// Drew could not answer with certainty.
    Declined {
        /// The step that declined.
        at: String,
        /// Why.
        reason: DeclineReason,
        /// The step's score, when the decline was a threshold miss.
        score: Option<Probability>,
        /// The step's threshold, if it has one.
        threshold: Option<Probability>,
        /// Best reranked sources, shown under the fixed decline message.
        sources: Vec<Citation>,
    },
    /// An error stopped the run. Users see a fixed message and the run id, never `error`.
    Failed {
        /// The step that failed.
        at: String,
        /// The error rendered with `{:#?}`.
        error: String,
    },
}

/// Built only from a verified run, so every claim and option cites a chunk in `citations` or the prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// `None` for `open`.
    pub verdict: Option<Verdict>,
    /// The explanation, one cited claim per entry, in order.
    pub why: Vec<Cited<String>>,
    /// Numbered in order: the first citation renders as `[1]`.
    pub citations: Vec<Citation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "category", rename_all = "snake_case")]
/// System One's "what", per decisive category.
pub enum Verdict {
    /// Yes or no.
    Boolean {
        /// The answer.
        yes: bool,
        /// Probability of the given answer, so at least `0.5`.
        confidence: Probability,
    },
    /// One option chosen from the candidates.
    Choice {
        /// The chosen option.
        selected: Cited<String>,
        /// Every other candidate, with why it lost.
        rejected: Vec<Rejected>,
    },
    /// Items ordered by a criterion from the question.
    Ranking {
        /// The criterion, as stated in the question.
        criterion: String,
        /// Best first.
        items: Vec<Cited<String>>,
    },
}

/// A choice candidate that was not selected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rejected {
    /// The candidate.
    pub option: Cited<String>,
    /// One line on why it lost.
    pub reason: String,
}

/// A source as shown to the user. Every string comes from ingested text, so renderers escape it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// The exact chunk version cited.
    pub chunk: ChunkRef,
    /// The document's title.
    pub title: String,
    /// Deep link to the section.
    pub url: String,
    /// The chunk's first 200 characters, with `…` when cut.
    pub excerpt: String,
}

/// Runs the whole pipeline for one question. Never errors: failures are an `Outcome`.
#[async_trait]
pub trait Answerer: Send + Sync {
    /// Answers, declines or fails, always with a report to persist.
    async fn answer(&self, question: Question) -> RunReport;
}

/// Returns the same outcome for every question, under a fresh run id.
pub struct CannedAnswerer {
    /// Reported as [`RunReport::category`].
    pub category: Option<CategoryKind>,
    /// Reported as [`RunReport::outcome`].
    pub outcome: Outcome,
}

#[async_trait]
impl Answerer for CannedAnswerer {
    async fn answer(&self, question: Question) -> RunReport {
        RunReport {
            id: RunId::new(),
            question,
            category: self.category,
            outcome: self.outcome.clone(),
            trace: Trace::default(),
        }
    }
}
