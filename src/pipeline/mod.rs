//! Typestate pipeline with a Writer-style trace. See `docs/design.md` section 5.
//!
//! This module owns the state machine: a [`Run`] carries its category and state as types, [`Run::then`] applies a
//! [`Step`], times it, appends its [`trace::TraceEntry`] and checks its threshold, and a decline or failure becomes
//! [`Halted`] with the trace so far. [`steps`] holds the concrete steps and the per-category pipeline types;
//! [`answerer::PipelineAnswerer`] drives a question through them; [`report`] is what is persisted and shown. Steps never
//! see the trace, and model clients stay behind the ports in [`crate::llm`] and [`crate::retrieval`].
//!
//! Invalid pipelines do not compile. There is no `Answer<Open>` step, so `Decisive<Open>` is not a pipeline:
//!
//! ```compile_fail,E0599
//! use drew::pipeline::{Open, Pipeline, Run, steps::{Decisive, Reranked}};
//!
//! async fn decide(pipeline: &Decisive<Open>, run: Run<Open, Reranked>) {
//!     let _ = pipeline.drive(run).await;
//! }
//! ```
//!
//! Only the classification join can change a run's category:
//!
//! ```compile_fail,E0624
//! use drew::pipeline::{Asked, Boolean, Run, Unclassified};
//!
//! fn skip_classify(run: Run<Unclassified, Asked>) -> Run<Boolean, Asked> {
//!     run.classified()
//! }
//! ```
//!
//! Only its own step can build a step's output, so a later step cannot be handed sources, a verdict, or claims
//! that no step produced. Each of these does not compile:
//!
//! ```compile_fail,E0423
//! use drew::{domain::AtLeastOne, pipeline::steps::Candidates, retrieval::Retrieved};
//!
//! fn forge(hits: AtLeastOne<Retrieved>) -> Candidates { Candidates(hits) }
//! ```
//!
//! ```compile_fail,E0423
//! use drew::{pipeline::steps::Reranked, retrieval::Retrieved};
//!
//! fn forge(hits: Vec<Retrieved>) -> Reranked { Reranked(hits) }
//! ```
//!
//! ```compile_fail,E0451
//! use drew::{domain::Probability, pipeline::steps::{Decided, Reranked}};
//!
//! fn forge(sources: Reranked, confidence: Probability) -> Decided {
//!     Decided { yes: true, confidence, sources }
//! }
//! ```
//!
//! ```compile_fail,E0451
//! use drew::pipeline::steps::{Explained, Explanation, Reranked};
//!
//! fn forge(explanation: Explanation, sources: Reranked) -> Explained {
//!     Explained { explanation, sources }
//! }
//! ```
//!
//! ```compile_fail,E0451
//! use drew::{domain::Probability, pipeline::steps::{Explained, ExplainedVerdict}};
//!
//! fn forge(explained: Explained, confidence: Probability) -> ExplainedVerdict {
//!     ExplainedVerdict { explained, yes: true, confidence }
//! }
//! ```
//!
//! ```compile_fail,E0451
//! use drew::pipeline::steps::{Reranked, Verified, VerifiedAnswer};
//!
//! fn forge(answer: VerifiedAnswer, sources: Reranked) -> Verified {
//!     Verified { answer, sources }
//! }
//! ```
//!
//! ```compile_fail,E0451
//! use drew::{domain::Probability, pipeline::steps::{Verified, VerifiedVerdict}};
//!
//! fn forge(verified: Verified, confidence: Probability) -> VerifiedVerdict {
//!     VerifiedVerdict { verified, yes: true, confidence }
//! }
//! ```
//!
//! ```compile_fail,E0451
//! use drew::{domain::Cited, pipeline::steps::VerifiedAnswer};
//!
//! fn forge(claims: Vec<Cited<String>>) -> VerifiedAnswer { VerifiedAnswer { claims } }
//! ```

pub mod answerer;
pub mod report;
pub mod steps;
pub mod trace;

use std::{future::Future, marker::PhantomData, time::Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Probability, Question, RunId},
    llm::ModelRef,
};

use self::trace::{StepOutcome, Trace, TraceEntry, TraceRecord};

/// A question category as a type. Implemented by [`Unclassified`] and one marker per [`CategoryKind`].
pub trait Category: Send + Sync + 'static {
    /// `None` before classification.
    const KIND: Option<CategoryKind>;
}

/// Category of a run before `Classify`; only the classification join moves a run out of it.
pub struct Unclassified;
/// Marker for [`CategoryKind::Boolean`] runs.
pub struct Boolean;
/// Marker for [`CategoryKind::Choice`] runs.
pub struct Choice;
/// Marker for [`CategoryKind::Ranking`] runs.
pub struct Ranking;
/// Marker for [`CategoryKind::Open`] runs.
pub struct Open;

impl Category for Unclassified {
    const KIND: Option<CategoryKind> = None;
}

macro_rules! category {
    ($($marker:ident),*) => {
        $(impl Category for $marker {
            const KIND: Option<CategoryKind> = Some(CategoryKind::$marker);
        })*
    };
}

category!(Boolean, Choice, Ranking, Open);

/// Initial state of every run: only the question is known.
#[derive(Clone, Copy, Debug)]
pub struct Asked;

/// One question's progress through the pipeline. `TCategory` is a [`Category`] marker and `TState` is the last
/// step's output, so a step applies only where its input type matches.
pub struct Run<TCategory, TState> {
    id: RunId,
    question: Question,
    state: TState,
    trace: Trace,
    category: PhantomData<TCategory>,
}

impl Run<Unclassified, Asked> {
    /// A fresh run with a new [`RunId`] and an empty trace.
    pub fn new(question: Question) -> Self {
        Self {
            id: RunId::new(),
            question,
            state: Asked,
            trace: Trace::default(),
            category: PhantomData,
        }
    }
}

/// A step is generic over the categories it supports (`Answer<Boolean>` is a step, `Answer<Open>` is not),
/// rather than `Step` being generic over the category: that keeps the blanket `Pipeline` impl coherent with `Seq`.
pub trait Step: Send + Sync {
    /// Recorded as [`trace::TraceEntry::step`] and [`Halted::at`]; snake case.
    const NAME: &'static str;
    /// The run category this step applies to.
    type Category: Category;
    /// The run state the step consumes.
    type Input: Send;
    /// The run state the step produces.
    type Output: Send;
    /// The step's own trace data.
    type Record: Into<TraceRecord> + Send;

    /// Minimum score to continue. A step with a threshold must return a score.
    fn threshold(&self) -> Option<Probability> {
        None
    }

    /// Does the step's work. A domain decline returns [`StepError::decline`] with the record attached; any other
    /// error is a failure.
    fn run(
        &self,
        question: &Question,
        input: Self::Input,
    ) -> impl Future<Output = Result<Scored<Self::Output, Self::Record>, StepError>> + Send;
}

/// A step's successful result: its output, its trace record, and what [`Run::then`] checks and records.
pub struct Scored<TOutput, TRecord> {
    /// The next run state.
    pub output: TOutput,
    /// Appended to the trace.
    pub record: TRecord,
    /// Compared with [`Step::threshold`]; required when the step has one.
    pub score: Option<Probability>,
    /// The model the step called, if any.
    pub model: Option<ModelRef>,
    /// Version of the prompt sent to `model`.
    pub prompt_version: Option<&'static str>,
}

impl<TOutput, TRecord> Scored<TOutput, TRecord> {
    /// Unscored, with no model.
    pub fn new(output: TOutput, record: TRecord) -> Self {
        Self {
            output,
            record,
            score: None,
            model: None,
            prompt_version: None,
        }
    }

    /// Sets [`Scored::score`].
    pub fn with_score(self, score: Probability) -> Self {
        Self {
            score: Some(score),
            ..self
        }
    }

    /// Sets the model and the prompt version sent to it.
    pub fn with_model(self, model: ModelRef, prompt_version: &'static str) -> Self {
        Self {
            model: Some(model),
            prompt_version: Some(prompt_version),
            ..self
        }
    }
}

/// Why a run declined: Drew answers "can't answer with certainty" rather than guess.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum DeclineReason {
    /// A step's score was below its threshold; set by [`Run::then`], never by a step.
    LowConfidence,
    /// Model output failed DTO deserialization or `TryFrom` conversion.
    InvalidModelOutput(String),
    /// Retrieval or reranking left no source to answer from.
    NoSources,
    /// The answer could not be verified against this run's sources, or the model found no answer in them.
    Unverified(String),
    /// `Explain` found that the sources it would cite contradict System One's verdict.
    ContradictsVerdict,
}

/// A step that cannot produce its output. The record, when present, keeps the raw model output in the trace.
#[derive(Debug)]
pub struct StepError {
    /// Decline or failure.
    pub kind: StepErrorKind,
    /// Appended to the trace in place of the step's normal record.
    pub record: Option<TraceRecord>,
    /// The model the step called, if any.
    pub model: Option<ModelRef>,
    /// Version of the prompt sent to `model`.
    pub prompt_version: Option<&'static str>,
}

/// Whether a [`StepError`] is a domain decline or a failure; the run's status follows from it.
#[derive(Debug)]
pub enum StepErrorKind {
    /// The step ran but cannot answer with certainty.
    Decline(DeclineReason),
    /// Anything else, such as a model or database error; any `?` in a step produces this.
    Failed(eros::ErrorUnion),
}

impl StepError {
    /// A decline with no record yet.
    pub fn decline(reason: DeclineReason) -> Self {
        Self {
            kind: StepErrorKind::Decline(reason),
            record: None,
            model: None,
            prompt_version: None,
        }
    }

    /// Attaches the trace record, so the raw model output survives a decline.
    pub fn with_record(self, record: impl Into<TraceRecord>) -> Self {
        Self {
            record: Some(record.into()),
            ..self
        }
    }

    /// Sets the model and the prompt version sent to it.
    pub fn with_model(self, model: ModelRef, prompt_version: &'static str) -> Self {
        Self {
            model: Some(model),
            prompt_version: Some(prompt_version),
            ..self
        }
    }
}

impl<E: Into<eros::ErrorUnion>> From<E> for StepError {
    fn from(error: E) -> Self {
        Self {
            kind: StepErrorKind::Failed(error.into()),
            record: None,
            model: None,
            prompt_version: None,
        }
    }
}

/// A run that stopped at a step: declined or failed. It still carries the whole trace up to and including that step.
#[derive(Debug)]
pub struct Halted {
    /// The run's id.
    pub id: RunId,
    /// The question asked.
    pub question: Question,
    /// Every entry so far, including the halting step's.
    pub trace: Trace,
    /// `None` when the run halted before classification.
    pub category: Option<CategoryKind>,
    /// [`Step::NAME`] of the step that halted.
    pub at: &'static str,
    /// Why it halted.
    pub halt: Halt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// How a run halted.
pub enum Halt {
    /// Drew cannot answer with certainty.
    Declined {
        /// Why.
        reason: DeclineReason,
        /// The step's score, when the decline was a threshold miss.
        score: Option<Probability>,
        /// The step's threshold, if it has one.
        threshold: Option<Probability>,
    },
    /// An error stopped the run.
    Failed {
        /// The error rendered with `{:#?}`: root, causes and context. Audit data; never shown to users.
        error: String,
    },
}

impl<TCategory: Category, TState: Send> Run<TCategory, TState> {
    /// The run's id, fixed at [`Run::new`].
    pub fn id(&self) -> RunId {
        self.id
    }

    /// The question asked.
    pub fn question(&self) -> &Question {
        &self.question
    }

    /// The last step's output.
    pub fn state(&self) -> &TState {
        &self.state
    }

    /// Every step entry so far, in the order they were appended.
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// Ends the run, for building its report.
    pub fn into_parts(self) -> (RunId, Question, TState, Trace) {
        (self.id, self.question, self.state, self.trace)
    }

    /// Times the step, appends its trace entry, checks its threshold, and continues or halts.
    ///
    /// # Errors
    ///
    /// [`Halted`] when the step declines or fails, its score is below its threshold, or it has a threshold but
    /// returned no score.
    pub async fn then<TStep>(self, step: &TStep) -> Result<Run<TCategory, TStep::Output>, Halted>
    where
        TStep: Step<Category = TCategory, Input = TState>,
    {
        let Self {
            id,
            question,
            state,
            mut trace,
            ..
        } = self;
        let (entry, result) = execute(step, &question, state, None).await;
        trace.push(entry);
        match result {
            Ok(output) => Ok(Run {
                id,
                question,
                state: output,
                trace,
                category: PhantomData,
            }),
            Err(halt) => Err(Halted {
                id,
                question,
                trace,
                category: TCategory::KIND,
                at: TStep::NAME,
                halt,
            }),
        }
    }

    /// Runs two steps concurrently on the same state. Both entries are traced even when one halts, left first, with
    /// `group` as their [`trace::TraceEntry::group`].
    ///
    /// # Errors
    ///
    /// [`Halted`] as for [`Run::then`]; when both halt, the left step's halt is reported.
    pub async fn both<TLeft, TRight>(
        self,
        group: &'static str,
        left: &TLeft,
        right: &TRight,
    ) -> Result<Run<TCategory, (TLeft::Output, TRight::Output)>, Halted>
    where
        TState: Clone,
        TLeft: Step<Category = TCategory, Input = TState>,
        TRight: Step<Category = TCategory, Input = TState>,
    {
        let Self {
            id,
            question,
            state,
            mut trace,
            ..
        } = self;
        let ((left_entry, left_result), (right_entry, right_result)) = futures::join!(
            execute(left, &question, state.clone(), Some(group)),
            execute(right, &question, state, Some(group)),
        );
        trace.push(left_entry);
        trace.push(right_entry);
        let halted = |trace, at, halt| Halted {
            id,
            question: question.clone(),
            trace,
            category: TCategory::KIND,
            at,
            halt,
        };
        match (left_result, right_result) {
            (Ok(left), Ok(right)) => Ok(Run {
                id,
                question: question.clone(),
                state: (left, right),
                trace,
                category: PhantomData,
            }),
            (Err(halt), _) => Err(halted(trace, TLeft::NAME, halt)),
            (_, Err(halt)) => Err(halted(trace, TRight::NAME, halt)),
        }
    }

    /// Restructures the state without a step. Not traced, so only for plumbing between steps.
    pub(crate) fn map<TNext: Send>(self, f: impl FnOnce(TState) -> TNext) -> Run<TCategory, TNext> {
        Run {
            id: self.id,
            question: self.question,
            state: f(self.state),
            trace: self.trace,
            category: PhantomData,
        }
    }

    /// Re-tags the run with the category decided by `Classify`. Only the classification join may call this.
    pub(crate) fn classified<TNext: Category>(self) -> Run<TNext, TState> {
        Run {
            id: self.id,
            question: self.question,
            state: self.state,
            trace: self.trace,
            category: PhantomData,
        }
    }
}

async fn execute<TStep: Step>(
    step: &TStep,
    question: &Question,
    input: TStep::Input,
    group: Option<&'static str>,
) -> (TraceEntry, Result<TStep::Output, Halt>) {
    let started_at = Utc::now();
    let started = Instant::now();
    let result = step.run(question, input).await;
    let duration_ms = started.elapsed().as_millis() as u64;
    let threshold = step.threshold();

    let entry = |outcome, record, model, prompt_version: Option<&'static str>| TraceEntry {
        step: TStep::NAME.to_owned(),
        group: group.map(str::to_owned),
        started_at,
        duration_ms,
        model,
        prompt_version: prompt_version.map(str::to_owned),
        outcome,
        record,
    };

    match result {
        Ok(scored) => {
            let halt = match (scored.score, threshold) {
                (Some(score), Some(threshold)) if score < threshold => Some(Halt::Declined {
                    reason: DeclineReason::LowConfidence,
                    score: Some(score),
                    threshold: Some(threshold),
                }),
                (None, Some(_)) => Some(Halt::Failed {
                    error: format!("{} has a threshold but returned no score", TStep::NAME),
                }),
                _ => None,
            };
            let outcome = match &halt {
                None => StepOutcome::Passed {
                    score: scored.score,
                    threshold,
                },
                Some(halt) => outcome_of(halt),
            };
            let entry = entry(
                outcome,
                Some(scored.record.into()),
                scored.model,
                scored.prompt_version,
            );
            match halt {
                None => (entry, Ok(scored.output)),
                Some(halt) => (entry, Err(halt)),
            }
        }
        Err(error) => {
            let halt = match error.kind {
                StepErrorKind::Decline(reason) => Halt::Declined {
                    reason,
                    score: None,
                    threshold,
                },
                StepErrorKind::Failed(error) => Halt::Failed {
                    error: format!("{error:#?}"),
                },
            };
            (
                entry(
                    outcome_of(&halt),
                    error.record,
                    error.model,
                    error.prompt_version,
                ),
                Err(halt),
            )
        }
    }
}

fn outcome_of(halt: &Halt) -> StepOutcome {
    match halt.clone() {
        Halt::Declined {
            reason,
            score,
            threshold,
        } => StepOutcome::Declined {
            reason,
            score,
            threshold,
        },
        Halt::Failed { error } => StepOutcome::Failed { error },
    }
}

/// A typed path through the state machine. Every `Step` is a one-transition pipeline.
pub trait Pipeline: Send + Sync {
    /// The run category the pipeline applies to.
    type Category: Category;
    /// The run state the first step consumes.
    type Input: Send;
    /// The run state the last step produces.
    type Output: Send;

    /// Applies every step in order, stopping at the first halt.
    fn drive(
        &self,
        run: Run<Self::Category, Self::Input>,
    ) -> impl Future<Output = Result<Run<Self::Category, Self::Output>, Halted>> + Send;
}

impl<TStep: Step> Pipeline for TStep {
    type Category = TStep::Category;
    type Input = TStep::Input;
    type Output = TStep::Output;

    async fn drive(
        &self,
        run: Run<TStep::Category, TStep::Input>,
    ) -> Result<Run<TStep::Category, TStep::Output>, Halted> {
        run.then(self).await
    }
}

/// Composes two pipelines; only a valid pipeline when the first one's output is the second one's input.
///
/// ```compile_fail,E0599
/// use drew::{domain::Question, pipeline::{trace::VerifyRecord, *}};
///
/// struct A;
/// struct B;
/// struct ToA;
///
/// impl Step for ToA {
///     const NAME: &'static str = "to_a";
///     type Category = Open;
///     type Input = B;
///     type Output = A;
///     type Record = VerifyRecord;
///
///     async fn run(&self, _: &Question, _: B) -> Result<Scored<A, VerifyRecord>, StepError> {
///         unimplemented!()
///     }
/// }
///
/// async fn twice(run: Run<Open, B>) {
///     // ToA outputs A, but the second ToA needs B.
///     let _ = Seq(ToA, ToA).drive(run).await;
/// }
/// ```
pub struct Seq<TFirst, TRest>(pub TFirst, pub TRest);

impl<TFirst, TRest> Pipeline for Seq<TFirst, TRest>
where
    TFirst: Pipeline,
    TRest: Pipeline<Category = TFirst::Category, Input = TFirst::Output>,
{
    type Category = TFirst::Category;
    type Input = TFirst::Input;
    type Output = TRest::Output;

    async fn drive(
        &self,
        run: Run<TFirst::Category, TFirst::Input>,
    ) -> Result<Run<TFirst::Category, TRest::Output>, Halted> {
        self.1.drive(self.0.drive(run).await?).await
    }
}

#[cfg(test)]
mod tests;
