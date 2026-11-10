//! The provenance judge: yes/no and one-of questions over a text, answered with probabilities.
//!
//! [`Judge`] is the port every judged step asks through: `Classify`, `Rerank`, `Answer`, `Admit` and `Verify`. It
//! names no vendor and no wire format: question ids, decision shapes and the backend's dialect belong to the
//! adapter. [`SystemOneJudge`] adapts any [`SystemOne`] backend (Kev, system-one.dev, Clef on Workers AI); another judge, such as an LLM asked over
//! [`super::ChatModel`], is one more adapter.

// The wire format is this adapter's own; the fence in `clippy.toml` keeps it out of the pipeline.
#![allow(clippy::disallowed_types)]

use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use serde::Serialize;

use crate::domain::{NotAProbability, Probability};

use super::{
    ModelRef,
    system_one::{Decision, DecisionAnswer, DecisionRequest, Options, SystemOne},
};

/// Several questions about one text.
#[derive(Clone, Debug, PartialEq)]
pub struct JudgeRequest {
    /// The text every question is about. Ingested text: the adapter delimits it as data.
    pub text: String,
    /// Answered in this order.
    pub questions: Vec<JudgeQuestion>,
}

/// One question, and the kind of answer it takes.
#[derive(Clone, Debug, PartialEq)]
pub enum JudgeQuestion {
    /// Answered with the probability of yes.
    YesNo {
        /// The question, phrased for a yes/no answer.
        instructions: String,
    },
    /// Answered with a probability per option.
    OneOf {
        /// The question.
        instructions: String,
        /// At least one option, in the order they are offered; option order can change an answer.
        options: Vec<JudgeOption>,
    },
}

/// An option of [`JudgeQuestion::OneOf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JudgeOption {
    /// Unique within its question.
    pub name: &'static str,
    /// What choosing it means.
    pub description: &'static str,
}

/// A judge's answer to one question, unchecked. Read through [`Judgements::yes`] and [`Judgements::one_of`];
/// serialized as is into trace records.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RawJudgement {
    /// Probability of yes.
    YesNo(f32),
    /// Probability per option, in the question's option order.
    OneOf(Vec<f32>),
    /// The judge gave no answer of the kind asked for.
    Missing,
}

/// A judge's answers, in question order. Probabilities are checked when read, so one bad answer declines the step
/// that reads it instead of failing the whole response.
#[derive(Clone, Debug, PartialEq)]
pub struct Judgements(Vec<RawJudgement>);

/// An answer that is missing, of another kind than asked, or whose probability is not one.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum InvalidJudgement {
    /// No answer of the kind asked for.
    #[error("expected a {expected} judgement for question {index}")]
    Unexpected {
        /// The question's position in the request.
        index: usize,
        /// The kind asked for.
        expected: &'static str,
    },
    /// A one-of answer with no options.
    #[error("the judgement for question {index} has no options")]
    NoOptions {
        /// The question's position in the request.
        index: usize,
    },
    /// A probability outside `0..=1` or `NaN`.
    #[error("judgement for question {index}: {source}")]
    OutOfRange {
        /// The question's position in the request.
        index: usize,
        /// The rejected value.
        #[source]
        source: NotAProbability,
    },
}

impl Judgements {
    /// Answers in question order; a question past the end reads as [`RawJudgement::Missing`].
    pub fn new(answers: Vec<RawJudgement>) -> Self {
        Self(answers)
    }

    /// The answer to the question at `index` as the judge gave it, for the trace.
    pub fn raw(&self, index: usize) -> &RawJudgement {
        self.0.get(index).unwrap_or(&RawJudgement::Missing)
    }

    /// The probability of yes for the yes/no question at `index`.
    ///
    /// # Errors
    ///
    /// [`InvalidJudgement::Unexpected`] or [`InvalidJudgement::OutOfRange`].
    pub fn yes(&self, index: usize) -> Result<Probability, InvalidJudgement> {
        match self.0.get(index) {
            Some(RawJudgement::YesNo(p)) => probability(index, *p),
            _ => Err(InvalidJudgement::Unexpected {
                index,
                expected: "yes/no",
            }),
        }
    }

    /// The most probable option of the one-of question at `index`, by position, and its probability. The chosen
    /// option's own probability, not an entropy-style confidence: a cutoff written against class probability then
    /// means what it says.
    ///
    /// # Errors
    ///
    /// Any [`InvalidJudgement`]; every option's probability is checked, not only the chosen one's.
    pub fn one_of(&self, index: usize) -> Result<(usize, Probability), InvalidJudgement> {
        let Some(RawJudgement::OneOf(probabilities)) = self.0.get(index) else {
            return Err(InvalidJudgement::Unexpected {
                index,
                expected: "one-of",
            });
        };
        let checked = probabilities
            .iter()
            .map(|&p| probability(index, p))
            .collect::<Result<Vec<_>, _>>()?;
        checked
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.get().total_cmp(&b.1.get()))
            .ok_or(InvalidJudgement::NoOptions { index })
    }
}

fn probability(index: usize, value: f32) -> Result<Probability, InvalidJudgement> {
    Probability::new(value).map_err(|source| InvalidJudgement::OutOfRange { index, source })
}

/// Answers [`JudgeQuestion`]s about a text with probabilities. It only picks among the options it is given.
#[async_trait]
pub trait Judge: Send + Sync {
    /// Recorded in the trace of every step that asks it.
    fn model(&self) -> ModelRef;

    /// Answers every question in `request`, in order.
    ///
    /// # Errors
    ///
    /// When the backend fails, times out, or returns a body that does not parse. Missing or out-of-range answers
    /// do not fail here; see [`Judgements`].
    async fn judge(&self, request: JudgeRequest) -> eros::Result<Judgements>;
}

/// [`Judge`] over any [`SystemOne`] backend: one decision request per judge request, questions keyed by position.
pub struct SystemOneJudge {
    system_one: Arc<dyn SystemOne>,
}

impl SystemOneJudge {
    /// Judges with `system_one`, whose timeouts apply.
    pub fn new(system_one: Arc<dyn SystemOne>) -> Self {
        Self { system_one }
    }
}

fn question_id(index: usize) -> String {
    format!("q{index}")
}

fn decision(question: &JudgeQuestion) -> Decision {
    match question {
        JudgeQuestion::YesNo { instructions } => Decision::Noul {
            instructions: instructions.clone(),
        },
        JudgeQuestion::OneOf {
            instructions,
            options,
        } => Decision::Choice {
            instructions: instructions.clone(),
            criteria: Options(
                options
                    .iter()
                    .map(|option| (option.name.to_owned(), Some(option.description.to_owned())))
                    .collect(),
            ),
        },
    }
}

fn judgement(question: &JudgeQuestion, answer: Option<&DecisionAnswer>) -> RawJudgement {
    match (question, answer) {
        (JudgeQuestion::YesNo { .. }, Some(DecisionAnswer::Noul { noul })) => {
            RawJudgement::YesNo(*noul)
        }
        (
            JudgeQuestion::OneOf { options, .. },
            Some(DecisionAnswer::Choice { probabilities, .. }),
        ) => options
            .iter()
            .map(|option| probabilities.get(option.name).copied())
            .collect::<Option<Vec<_>>>()
            .map_or(RawJudgement::Missing, RawJudgement::OneOf),
        _ => RawJudgement::Missing,
    }
}

#[async_trait]
impl Judge for SystemOneJudge {
    fn model(&self) -> ModelRef {
        self.system_one.model()
    }

    async fn judge(&self, request: JudgeRequest) -> eros::Result<Judgements> {
        let decisions = DecisionRequest {
            state: request.text,
            model: String::new(),
            questions: request
                .questions
                .iter()
                .enumerate()
                .map(|(index, question)| (question_id(index), decision(question)))
                .collect::<BTreeMap<_, _>>(),
        };
        let response = self.system_one.decide(decisions).await?;
        Ok(Judgements::new(
            request
                .questions
                .iter()
                .enumerate()
                .map(|(index, question)| {
                    judgement(question, response.answers.get(&question_id(index)))
                })
                .collect(),
        ))
    }
}

#[cfg(any(test, feature = "testing"))]
pub use super::fakes::FnJudge;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_of_picks_the_most_probable_option_and_reports_its_own_probability() {
        let judgements = Judgements::new(vec![RawJudgement::OneOf(vec![0.7939, 0.048, 0.1581])]);
        let (chosen, p) = judgements.one_of(0).unwrap();
        assert_eq!(chosen, 0);
        assert!((p.get() - 0.7939).abs() < 1e-6);
    }

    #[test]
    fn invalid_answers_are_rejected_when_read() {
        let judgements = Judgements::new(vec![
            RawJudgement::YesNo(1.7),
            RawJudgement::OneOf(vec![0.2, f32::NAN]),
            RawJudgement::OneOf(vec![]),
            RawJudgement::Missing,
        ]);
        assert!(matches!(
            judgements.yes(0),
            Err(InvalidJudgement::OutOfRange { index: 0, .. })
        ));
        assert!(matches!(
            judgements.one_of(1),
            Err(InvalidJudgement::OutOfRange { index: 1, .. })
        ));
        assert_eq!(
            judgements.one_of(2),
            Err(InvalidJudgement::NoOptions { index: 2 })
        );
        assert!(matches!(
            judgements.yes(3),
            Err(InvalidJudgement::Unexpected { index: 3, .. })
        ));
        assert!(matches!(
            judgements.one_of(0),
            Err(InvalidJudgement::Unexpected { index: 0, .. })
        ));
        assert!(matches!(
            judgements.yes(9),
            Err(InvalidJudgement::Unexpected { index: 9, .. })
        ));
    }

    #[tokio::test]
    async fn a_system_one_that_cannot_answer_fails_the_judgement() {
        let judge = SystemOneJudge::new(Arc::new(crate::llm::fakes::FailingModel));
        let request = JudgeRequest {
            text: "text".into(),
            questions: vec![JudgeQuestion::YesNo {
                instructions: "?".into(),
            }],
        };
        let error = judge.judge(request).await.unwrap_err();
        assert!(error.to_string().contains("connection refused"), "{error}");
    }

    #[test]
    fn a_choice_missing_an_option_reads_as_missing() {
        let question = JudgeQuestion::OneOf {
            instructions: "?".into(),
            options: vec![
                JudgeOption {
                    name: "a",
                    description: "A",
                },
                JudgeOption {
                    name: "b",
                    description: "B",
                },
            ],
        };
        let answer = DecisionAnswer::Choice {
            choice: "a".into(),
            confidence: 0.9,
            probabilities: BTreeMap::from([("a".into(), 0.9)]),
        };
        assert_eq!(judgement(&question, Some(&answer)), RawJudgement::Missing);
        assert_eq!(
            judgement(&question, Some(&DecisionAnswer::Noul { noul: 0.5 })),
            RawJudgement::Missing
        );
    }
}
