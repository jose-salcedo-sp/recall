use std::sync::Arc;

use crate::{
    domain::{AtLeastOne, CategoryKind, Probability, Question},
    llm::judge::{Judge, JudgeOption, JudgeQuestion, JudgeRequest},
    pipeline::{
        Ask, Asked, DeclineReason, Scored, Step, StepError, Unclassified,
        trace::{ClassifyMode, ClassifyRecord},
    },
};

/// Recorded as the trace entry's prompt version; bump on any change to the question or its options.
pub const CLASSIFY_PROMPT_VERSION: &str = "classify-v1";

/// In the order offered; each name is its category's [`CategoryKind::as_str`].
const OPTIONS: [(CategoryKind, JudgeOption); 4] = [
    (
        CategoryKind::Boolean,
        JudgeOption {
            name: "boolean",
            description: "A yes or no answer",
        },
    ),
    (
        CategoryKind::Choice,
        JudgeOption {
            name: "choice",
            description: "Picking one of two or more options named in the question",
        },
    ),
    (
        CategoryKind::Ranking,
        JudgeOption {
            name: "ranking",
            description: "Ordering several items named in the question by a stated criterion",
        },
    ),
    (
        CategoryKind::Open,
        JudgeOption {
            name: "open",
            description: "An explanation, a how-to, or any other free-form answer",
        },
    ),
];

/// Asks the judge which category the question is, or, when the pipeline has no judge, records every question as
/// `open`.
///
/// A decisive category chosen with a probability below `min_confidence` falls back to `open`: an explanation
/// cannot be wrong the way a forced yes or no can. The step's score stays the judge's probability.
pub struct Classify {
    mode: Classifier,
}

enum Classifier {
    AllOpen,
    Judged {
        judge: Arc<dyn Judge>,
        min_confidence: Option<Probability>,
    },
}

impl Classify {
    /// Every question is `open`, unscored.
    pub fn all_open() -> Self {
        Self {
            mode: Classifier::AllOpen,
        }
    }

    /// `judge` chooses the category. Below `min_confidence` a decisive category falls back to `open`; `None` never
    /// falls back.
    pub fn judged(judge: Arc<dyn Judge>, min_confidence: Option<Probability>) -> Self {
        Self {
            mode: Classifier::Judged {
                judge,
                min_confidence,
            },
        }
    }

    /// How the step chooses the category.
    pub fn mode(&self) -> ClassifyMode {
        match self.mode {
            Classifier::AllOpen => ClassifyMode::AllOpen,
            Classifier::Judged { .. } => ClassifyMode::Judged,
        }
    }
}

fn request(question: &Question) -> JudgeRequest {
    JudgeRequest {
        text: question.as_str().to_owned(),
        questions: vec![JudgeQuestion::OneOf {
            instructions: "What kind of answer does this question ask for?".to_owned(),
            options: AtLeastOne::try_from(
                OPTIONS
                    .iter()
                    .map(|(_, option)| *option)
                    .collect::<Vec<_>>(),
            )
            .expect("OPTIONS lists every category"),
        }],
    }
}

impl Step for Classify {
    const NAME: &'static str = "classify";
    type Category = Unclassified;
    type Input = Asked;
    type Output = CategoryKind;
    type Record = ClassifyRecord;

    async fn run(
        &self,
        ask: &Ask,
        _input: Asked,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        let question = ask.question();
        let (judge, min_confidence) = match &self.mode {
            Classifier::AllOpen => {
                let category = CategoryKind::Open;
                return Ok(Scored::new(category, ClassifyRecord::AllOpen));
            }
            Classifier::Judged {
                judge,
                min_confidence,
            } => (judge, *min_confidence),
        };

        let model = judge.model();
        let judgements = judge.judge(request(question)).await.map_err(|error| {
            StepError::from(error).with_model(model.clone(), CLASSIFY_PROMPT_VERSION)
        })?;
        let raw = serde_json::to_string(judgements.raw(0))
            .expect("a judgement is plain data, which always serializes");
        let (chosen, score) = judgements.one_of(0).map_err(|error| {
            StepError::decline(DeclineReason::InvalidModelOutput(error.to_string()))
                .with_record(ClassifyRecord::Judged {
                    raw: raw.clone(),
                    category: None,
                    fallback_from: None,
                })
                .with_model(model.clone(), CLASSIFY_PROMPT_VERSION)
        })?;
        let chosen = OPTIONS[chosen].0;
        let unsure = min_confidence.is_some_and(|min| score < min);
        let (category, fallback_from) = if unsure && chosen != CategoryKind::Open {
            (CategoryKind::Open, Some(chosen))
        } else {
            (chosen, None)
        };
        let record = ClassifyRecord::Judged {
            raw,
            category: Some(category),
            fallback_from,
        };
        Ok(Scored::new(category, record)
            .with_score(score)
            .with_model(model, CLASSIFY_PROMPT_VERSION))
    }
}

#[cfg(test)]
mod tests {
    use crate::llm::judge::{FnJudge, RawJudgement};

    use super::*;

    fn p(value: f32) -> Probability {
        Probability::new(value).unwrap()
    }

    /// Gives `category` probability `probability` and splits the rest evenly over the other options.
    fn choosing(category: &'static str, probability: f32) -> Arc<FnJudge> {
        Arc::new(FnJudge::new(move |_, question| {
            let JudgeQuestion::OneOf { options, .. } = question else {
                return RawJudgement::Missing;
            };
            let rest = (1.0 - probability) / (options.len() - 1) as f32;
            RawJudgement::OneOf(
                options
                    .iter()
                    .map(|option| {
                        if option.name == category {
                            probability
                        } else {
                            rest
                        }
                    })
                    .collect(),
            )
        }))
    }

    async fn classify(
        judge: &Arc<FnJudge>,
        min_confidence: Option<Probability>,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        Classify::judged(Arc::<FnJudge>::clone(judge), min_confidence)
            .run(
                &Ask::unrestricted("Does ROLE_USERS hold role assignments?"),
                Asked,
            )
            .await
    }

    #[test]
    fn option_names_are_the_category_names() {
        for (category, option) in OPTIONS {
            assert_eq!(option.name, category.as_str());
        }
    }

    #[tokio::test]
    async fn the_most_probable_option_and_its_probability_become_the_category_and_score() {
        let judge = choosing("boolean", 0.87);

        let scored = classify(&judge, None).await.unwrap();

        assert_eq!(scored.output, CategoryKind::Boolean);
        assert_eq!(scored.score, Some(p(0.87)));
        assert_eq!(scored.record.mode(), ClassifyMode::Judged);
        let ClassifyRecord::Judged { raw, .. } = &scored.record else {
            panic!("{:?}", scored.record);
        };
        assert!(raw.starts_with(r#"{"one_of":"#));
        assert_eq!(scored.prompt_version, Some(CLASSIFY_PROMPT_VERSION));
        let request = &judge.requests()[0];
        assert_eq!(request.text, "Does ROLE_USERS hold role assignments?");
        let JudgeQuestion::OneOf { options, .. } = &request.questions[0] else {
            panic!("expected a one-of question");
        };
        let names: Vec<_> = options.iter().map(|option| option.name).collect();
        assert_eq!(names, ["boolean", "choice", "ranking", "open"]);
    }

    #[tokio::test]
    async fn an_unsure_decisive_category_falls_back_to_open() {
        let unsure = |category, probability| async move {
            classify(&choosing(category, probability), Some(p(0.6)))
                .await
                .unwrap()
        };

        let scored = unsure("boolean", 0.55).await;
        assert_eq!(scored.output, CategoryKind::Open);
        assert_eq!(scored.score, Some(p(0.55)));
        assert!(matches!(
            scored.record,
            ClassifyRecord::Judged {
                fallback_from: Some(CategoryKind::Boolean),
                ..
            }
        ));

        let scored = unsure("boolean", 0.65).await;
        assert_eq!(scored.output, CategoryKind::Boolean);
        assert!(matches!(
            scored.record,
            ClassifyRecord::Judged {
                fallback_from: None,
                ..
            }
        ));

        let scored = unsure("open", 0.3).await;
        assert_eq!(scored.output, CategoryKind::Open);
        assert!(matches!(
            scored.record,
            ClassifyRecord::Judged {
                fallback_from: None,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn a_missing_answer_declines_with_the_raw_answer() {
        let judge = Arc::new(FnJudge::new(|_, _| RawJudgement::Missing));

        let error = classify(&judge, None).await.err().unwrap();

        assert!(matches!(
            error.kind,
            crate::pipeline::StepErrorKind::Decline(DeclineReason::InvalidModelOutput(_))
        ));
        assert!(error.record.is_some());
        assert_eq!(error.prompt_version, Some(CLASSIFY_PROMPT_VERSION));
    }

    #[tokio::test]
    async fn a_probability_outside_zero_to_one_declines_with_the_raw_answer() {
        for bad in [1.3, -0.2, f32::NAN] {
            let error = classify(&choosing("boolean", bad), None)
                .await
                .err()
                .unwrap();

            assert!(
                matches!(
                    error.kind,
                    crate::pipeline::StepErrorKind::Decline(DeclineReason::InvalidModelOutput(_))
                ),
                "{bad}"
            );
            assert!(error.record.is_some(), "{bad}");
        }
    }

    #[tokio::test]
    async fn all_open_records_every_question_as_open_and_unscored() {
        let step = Classify::all_open();
        assert_eq!(step.mode(), ClassifyMode::AllOpen);
        let scored = step
            .run(&Ask::unrestricted("anything"), Asked)
            .await
            .unwrap();
        assert_eq!(scored.output, CategoryKind::Open);
        assert_eq!(scored.score, None);
        assert_eq!(scored.record.mode(), ClassifyMode::AllOpen);
    }
}
