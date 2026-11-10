use std::{collections::BTreeMap, sync::Arc};

use crate::{
    domain::{CategoryKind, Probability, Question},
    llm::system_one::{Decision, DecisionRequest, DecisionResponse, Options, SystemOne},
    pipeline::{
        Asked, DeclineReason, Scored, Step, StepError, Unclassified, trace::ClassifyRecord,
    },
};

/// Recorded as the trace entry's prompt version; bump on any change to the question or its options.
pub const CLASSIFY_PROMPT_VERSION: &str = "classify-v1";

const QUESTION_ID: &str = "category";

const OPTIONS: [(CategoryKind, &str); 4] = [
    (CategoryKind::Boolean, "A yes or no answer"),
    (
        CategoryKind::Choice,
        "Picking one of two or more options named in the question",
    ),
    (
        CategoryKind::Ranking,
        "Ordering several items named in the question by a stated criterion",
    ),
    (
        CategoryKind::Open,
        "An explanation, a how-to, or any other free-form answer",
    ),
];

/// Asks System One which category the question is. Without a System One every question is `open`.
///
/// A decisive category chosen with a probability below `min_confidence` falls back to `open`: an explanation
/// cannot be wrong the way a forced yes or no can. The step's score stays System One's probability.
pub struct Classify {
    system_one: Option<Arc<dyn SystemOne>>,
    min_confidence: Option<Probability>,
}

impl Classify {
    /// `None` classifies every question as `open`, unscored.
    pub fn new(system_one: Option<Arc<dyn SystemOne>>) -> Self {
        Self {
            system_one,
            min_confidence: None,
        }
    }

    /// Sets the probability below which a decisive category falls back to `open`; `None` never falls back.
    pub fn with_min_confidence(self, min_confidence: Option<Probability>) -> Self {
        Self {
            min_confidence,
            ..self
        }
    }

    fn request(question: &Question) -> DecisionRequest {
        let options = OPTIONS
            .iter()
            .map(|(category, description)| {
                (
                    category.as_str().to_owned(),
                    Some((*description).to_owned()),
                )
            })
            .collect();
        DecisionRequest {
            state: question.as_str().to_owned(),
            model: String::new(),
            questions: BTreeMap::from([(
                QUESTION_ID.to_owned(),
                Decision::Choice {
                    instructions: "What kind of answer does this question ask for?".to_owned(),
                    criteria: Options(options),
                },
            )]),
        }
    }
}

/// The chosen category and the calibrated probability System One gave it.
fn parse(response: &DecisionResponse) -> Result<(CategoryKind, Probability), DeclineReason> {
    let invalid = |detail: String| DeclineReason::InvalidModelOutput(detail);
    let (choice, probability) = response
        .choice(QUESTION_ID)
        .map_err(|error| invalid(error.to_string()))?;
    let category = CategoryKind::try_from(choice).map_err(|error| invalid(error.to_string()))?;
    Ok((category, probability))
}

impl Step for Classify {
    const NAME: &'static str = "classify";
    type Category = Unclassified;
    type Input = Asked;
    type Output = CategoryKind;
    type Record = ClassifyRecord;

    async fn run(
        &self,
        question: &Question,
        _input: Asked,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        let Some(system_one) = &self.system_one else {
            let category = CategoryKind::Open;
            let record = ClassifyRecord {
                raw: None,
                category: Some(category),
                fallback_from: None,
            };
            return Ok(Scored::new(category, record));
        };

        let model = system_one.model();
        let response = system_one
            .decide(Self::request(question))
            .await
            .map_err(|error| {
                StepError::from(error).with_model(model.clone(), CLASSIFY_PROMPT_VERSION)
            })?;
        let answer = response.answers.get(QUESTION_ID);
        let raw = answer.and_then(|answer| serde_json::to_string(answer).ok());
        let (chosen, score) = parse(&response).map_err(|reason| {
            StepError::decline(reason)
                .with_record(ClassifyRecord {
                    raw: raw.clone(),
                    category: None,
                    fallback_from: None,
                })
                .with_model(model.clone(), CLASSIFY_PROMPT_VERSION)
        })?;
        let unsure = self.min_confidence.is_some_and(|min| score < min);
        let (category, fallback_from) = if unsure && chosen != CategoryKind::Open {
            (CategoryKind::Open, Some(chosen))
        } else {
            (chosen, None)
        };
        let record = ClassifyRecord {
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
    use crate::llm::system_one::{DecisionAnswer, ScriptedSystemOne};

    use super::*;

    fn p(value: f32) -> Probability {
        Probability::new(value).unwrap()
    }

    fn choice(choice: &str, probability: f32) -> DecisionResponse {
        DecisionResponse {
            answers: BTreeMap::from([(
                QUESTION_ID.to_owned(),
                DecisionAnswer::Choice {
                    choice: choice.into(),
                    confidence: 0.5,
                    probabilities: BTreeMap::from([(choice.into(), probability)]),
                },
            )]),
        }
    }

    async fn classify(
        system_one: &Arc<ScriptedSystemOne>,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        Classify::new(Some(Arc::<ScriptedSystemOne>::clone(system_one)))
            .run(
                &Question::new("Does AGR_USERS hold role assignments?"),
                Asked,
            )
            .await
    }

    #[tokio::test]
    async fn the_choice_and_its_probability_become_the_category_and_score() {
        let system_one = Arc::new(ScriptedSystemOne::new([Ok(choice("boolean", 0.87))]));

        let scored = classify(&system_one).await.unwrap();

        assert_eq!(scored.output, CategoryKind::Boolean);
        assert_eq!(scored.score, Some(p(0.87)));
        assert_eq!(scored.prompt_version, Some(CLASSIFY_PROMPT_VERSION));
        let request = &system_one.requests()[0];
        assert_eq!(request.state, "Does AGR_USERS hold role assignments?");
        let Decision::Choice { criteria, .. } = &request.questions[QUESTION_ID] else {
            panic!("expected a choice question");
        };
        let names: Vec<_> = criteria.0.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["boolean", "choice", "ranking", "open"]);
    }

    #[tokio::test]
    async fn an_unsure_decisive_category_falls_back_to_open() {
        let unsure = |category: &str, probability: f32| {
            let system_one = Arc::new(ScriptedSystemOne::new([Ok(choice(category, probability))]));
            async move {
                Classify::new(Some(system_one))
                    .with_min_confidence(Some(p(0.6)))
                    .run(&Question::new("q"), Asked)
                    .await
                    .unwrap()
            }
        };

        let scored = unsure("boolean", 0.55).await;
        assert_eq!(scored.output, CategoryKind::Open);
        assert_eq!(scored.score, Some(p(0.55)));
        assert_eq!(scored.record.fallback_from, Some(CategoryKind::Boolean));

        let scored = unsure("boolean", 0.65).await;
        assert_eq!(scored.output, CategoryKind::Boolean);
        assert_eq!(scored.record.fallback_from, None);

        let scored = unsure("open", 0.3).await;
        assert_eq!(scored.output, CategoryKind::Open);
        assert_eq!(scored.record.fallback_from, None);
    }

    #[tokio::test]
    async fn an_unknown_category_declines_with_the_raw_answer() {
        let system_one = Arc::new(ScriptedSystemOne::new([Ok(choice("haiku", 0.9))]));

        let error = classify(&system_one).await.err().unwrap();

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
            let system_one = Arc::new(ScriptedSystemOne::new([Ok(choice("boolean", bad))]));

            let error = classify(&system_one).await.err().unwrap();

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
    async fn without_system_one_every_question_is_open_and_unscored() {
        let scored = Classify::new(None)
            .run(&Question::new("anything"), Asked)
            .await
            .unwrap();
        assert_eq!(scored.output, CategoryKind::Open);
        assert_eq!(scored.score, None);
    }
}
