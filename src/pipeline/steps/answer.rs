use std::{collections::BTreeMap, fmt::Write as _, marker::PhantomData, sync::Arc};

use serde_json::json;

use crate::{
    domain::{Probability, Question},
    llm::system_one::{Decision, DecisionRequest, SystemOne},
    pipeline::{Boolean, DeclineReason, Scored, Step, StepError, trace::AnswerRecord},
};

use super::{MAX_STATE_CHARS, Reranked};

/// Recorded as the trace entry's prompt version; bump on any change to the question System One is asked.
pub const ANSWER_PROMPT_VERSION: &str = "answer-boolean-v1";

const QUESTION_ID: &str = "answer";

/// System One answers the question over the reranked sources: the "what". `Explain` then writes the "why".
pub struct Answer<TCategory> {
    system_one: Option<Arc<dyn SystemOne>>,
    threshold: Option<Probability>,
    category: PhantomData<TCategory>,
}

impl<TCategory> Answer<TCategory> {
    /// Without a System One the step fails every run. `threshold` applies to the verdict's confidence.
    pub fn new(system_one: Option<Arc<dyn SystemOne>>, threshold: Option<Probability>) -> Self {
        Self {
            system_one,
            threshold,
            category: PhantomData,
        }
    }
}

/// Output of `Answer<Boolean>`: the verdict, how sure System One is of it, and the sources it read.
pub struct Decided {
    pub(super) yes: bool,
    /// `max(p, 1 - p)` for System One's probability of yes `p`.
    pub(super) confidence: Probability,
    pub(super) sources: Reranked,
}

/// The sources in reranked order, each under its title and heading, cut to System One's budget.
fn state(sources: &Reranked) -> String {
    let mut state = String::new();
    for (n, hit) in sources.0.iter().enumerate() {
        let chunk = &hit.chunk;
        let heading = chunk
            .heading
            .as_deref()
            .map(|heading| format!(" > {heading}"))
            .unwrap_or_default();
        let _ = write!(
            state,
            "Source {}: {}{heading}\n\n{}\n\n",
            n + 1,
            chunk.title,
            chunk.text
        );
        if state.chars().count() >= MAX_STATE_CHARS {
            break;
        }
    }
    match state.char_indices().nth(MAX_STATE_CHARS) {
        Some((end, _)) => state[..end].to_owned(),
        None => state.trim_end().to_owned(),
    }
}

fn request(question: &Question, sources: &Reranked) -> DecisionRequest {
    DecisionRequest {
        state: state(sources),
        model: String::new(),
        questions: BTreeMap::from([(
            QUESTION_ID.to_owned(),
            Decision::Noul {
                instructions: format!(
                    "Going only by these sources, is the answer to this question yes: {}",
                    question.as_str()
                ),
            },
        )]),
    }
}

impl Step for Answer<Boolean> {
    const NAME: &'static str = "answer";
    type Category = Boolean;
    type Input = Reranked;
    type Output = Decided;
    type Record = AnswerRecord;

    fn threshold(&self) -> Option<Probability> {
        self.threshold
    }

    async fn run(
        &self,
        question: &Question,
        sources: Reranked,
    ) -> Result<Scored<Decided, AnswerRecord>, StepError> {
        let Some(system_one) = &self.system_one else {
            return Err(eros::error!("the answer step needs a System One server").into());
        };
        if sources.0.is_empty() {
            return Err(StepError::decline(DeclineReason::NoSources));
        }
        let model = system_one.model();
        let response = system_one
            .decide(request(question, &sources))
            .await
            .map_err(|error| {
                StepError::from(error).with_model(model.clone(), ANSWER_PROMPT_VERSION)
            })?;
        let raw = response
            .answers
            .get(QUESTION_ID)
            .and_then(|answer| serde_json::to_string(answer).ok());
        let p = match response.yes_no(QUESTION_ID) {
            Ok(p) => p,
            Err(error) => {
                return Err(StepError::decline(DeclineReason::InvalidModelOutput(
                    error.to_string(),
                ))
                .with_record(AnswerRecord { raw, parsed: None })
                .with_model(model, ANSWER_PROMPT_VERSION));
            }
        };
        let yes = p.get() >= 0.5;
        let confidence = if yes { p } else { p.complement() };
        let record = AnswerRecord {
            raw,
            parsed: Some(json!({ "yes": yes, "probability_yes": p })),
        };
        Ok(Scored::new(
            Decided {
                yes,
                confidence,
                sources,
            },
            record,
        )
        .with_score(confidence)
        .with_model(model, ANSWER_PROMPT_VERSION))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        llm::system_one::{DecisionAnswer, DecisionResponse, ScriptedSystemOne},
        pipeline::{StepErrorKind, steps::fixtures::chunk},
        retrieval::{Relevance, Retrieved},
    };

    use super::*;

    fn noul(p: f32) -> DecisionResponse {
        DecisionResponse {
            answers: BTreeMap::from([(QUESTION_ID.into(), DecisionAnswer::Noul { noul: p })]),
        }
    }

    fn sources() -> Reranked {
        Reranked(vec![Retrieved {
            chunk: chunk(
                "users-only",
                "Mitigating controls",
                "Controls map to users only.",
            ),
            score: Relevance(0.9),
            context_for: None,
        }])
    }

    fn p(value: f32) -> Probability {
        Probability::new(value).unwrap()
    }

    async fn run(p: f32) -> Result<Scored<Decided, AnswerRecord>, StepError> {
        let system_one = Arc::new(ScriptedSystemOne::new([Ok(noul(p))]));
        Answer::<Boolean>::new(Some(system_one), None)
            .run(&Question::new("Can a control map to a role?"), sources())
            .await
    }

    async fn answer(p: f32) -> (Scored<Decided, AnswerRecord>, Arc<ScriptedSystemOne>) {
        let system_one = Arc::new(ScriptedSystemOne::new([Ok(noul(p))]));
        let scored =
            Answer::<Boolean>::new(Some(Arc::<ScriptedSystemOne>::clone(&system_one)), None)
                .run(&Question::new("Can a control map to a role?"), sources())
                .await
                .unwrap();
        (scored, system_one)
    }

    #[tokio::test]
    async fn a_probability_outside_zero_to_one_declines_with_the_raw_answer() {
        for bad in [1.7, -0.3, f32::NAN] {
            let error = run(bad).await.err().unwrap();
            assert!(
                matches!(
                    error.kind,
                    StepErrorKind::Decline(DeclineReason::InvalidModelOutput(_))
                ),
                "{bad}"
            );
            assert!(error.record.is_some(), "{bad}");
        }
    }

    #[tokio::test]
    async fn the_probability_of_yes_becomes_the_verdict_and_its_confidence() {
        let (scored, system_one) = answer(0.1).await;
        assert!(!scored.output.yes);
        assert_eq!(scored.score, Some(p(0.9)));
        assert_eq!(scored.output.confidence, p(0.9));
        assert_eq!(scored.prompt_version, Some(ANSWER_PROMPT_VERSION));

        let request = &system_one.requests()[0];
        assert!(
            request
                .state
                .starts_with("Source 1: Mitigating controls > users-only")
        );
        let Decision::Noul { instructions } = &request.questions[QUESTION_ID] else {
            panic!("expected a yes/no question");
        };
        assert!(instructions.ends_with("Can a control map to a role?"));

        let (scored, _) = answer(0.8).await;
        assert!(scored.output.yes);
        assert_eq!(scored.score, Some(p(0.8)));
    }

    #[test]
    fn long_sources_are_cut_to_the_budget() {
        let long = Reranked(
            (0..3)
                .map(|i| Retrieved {
                    chunk: chunk(&format!("s{i}"), "Big", &"x".repeat(MAX_STATE_CHARS)),
                    score: Relevance(1.0),
                    context_for: None,
                })
                .collect(),
        );
        assert_eq!(state(&long).chars().count(), MAX_STATE_CHARS);
    }
}
