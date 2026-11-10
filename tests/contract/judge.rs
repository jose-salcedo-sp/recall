//! Judge answers are model output, so the contract is the mapping: every answer comes back in question order, a
//! one-of answer has one probability per option in option order, and a missing answer is read as invalid instead
//! of failing the call.

use std::sync::Arc;

use drew::llm::judge::{InvalidJudgement, Judge, JudgeOption, JudgeQuestion, JudgeRequest};

/// What the adapter under test is set up to answer: the yes/no question's probability, and the one-of question's
/// probability per option of [`request`], or nothing for it.
#[derive(Clone, Copy)]
pub struct Answers {
    pub yes: f32,
    pub one_of: Option<[f32; 2]>,
}

const OPTIONS: [JudgeOption; 2] = [
    JudgeOption {
        name: "supports",
        description: "The text supports the claim.",
    },
    JudgeOption {
        name: "says_nothing",
        description: "The text is silent on the claim.",
    },
];

fn request() -> JudgeRequest {
    JudgeRequest {
        text: "AGR_USERS maps users to roles.".into(),
        questions: vec![
            JudgeQuestion::YesNo {
                instructions: "Does this text name a table?".into(),
            },
            JudgeQuestion::OneOf {
                instructions: "How does this text relate to the claim: AGR_USERS stores roles"
                    .into(),
                options: OPTIONS.to_vec(),
            },
        ],
    }
}

pub async fn answers_come_back_in_question_and_option_order(
    make: impl AsyncFnOnce(Answers) -> Arc<dyn Judge>,
) -> eros::Result<()> {
    let judge = make(Answers {
        yes: 0.25,
        one_of: Some([0.2, 0.8]),
    })
    .await;

    let judgements = judge.judge(request()).await?;

    assert_eq!(judgements.yes(0)?.get(), 0.25);
    let (chosen, probability) = judgements.one_of(1)?;
    assert_eq!((chosen, probability.get()), (1, 0.8));
    Ok(())
}

pub async fn a_missing_answer_is_invalid_not_a_failed_call(
    make: impl AsyncFnOnce(Answers) -> Arc<dyn Judge>,
) -> eros::Result<()> {
    let judge = make(Answers {
        yes: 0.5,
        one_of: None,
    })
    .await;

    let judgements = judge.judge(request()).await?;

    assert_eq!(judgements.yes(0)?.get(), 0.5);
    assert!(matches!(
        judgements.one_of(1),
        Err(InvalidJudgement::Unexpected { index: 1, .. })
    ));
    Ok(())
}

macro_rules! judge_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn answers_come_back_in_question_and_option_order() -> eros::Result<()> {
            crate::judge::answers_come_back_in_question_and_option_order($make).await
        }

        #[tokio::test]
        async fn a_missing_answer_is_invalid_not_a_failed_call() -> eros::Result<()> {
            crate::judge::a_missing_answer_is_invalid_not_a_failed_call($make).await
        }
    };
}

mod fake {
    use std::sync::Arc;

    use drew::llm::judge::{FnJudge, Judge, JudgeQuestion, RawJudgement};

    use super::Answers;

    async fn fake(answers: Answers) -> Arc<dyn Judge> {
        Arc::new(FnJudge::new(move |_, question| match question {
            JudgeQuestion::YesNo { .. } => RawJudgement::YesNo(answers.yes),
            JudgeQuestion::OneOf { .. } => answers
                .one_of
                .map_or(RawJudgement::Missing, |p| RawJudgement::OneOf(p.to_vec())),
        }))
    }

    judge_contract!(fake);
}

mod system_one {
    use std::sync::Arc;

    use async_trait::async_trait;
    use drew::llm::{
        ModelRef,
        judge::{Judge, JudgeRequest, Judgements, SystemOneJudge},
        system_one::HttpSystemOne,
    };
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, method, path},
    };

    use super::{Answers, OPTIONS};

    /// Kev behind [`SystemOneJudge`]: it only answers a request that keeps the text as the state and the options in
    /// order with their descriptions.
    async fn kev(answers: Answers) -> Arc<dyn Judge> {
        let mut body = json!({
            "model": "kev-latest",
            "answers": { "q0": { "type": "noul", "noul": answers.yes } },
        });
        if let Some([supports, says_nothing]) = answers.one_of {
            body["answers"]["q1"] = json!({
                "type": "choice",
                "choice": if supports > says_nothing { "supports" } else { "says_nothing" },
                "confidence": 0.4,
                "probabilities": { "supports": supports, "says_nothing": says_nothing },
            });
        }
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(body_partial_json(json!({
                "state": "AGR_USERS maps users to roles.",
                "questions": {
                    "q0": { "type": "noul" },
                    "q1": {
                        "type": "choice",
                        "criteria": {
                            OPTIONS[0].name: OPTIONS[0].description,
                            OPTIONS[1].name: OPTIONS[1].description,
                        },
                    },
                },
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        let system_one =
            HttpSystemOne::new(&server.uri().parse().unwrap(), "kev-latest", None).unwrap();
        Arc::new(Served {
            judge: SystemOneJudge::new(Arc::new(system_one)),
            _server: server,
        })
    }

    /// Keeps the mock server up, and its expectations checked on drop, for as long as the judge is used.
    struct Served {
        judge: SystemOneJudge,
        _server: MockServer,
    }

    #[async_trait]
    impl Judge for Served {
        fn model(&self) -> ModelRef {
            self.judge.model()
        }

        async fn judge(&self, request: JudgeRequest) -> eros::Result<Judgements> {
            self.judge.judge(request).await
        }
    }

    judge_contract!(kev);
}
