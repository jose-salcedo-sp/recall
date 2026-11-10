//! System One: decision questions over a text, answered with calibrated probabilities.
//!
//! The wire format is TypeSafe's `POST /v1/systemone`, served locally by Kev (github.com/jaredpalmer/kev) and hosted
//! by system-one.dev. Workers AI serves the same request and answers for its Clef models behind its own route and
//! envelope; see [`super::workers_ai`].

use std::{collections::BTreeMap, time::Duration};

use async_trait::async_trait;
use eros::ensure;
use reqwest::Url;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};

use crate::domain::{NotAProbability, Probability};

use super::{ApiKey, ModelRef};

/// The decision timeout of [`HttpSystemOne`].
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// One `POST /v1/systemone` call: several questions over one text.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionRequest {
    /// The text every question is asked about.
    pub state: String,
    /// Overwritten with the configured model name when sent.
    pub model: String,
    /// Keyed by an id the model never sees.
    pub questions: BTreeMap<String, Decision>,
}

/// One question, and the kind of answer it takes.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Decision {
    /// Yes or no.
    Noul {
        /// The question, phrased for a yes/no answer.
        instructions: String,
    },
    /// One of the options, each with an optional description.
    Choice {
        /// The question.
        instructions: String,
        /// The options.
        criteria: Options,
    },
    /// Levels ordered from lowest to highest.
    Score {
        /// The question.
        instructions: String,
        /// Level names, lowest first.
        criteria: Vec<String>,
    },
}

/// Choice options in the order given: option order can change an answer, so it stays deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct Options(pub Vec<(String, Option<String>)>);

impl Serialize for Options {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (option, description) in &self.0 {
            map.serialize_entry(option, description)?;
        }
        map.end()
    }
}

/// The wire format, unchecked. Steps read probabilities through `yes_no` and `choice`, which reject any outside
/// `0..=1`, so one bad answer declines its step instead of failing the whole response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    /// Keyed by the request's question ids.
    pub answers: BTreeMap<String, DecisionAnswer>,
}

/// The answer to one [`Decision`], of the same kind. Probabilities are unchecked here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    /// Answer to [`Decision::Noul`].
    Noul {
        /// Probability of yes.
        noul: f32,
    },
    /// Answer to [`Decision::Choice`].
    Choice {
        /// The most probable option.
        choice: String,
        /// The model's confidence in `choice`.
        confidence: f32,
        /// Probability per option.
        probabilities: BTreeMap<String, f32>,
    },
    /// Answer to [`Decision::Score`].
    Score {
        /// Mean level index, from 0.
        score: f32,
        /// The model's confidence in the score.
        confidence: f32,
        /// Probability per level.
        probabilities: BTreeMap<String, f32>,
    },
}

/// An answer that is not the kind asked for, or whose probability is not one.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum InvalidAnswer {
    /// The answer is missing or of another kind.
    #[error("expected a {expected} answer for {id:?}")]
    Unexpected {
        /// Question id.
        id: String,
        /// The kind asked for.
        expected: &'static str,
    },
    /// The chosen option has no entry in `probabilities`.
    #[error("no probability for {choice:?} in the answer for {id:?}")]
    NoProbability {
        /// Question id.
        id: String,
        /// The chosen option.
        choice: String,
    },
    /// The probability is outside `0..=1` or `NaN`.
    #[error("answer for {id:?}: {source}")]
    OutOfRange {
        /// Question id.
        id: String,
        /// The rejected value.

        #[source]
        source: NotAProbability,
    },
}

impl DecisionResponse {
    /// The probability of yes for the yes/no question `id`.
    ///
    /// # Errors
    ///
    /// [`InvalidAnswer::Unexpected`] or [`InvalidAnswer::OutOfRange`].
    pub fn yes_no(&self, id: &str) -> Result<Probability, InvalidAnswer> {
        match self.answers.get(id) {
            Some(DecisionAnswer::Noul { noul }) => probability(id, *noul),
            _ => Err(InvalidAnswer::Unexpected {
                id: id.to_owned(),
                expected: "yes/no",
            }),
        }
    }

    /// The option chosen for the choice question `id`, and the probability given to it.
    ///
    /// # Errors
    ///
    /// Any [`InvalidAnswer`].
    pub fn choice(&self, id: &str) -> Result<(&str, Probability), InvalidAnswer> {
        let Some(DecisionAnswer::Choice {
            choice,
            probabilities,
            ..
        }) = self.answers.get(id)
        else {
            return Err(InvalidAnswer::Unexpected {
                id: id.to_owned(),
                expected: "choice",
            });
        };
        let p = probabilities
            .get(choice)
            .ok_or_else(|| InvalidAnswer::NoProbability {
                id: id.to_owned(),
                choice: choice.clone(),
            })?;
        Ok((choice, probability(id, *p)?))
    }
}

fn probability(id: &str, value: f32) -> Result<Probability, InvalidAnswer> {
    Probability::new(value).map_err(|source| InvalidAnswer::OutOfRange {
        id: id.to_owned(),
        source,
    })
}

/// The "what" model: answers [`Decision`]s with calibrated probabilities. It only picks among options it is
/// given, so it cannot invent one.
#[async_trait]
pub trait SystemOne: Send + Sync {
    /// Recorded in the trace.
    fn model(&self) -> ModelRef;

    /// Answers every question in `request`.
    ///
    /// # Errors
    ///
    /// When the server fails, times out, or returns a body that does not parse. Out-of-range probabilities do not
    /// fail here; see [`DecisionResponse`].
    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse>;
}

/// The decision route under the server root.
pub(crate) const DECIDE_PATH: &str = "v1/systemone";

/// [`SystemOne`] over `POST {base_url}/v1/systemone`, with a 120 s request timeout.
pub struct HttpSystemOne {
    client: reqwest::Client,
    url: String,
    model: String,
    key: Option<ApiKey>,
}

impl HttpSystemOne {
    /// `base_url` is the server root, e.g. `http://127.0.0.1:8009` or `https://system-one.dev`.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(
        base_url: &Url,
        model: impl Into<String>,
        key: Option<ApiKey>,
    ) -> eros::Result<Self> {
        Ok(Self {
            client: client()?,
            url: format!("{}/{DECIDE_PATH}", base_url.as_str().trim_end_matches('/')),
            model: model.into(),
            key,
        })
    }
}

#[eros::context("building the System One HTTP client")]
fn client() -> eros::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()?)
}

#[async_trait]
impl SystemOne for HttpSystemOne {
    fn model(&self) -> ModelRef {
        ModelRef::new(&self.model, "system-one-v1")
    }

    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        self.post(request).await
    }
}

impl HttpSystemOne {
    #[eros::context("POST {}", self.url)]
    async fn post(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        let request = DecisionRequest {
            model: self.model.clone(),
            ..request
        };
        let mut post = self.client.post(&self.url).json(&request);
        if let Some(key) = &self.key {
            post = post.bearer_auth(key.expose());
        }
        let response = post.send().await?;
        let status = response.status();
        let text = response.text().await?;
        ensure!(status.is_success(), "returned {status}: {text}");
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(any(test, feature = "testing"))]
pub use super::fakes::ScriptedSystemOne;

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    use super::*;

    fn request() -> DecisionRequest {
        DecisionRequest {
            state: "Does AGR_USERS store role assignments?".into(),
            model: String::new(),
            questions: BTreeMap::from([(
                "category".into(),
                Decision::Choice {
                    instructions: "What kind of answer?".into(),
                    criteria: Options(vec![
                        ("open".into(), Some("Free-form".into())),
                        ("boolean".into(), None),
                    ]),
                },
            )]),
        }
    }

    #[tokio::test]
    async fn posts_the_typesafe_wire_format_and_parses_answers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(header("authorization", "Bearer secret"))
            .and(body_json(json!({
                "state": "Does AGR_USERS store role assignments?",
                "model": "kev-latest",
                "questions": { "category": {
                    "type": "choice",
                    "instructions": "What kind of answer?",
                    "criteria": { "open": "Free-form", "boolean": null }
                } }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "kev-latest",
                "answers": { "category": {
                    "type": "choice", "choice": "boolean", "confidence": 0.8,
                    "probabilities": { "boolean": 0.9, "open": 0.1 }
                } },
                "usage": { "input_tokens": 40, "output_tokens": 20 },
                "latency_ms": 120
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = HttpSystemOne::new(
            &format!("{}/", server.uri()).parse().unwrap(),
            "kev-latest",
            Some("secret".parse().unwrap()),
        )
        .unwrap();
        let response = client.decide(request()).await.unwrap();
        assert!(matches!(
            &response.answers["category"],
            DecisionAnswer::Choice { choice, .. } if choice == "boolean"
        ));
    }

    fn response(answer: DecisionAnswer) -> DecisionResponse {
        DecisionResponse {
            answers: BTreeMap::from([("q".into(), answer)]),
        }
    }

    #[test]
    fn yes_no_rejects_probabilities_outside_zero_to_one_and_nan() {
        for bad in [1.7, -0.3, f32::NAN] {
            assert!(matches!(
                response(DecisionAnswer::Noul { noul: bad }).yes_no("q"),
                Err(InvalidAnswer::OutOfRange { .. })
            ));
        }
        let ok = response(DecisionAnswer::Noul { noul: 0.25 });
        assert_eq!(ok.yes_no("q").unwrap().get(), 0.25);
        assert!(matches!(
            ok.yes_no("other"),
            Err(InvalidAnswer::Unexpected { .. })
        ));
        assert!(matches!(
            ok.choice("q"),
            Err(InvalidAnswer::Unexpected { .. })
        ));
    }

    #[test]
    fn choice_checks_the_chosen_probability() {
        let choice = |p: f32| {
            response(DecisionAnswer::Choice {
                choice: "open".into(),
                confidence: 0.5,
                probabilities: BTreeMap::from([("open".into(), p)]),
            })
        };
        assert_eq!(choice(0.8).choice("q").unwrap().0, "open");
        assert!(matches!(
            choice(1.3).choice("q"),
            Err(InvalidAnswer::OutOfRange { .. })
        ));
        assert!(matches!(
            choice(f32::NAN).choice("q"),
            Err(InvalidAnswer::OutOfRange { .. })
        ));
        let missing = response(DecisionAnswer::Choice {
            choice: "boolean".into(),
            confidence: 0.5,
            probabilities: BTreeMap::new(),
        });
        assert!(matches!(
            missing.choice("q"),
            Err(InvalidAnswer::NoProbability { .. })
        ));
    }

    #[tokio::test]
    async fn an_out_of_range_answer_still_arrives_so_its_step_can_decline() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "answers": { "q": { "type": "noul", "noul": 1.7 } }
            })))
            .mount(&server)
            .await;
        let client =
            HttpSystemOne::new(&server.uri().parse().unwrap(), "kev-latest", None).unwrap();

        let response = client.decide(request()).await.unwrap();

        assert!(matches!(
            response.yes_no("q"),
            Err(InvalidAnswer::OutOfRange { .. })
        ));
    }

    #[test]
    fn options_keep_their_order() {
        let options = Options(vec![("z".into(), None), ("a".into(), Some("first".into()))]);
        assert_eq!(
            serde_json::to_string(&options).unwrap(),
            r#"{"z":null,"a":"first"}"#
        );
    }
}
