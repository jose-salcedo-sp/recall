//! System One answers are model output, so the contract is the transport: what reaches the model, and that every
//! answer comes back under its question id.

use std::{collections::BTreeMap, sync::Arc};

use drew::llm::system_one::{
    Decision, DecisionAnswer, DecisionRequest, DecisionResponse, Options, SystemOne,
};
use futures::future::BoxFuture;
use serde_json::Value;

/// An adapter set up to give one reply, and the wire body of every request it sent.
pub struct Adapter {
    pub system_one: Arc<dyn SystemOne>,
    pub sent: Box<dyn Fn() -> BoxFuture<'static, Vec<Value>> + Send + Sync>,
}

fn request() -> DecisionRequest {
    DecisionRequest {
        state: "Does AGR_USERS store role assignments?".into(),
        model: String::new(),
        questions: BTreeMap::from([
            (
                "category".into(),
                Decision::Choice {
                    instructions: "What kind of answer?".into(),
                    criteria: Options(vec![
                        ("open".into(), Some("Free-form".into())),
                        ("boolean".into(), None),
                    ]),
                },
            ),
            (
                "relevant".into(),
                Decision::Noul {
                    instructions: "Does the text name a table?".into(),
                },
            ),
        ]),
    }
}

fn reply() -> DecisionResponse {
    DecisionResponse {
        answers: BTreeMap::from([
            (
                "category".into(),
                DecisionAnswer::Choice {
                    choice: "boolean".into(),
                    confidence: 0.75,
                    probabilities: BTreeMap::from([
                        ("boolean".into(), 0.875),
                        ("open".into(), 0.125),
                    ]),
                },
            ),
            ("relevant".into(), DecisionAnswer::Noul { noul: 0.5 }),
        ]),
    }
}

pub async fn answers_come_back_by_question_id(
    make: impl AsyncFnOnce(DecisionResponse) -> Adapter,
) -> eros::Result<()> {
    let adapter = make(reply()).await;

    let response = adapter.system_one.decide(request()).await?;

    assert_eq!(response, reply());
    Ok(())
}

pub async fn the_request_is_sent_under_the_adapters_model_name(
    make: impl AsyncFnOnce(DecisionResponse) -> Adapter,
) -> eros::Result<()> {
    let adapter = make(reply()).await;
    let request = DecisionRequest {
        model: "set by the caller".into(),
        ..request()
    };

    adapter.system_one.decide(request.clone()).await?;

    let expected = DecisionRequest {
        model: adapter.system_one.model().id,
        ..request
    };
    assert_eq!((adapter.sent)().await, [serde_json::to_value(&expected)?]);
    Ok(())
}

macro_rules! system_one_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn answers_come_back_by_question_id() -> eros::Result<()> {
            crate::system_one::answers_come_back_by_question_id($make).await
        }

        #[tokio::test]
        async fn the_request_is_sent_under_the_adapters_model_name() -> eros::Result<()> {
            crate::system_one::the_request_is_sent_under_the_adapters_model_name($make).await
        }
    };
}

mod scripted {
    use std::sync::Arc;

    use drew::llm::{fakes::ScriptedSystemOne, system_one::DecisionResponse};

    use super::Adapter;

    async fn scripted(reply: DecisionResponse) -> Adapter {
        let fake = Arc::new(ScriptedSystemOne::new([Ok(reply)]));
        Adapter {
            system_one: Arc::<ScriptedSystemOne>::clone(&fake),
            sent: Box::new(move || {
                let fake = Arc::clone(&fake);
                Box::pin(async move {
                    fake.requests()
                        .iter()
                        .map(|request| serde_json::to_value(request).unwrap())
                        .collect()
                })
            }),
        }
    }

    system_one_contract!(scripted);
}

mod http {
    use std::sync::Arc;

    use drew::llm::system_one::{DecisionResponse, HttpSystemOne, SystemOne};
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, request};

    fn client(server: &MockServer, key: Option<&str>) -> HttpSystemOne {
        HttpSystemOne::new(
            &format!("{}/", server.uri()).parse().unwrap(),
            "kev-latest",
            key.map(|key| key.parse().unwrap()),
        )
        .unwrap()
    }

    async fn responding(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    async fn kev(reply: DecisionResponse) -> Adapter {
        let mut body = serde_json::to_value(&reply).unwrap();
        body["model"] = json!("kev-latest");
        body["usage"] = json!({ "input_tokens": 40, "output_tokens": 20 });
        body["latency_ms"] = json!(120);
        let server = Arc::new(responding(ResponseTemplate::new(200).set_body_json(body)).await);
        Adapter {
            system_one: Arc::new(client(&server, None)),
            sent: Box::new(move || {
                let server = Arc::clone(&server);
                Box::pin(async move {
                    server
                        .received_requests()
                        .await
                        .unwrap()
                        .iter()
                        .map(|request| request.body_json().unwrap())
                        .collect()
                })
            }),
        }
    }

    system_one_contract!(kev);

    #[tokio::test]
    async fn an_error_status_carries_the_status_and_body() {
        let server =
            responding(ResponseTemplate::new(500).set_body_string("model not loaded")).await;

        let error = client(&server, None).decide(request()).await.unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains("500") && report.contains("model not loaded"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn a_body_that_is_not_json_is_an_error() {
        let server = responding(ResponseTemplate::new(200).set_body_string("<html>")).await;

        assert!(client(&server, None).decide(request()).await.is_err());
    }

    #[tokio::test]
    async fn the_key_is_sent_as_a_bearer_token_only_when_set() {
        for key in [Some("secret"), None] {
            let server =
                responding(ResponseTemplate::new(200).set_body_json(json!({ "answers": {} })))
                    .await;

            client(&server, key).decide(request()).await.unwrap();

            let sent = server.received_requests().await.unwrap();
            let authorization = sent[0]
                .headers
                .get("authorization")
                .map(|value| value.to_str().unwrap().to_owned());
            assert_eq!(authorization, key.map(|key| format!("Bearer {key}")));
        }
    }
}

mod workers_ai {
    use std::sync::Arc;

    use drew::llm::{
        system_one::{DecisionResponse, SystemOne},
        workers_ai::WorkersAiSystemOne,
    };
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, request};

    const CLEF_FLASH: &str = "@cf/cloudflare/clef-flash";
    const ROUTE: &str = "/client/v4/accounts/acc-123/ai/run/@cf/cloudflare/clef-flash";

    fn client(server: &MockServer, key: &str) -> WorkersAiSystemOne {
        WorkersAiSystemOne::new(
            "acc-123".parse().unwrap(),
            CLEF_FLASH.parse().unwrap(),
            key.parse().unwrap(),
        )
        .unwrap()
        .with_api_base(&format!("{}/client/v4/", server.uri()).parse().unwrap())
    }

    async fn responding(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(ROUTE))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    /// The System One body as the port sees it: the route names the model, and `model` carries only its selector.
    fn as_port_request(request: &Request) -> Value {
        let mut body: Value = request.body_json().unwrap();
        let route_model = request
            .url
            .path()
            .split("/ai/run/")
            .nth(1)
            .unwrap()
            .to_owned();
        assert_eq!(body["model"], json!("clef-flash"));
        body["model"] = json!(route_model);
        body
    }

    async fn clef_flash(reply: DecisionResponse) -> Adapter {
        let mut result = serde_json::to_value(&reply).unwrap();
        result["model"] = json!("clef-flash");
        result["usage"] = json!({ "input_tokens": 40, "output_tokens": 0 });
        let envelope = json!({ "result": result, "success": true, "errors": [], "messages": [] });
        let server = Arc::new(responding(ResponseTemplate::new(200).set_body_json(envelope)).await);
        Adapter {
            system_one: Arc::new(client(&server, "cf-token")),
            sent: Box::new(move || {
                let server = Arc::clone(&server);
                Box::pin(async move {
                    server
                        .received_requests()
                        .await
                        .unwrap()
                        .iter()
                        .map(as_port_request)
                        .collect()
                })
            }),
        }
    }

    system_one_contract!(clef_flash);

    #[tokio::test]
    async fn the_token_and_the_selector_are_sent_to_the_models_route() {
        let server = responding(ResponseTemplate::new(200).set_body_json(json!({
            "result": { "model": "clef-flash", "answers": {}, "usage": { "input_tokens": 1, "output_tokens": 0 } },
            "success": true, "errors": [], "messages": []
        })))
        .await;
        let client = client(&server, "cf-token");

        client.decide(request()).await.unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        assert_eq!(sent.url.path(), ROUTE);
        assert_eq!(
            sent.headers.get("authorization").unwrap(),
            "Bearer cf-token"
        );
        assert_eq!(sent.body_json::<Value>().unwrap()["model"], "clef-flash");
        assert_eq!(client.model().id, CLEF_FLASH);
    }

    #[tokio::test]
    async fn an_error_envelope_names_the_model_and_the_errors_but_not_the_token() {
        for status in [400, 200] {
            let server = responding(ResponseTemplate::new(status).set_body_json(json!({
                "result": null,
                "success": false,
                "errors": [{ "code": 5006, "message": "questions: must have at least 1 property" }],
                "messages": []
            })))
            .await;

            let error = client(&server, "cf-secret-token")
                .decide(request())
                .await
                .unwrap_err();

            let report = format!("{error:?}");
            assert!(
                report.contains(CLEF_FLASH)
                    && report.contains(&status.to_string())
                    && report.contains("5006: questions: must have at least 1 property"),
                "{report}"
            );
            assert!(!report.contains("cf-secret-token"), "{report}");
        }
    }

    #[tokio::test]
    async fn a_body_outside_the_envelope_is_an_error_naming_the_model() {
        let server =
            responding(ResponseTemplate::new(502).set_body_string("<html>bad gateway")).await;

        let error = client(&server, "cf-token")
            .decide(request())
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains(CLEF_FLASH) && report.contains("502") && report.contains("bad gateway"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn another_servers_json_404_names_its_host_and_quotes_200_bytes_of_its_body() {
        let detail = format!("Not Found {}", "x".repeat(300));
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({ "detail": detail })))
            .mount(&server)
            .await;

        let error = client(&server, "cf-token")
            .decide(request())
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        let host = server.address().to_string();
        let quoted = &format!(r#"{{"detail":"{detail}"}}"#)[..200];
        assert!(
            report.contains(&format!("at {host} returned 404 Not Found"))
                && report.contains(quoted)
                && !report.contains(&format!("{quoted}x"))
                && !report.contains("no error message"),
            "{report}"
        );
    }
}
