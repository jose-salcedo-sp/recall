//! Replies are model output, so the contract is the transport: the request reaches the model as given, and a text
//! or structured reply comes back as text.

use std::sync::Arc;

use drew::llm::{ChatModel, ChatRequest};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use wiremock::MockServer;

/// What the model is set up to answer.
#[derive(Clone)]
pub enum Reply {
    Text(String),
    /// A structured answer to a request with a `json_schema`.
    Json(Value),
}

/// An adapter set up to give one reply, and every request it sent, read back from the wire.
pub struct Adapter {
    pub chat: Arc<dyn ChatModel>,
    pub sent: Box<dyn Fn() -> BoxFuture<'static, Vec<ChatRequest>> + Send + Sync>,
}

fn request(json_schema: Option<Value>) -> ChatRequest {
    ChatRequest {
        system: "Answer from the sources.".into(),
        user: "<question>Which table maps users to roles?</question>".into(),
        max_tokens: 512,
        temperature: 0.25,
        json_schema,
    }
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": { "claims": { "type": "array", "items": { "type": "string" } } },
        "required": ["claims"],
    })
}

pub async fn a_text_reply_comes_back_as_its_text(
    make: impl AsyncFnOnce(Reply) -> Adapter,
) -> eros::Result<()> {
    let text = "AGR_USERS maps users to roles.";
    let adapter = make(Reply::Text(text.into())).await;

    let response = adapter.chat.complete(request(None)).await?;

    assert_eq!(response.text, text);
    Ok(())
}

pub async fn a_structured_reply_comes_back_as_json_text(
    make: impl AsyncFnOnce(Reply) -> Adapter,
) -> eros::Result<()> {
    let answer = json!({ "claims": ["AGR_USERS maps users to roles."] });
    let adapter = make(Reply::Json(answer.clone())).await;

    let response = adapter.chat.complete(request(Some(schema()))).await?;

    assert_eq!(serde_json::from_str::<Value>(&response.text)?, answer);
    Ok(())
}

pub async fn the_request_reaches_the_model_as_given(
    make: impl AsyncFn(Reply) -> Adapter,
) -> eros::Result<()> {
    for json_schema in [None, Some(schema())] {
        let reply = match json_schema {
            Some(_) => Reply::Json(json!({ "claims": [] })),
            None => Reply::Text("ok".into()),
        };
        let adapter = make(reply).await;

        adapter.chat.complete(request(json_schema.clone())).await?;

        assert_eq!((adapter.sent)().await, [request(json_schema)]);
    }
    Ok(())
}

/// The start of a structured reply, as a model cut off at `max_tokens` leaves it.
const CUT_OFF: &str = r#"{"claims": ["AGR_USERS maps users to"#;

/// Adapter-only: a fake has no token limit. `chat` replies with [`CUT_OFF`] and its backend's stopped-at-the-limit
/// reason; the error names the budget and the cause, not the partial JSON a parser would reject.
async fn a_reply_cut_off_at_max_tokens_is_an_error(chat: &dyn ChatModel, stopped: &str) {
    let error = chat.complete(request(Some(schema()))).await.unwrap_err();

    let report = format!("{error:#?}");
    assert!(
        report.contains(&format!("answer cut off at max_tokens=512 ({stopped})"))
            && report.contains("on a reasoning model max_tokens also bounds the reasoning"),
        "{report}"
    );
}

macro_rules! chat_model_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn a_text_reply_comes_back_as_its_text() -> eros::Result<()> {
            crate::chat_model::a_text_reply_comes_back_as_its_text($make).await
        }

        #[tokio::test]
        async fn a_structured_reply_comes_back_as_json_text() -> eros::Result<()> {
            crate::chat_model::a_structured_reply_comes_back_as_json_text($make).await
        }

        #[tokio::test]
        async fn the_request_reaches_the_model_as_given() -> eros::Result<()> {
            crate::chat_model::the_request_reaches_the_model_as_given($make).await
        }
    };
}

/// Reads back every request `server` received, parsed by `parse`.
fn sent_to(
    server: MockServer,
    parse: fn(&Value) -> ChatRequest,
) -> Box<dyn Fn() -> BoxFuture<'static, Vec<ChatRequest>> + Send + Sync> {
    let server = Arc::new(server);
    Box::new(move || {
        let server = Arc::clone(&server);
        Box::pin(async move {
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .map(|request| parse(&request.body_json().unwrap()))
                .collect()
        })
    })
}

fn as_f32(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

mod scripted {
    use std::sync::Arc;

    use drew::llm::fakes::ScriptedChat;

    use super::{Adapter, Reply};

    async fn scripted(reply: Reply) -> Adapter {
        let text = match reply {
            Reply::Text(text) => text,
            Reply::Json(value) => value.to_string(),
        };
        let fake = Arc::new(ScriptedChat::replying([text]));
        Adapter {
            chat: Arc::<ScriptedChat>::clone(&fake),
            sent: Box::new(move || {
                let fake = Arc::clone(&fake);
                Box::pin(async move { fake.requests() })
            }),
        }
    }

    chat_model_contract!(scripted);
}

mod openai {
    use std::sync::Arc;

    use drew::llm::{
        ChatModel, ChatRequest,
        openai::{ChatOptions, OpenAiChat, ReasoningEffort, SchemaFormat},
    };
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, CUT_OFF, Reply, as_f32, request, schema, sent_to};

    /// Workers AI's `@cf/qwen/qwen3.8-27b` as configured in `.env.example`, but with the bare schema form.
    const WORKERS_AI: ChatOptions = ChatOptions {
        reasoning_effort: Some(ReasoningEffort::Medium),
        schema_format: SchemaFormat::Bare,
    };

    fn sent(body: &Value) -> ChatRequest {
        ChatRequest {
            system: body["messages"][0]["content"].as_str().unwrap().into(),
            user: body["messages"][1]["content"].as_str().unwrap().into(),
            max_tokens: body["max_tokens"].as_u64().unwrap().try_into().unwrap(),
            temperature: as_f32(&body["temperature"]),
            json_schema: body.get("response_format").map(|format| {
                let json_schema = &format["json_schema"];
                match json_schema.get("schema") {
                    Some(schema) if json_schema.get("name").is_some() => schema.clone(),
                    _ => json_schema.clone(),
                }
            }),
        }
    }

    async fn server(message: Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "choices": [{ "message": message }] })),
            )
            .mount(&server)
            .await;
        server
    }

    fn content(reply: Reply) -> String {
        match reply {
            Reply::Text(text) => text,
            Reply::Json(value) => value.to_string(),
        }
    }

    fn chat(server: &MockServer, key: Option<&str>, options: ChatOptions) -> OpenAiChat {
        OpenAiChat::new(
            &format!("{}/v1", server.uri()).parse().unwrap(),
            "qwen3:14b",
        )
        .unwrap()
        .with_key(key.map(|key| key.parse().unwrap()))
        .with_options(options)
    }

    async fn openai(reply: Reply) -> Adapter {
        let server = server(json!({ "role": "assistant", "content": content(reply) })).await;
        Adapter {
            chat: Arc::new(chat(&server, None, ChatOptions::default())),
            sent: sent_to(server, sent),
        }
    }

    chat_model_contract!(openai);

    /// A reasoning model's partial answer, with `length` as the completion's finish reason.
    fn cut_off() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({ "choices": [{
            "message": { "role": "assistant", "reasoning": "The user wants JSON.", "content": CUT_OFF },
            "finish_reason": "length",
        }] }))
    }

    #[tokio::test]
    async fn a_reply_cut_off_at_max_tokens_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(cut_off())
            .mount(&server)
            .await;

        crate::chat_model::a_reply_cut_off_at_max_tokens_is_an_error(
            &chat(&server, None, WORKERS_AI),
            "finish_reason length",
        )
        .await;
    }

    mod workers_ai {
        use drew::llm::workers_ai::WorkersAiChat;
        use wiremock::Respond;

        use super::*;

        const QWEN: &str = "@cf/qwen/qwen3.8-27b";
        const ROUTE: &str = "/client/v4/accounts/acc-123/ai/v1/chat/completions";
        /// `.env.example`'s Explain settings: `medium` effort, OpenAI's schema form.
        const ENV_EXAMPLE: ChatOptions = ChatOptions {
            reasoning_effort: Some(ReasoningEffort::Medium),
            schema_format: SchemaFormat::Openai,
        };

        async fn responding(response: impl Respond + 'static) -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(ROUTE))
                .respond_with(response)
                .mount(&server)
                .await;
            server
        }

        fn qwen(server: &MockServer, key: &str, options: ChatOptions) -> WorkersAiChat {
            WorkersAiChat::new(
                "acc-123".parse().unwrap(),
                QWEN.parse().unwrap(),
                key.parse().unwrap(),
                options,
            )
            .unwrap()
            .with_api_base(&format!("{}/client/v4/", server.uri()).parse().unwrap())
        }

        /// A reasoning model that returns its reasoning in a separate field and also inlines a `<think>` block.
        async fn workers_ai(reply: Reply) -> Adapter {
            let message = json!({
                "role": "assistant",
                "reasoning": "The user wants JSON.",
                "reasoning_content": "The user wants JSON.",
                "content": format!("<think>draft</think>{}", content(reply)),
            });
            let server = responding(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "choices": [{ "message": message }] })),
            )
            .await;
            Adapter {
                chat: Arc::new(qwen(&server, "cf-token", ENV_EXAMPLE)),
                sent: sent_to(server, sent),
            }
        }

        chat_model_contract!(workers_ai);

        #[tokio::test]
        async fn a_reply_cut_off_at_max_tokens_is_an_error() {
            let server = responding(cut_off()).await;

            crate::chat_model::a_reply_cut_off_at_max_tokens_is_an_error(
                &qwen(&server, "cf-token", ENV_EXAMPLE),
                "finish_reason length",
            )
            .await;
        }

        #[tokio::test]
        async fn the_settings_reach_the_accounts_route_with_the_token() {
            let server = responding(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "choices": [{ "message": { "content": "{}" } }] })),
            )
            .await;
            let chat = qwen(&server, "cf-token", WORKERS_AI);

            chat.complete(request(Some(schema()))).await.unwrap();

            let sent = &server.received_requests().await.unwrap()[0];
            assert_eq!(sent.url.path(), ROUTE);
            assert_eq!(
                sent.headers.get("authorization").unwrap(),
                "Bearer cf-token"
            );
            let body: Value = sent.body_json().unwrap();
            assert_eq!(body["model"], QWEN);
            assert_eq!(body["reasoning_effort"], "medium");
            assert_eq!(body["response_format"]["json_schema"], schema());
            assert_eq!(
                chat.model(),
                drew::llm::ModelRef::new(
                    QWEN,
                    "openai-compatible;reasoning_effort=medium;schema=bare"
                )
            );
        }

        #[tokio::test]
        async fn an_error_envelope_names_the_model_and_its_errors_but_not_the_token() {
            for status in [400, 200] {
                let server = responding(ResponseTemplate::new(status).set_body_json(json!({
                    "result": null,
                    "success": false,
                    "errors": [{ "code": 5007, "message": "No such model @cf/qwen/qwen3.8-27b" }],
                    "messages": [],
                })))
                .await;

                let error = qwen(&server, "cf-secret-token", ENV_EXAMPLE)
                    .complete(request(None))
                    .await
                    .unwrap_err();

                let report = format!("{error:?}");
                assert!(
                    report.contains(&format!("{QWEN} failed with {status}"))
                        && report.contains("5007: No such model"),
                    "{report}"
                );
                assert!(!report.contains("cf-secret-token"), "{report}");
            }
        }
    }

    async fn first_body(server: &MockServer) -> (Option<String>, Value) {
        let requests = server.received_requests().await.unwrap();
        let authorization = requests[0]
            .headers
            .get("authorization")
            .map(|value| value.to_str().unwrap().to_owned());
        (authorization, requests[0].body_json().unwrap())
    }

    #[tokio::test]
    async fn the_key_is_sent_as_a_bearer_token_only_when_set() {
        for key in [Some("cf-token"), None] {
            let server = server(json!({ "content": "ok" })).await;

            chat(&server, key, ChatOptions::default())
                .complete(request(None))
                .await
                .unwrap();

            let (authorization, _) = first_body(&server).await;
            assert_eq!(authorization, key.map(|key| format!("Bearer {key}")));
        }
    }

    #[tokio::test]
    async fn reasoning_effort_is_sent_only_when_set() {
        for effort in [Some(ReasoningEffort::Medium), None] {
            let server = server(json!({ "content": "ok" })).await;
            let options = ChatOptions {
                reasoning_effort: effort,
                ..ChatOptions::default()
            };

            chat(&server, None, options)
                .complete(request(None))
                .await
                .unwrap();

            let (_, body) = first_body(&server).await;
            assert_eq!(
                body.get("reasoning_effort"),
                effort.map(|effort| json!(effort.as_str())).as_ref()
            );
        }
    }

    #[tokio::test]
    async fn each_schema_format_has_its_wire_shape() {
        for (format, expected) in [
            (
                SchemaFormat::Openai,
                json!({ "name": "respond", "strict": true, "schema": schema() }),
            ),
            (SchemaFormat::Bare, schema()),
        ] {
            let server = server(json!({ "content": "{\"claims\": []}" })).await;
            let options = ChatOptions {
                schema_format: format,
                ..ChatOptions::default()
            };

            chat(&server, None, options)
                .complete(request(Some(schema())))
                .await
                .unwrap();

            let (_, body) = first_body(&server).await;
            assert_eq!(
                body["response_format"],
                json!({ "type": "json_schema", "json_schema": expected })
            );
        }
    }

    #[test]
    fn the_options_are_part_of_the_model_version() {
        let model = |options| {
            OpenAiChat::new(
                &"http://127.0.0.1:9/v1".parse().unwrap(),
                "@cf/qwen/qwen3.8-27b",
            )
            .unwrap()
            .with_options(options)
            .model()
        };
        assert_eq!(model(ChatOptions::default()).version, "openai-compatible");
        assert_eq!(
            model(WORKERS_AI).version,
            "openai-compatible;reasoning_effort=medium;schema=bare"
        );
    }

    #[tokio::test]
    async fn an_error_never_shows_the_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string("bad token"))
            .mount(&server)
            .await;

        let error = chat(&server, Some("cf-secret-token"), WORKERS_AI)
            .complete(request(None))
            .await
            .unwrap_err();

        let report = format!("{error:?}");
        assert!(report.contains("401"), "{report}");
        assert!(!report.contains("cf-secret-token"), "{report}");
    }
}

mod bedrock {
    use std::sync::Arc;

    use drew::llm::{ChatModel, ChatRequest, bedrock_chat::BedrockChat};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path_regex},
    };

    use super::{Adapter, Reply, as_f32, request, sent_to};

    const CLAUDE: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";

    fn sent(body: &Value) -> ChatRequest {
        ChatRequest {
            system: body["system"][0]["text"].as_str().unwrap().into(),
            user: body["messages"][0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .into(),
            max_tokens: body["inferenceConfig"]["maxTokens"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap(),
            temperature: as_f32(&body["inferenceConfig"]["temperature"]),
            json_schema: body
                .get("toolConfig")
                .map(|tools| tools["tools"][0]["toolSpec"]["inputSchema"]["json"].clone()),
        }
    }

    fn converse(reply: Reply) -> Value {
        let (content, stop) = match reply {
            Reply::Text(text) => (json!({ "text": text }), "end_turn"),
            Reply::Json(input) => (
                json!({ "toolUse": { "toolUseId": "tooluse_1", "name": "respond", "input": input } }),
                "tool_use",
            ),
        };
        json!({
            "output": { "message": { "role": "assistant", "content": [content] } },
            "stopReason": stop,
            "usage": { "inputTokens": 40, "outputTokens": 20, "totalTokens": 60 },
            "metrics": { "latencyMs": 120 },
        })
    }

    async fn server(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex("^/model/[^/]+/converse$"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    async fn bedrock(reply: Reply) -> Adapter {
        let server = server(ResponseTemplate::new(200).set_body_json(converse(reply))).await;
        Adapter {
            chat: Arc::new(BedrockChat::new(crate::bedrock::client(&server), CLAUDE)),
            sent: sent_to(server, sent),
        }
    }

    chat_model_contract!(bedrock);

    #[tokio::test]
    async fn a_reply_cut_off_at_max_tokens_is_an_error() {
        let mut reply = converse(Reply::Text(super::CUT_OFF.into()));
        reply["stopReason"] = json!("max_tokens");
        let server = server(ResponseTemplate::new(200).set_body_json(reply)).await;

        crate::chat_model::a_reply_cut_off_at_max_tokens_is_an_error(
            &BedrockChat::new(crate::bedrock::client(&server), CLAUDE),
            "stop reason max_tokens",
        )
        .await;
    }

    #[tokio::test]
    async fn a_rejected_call_names_the_model_and_the_reason() {
        let server = server(
            ResponseTemplate::new(400)
                .insert_header("x-amzn-errortype", "ValidationException")
                .set_body_json(json!({ "message": "max_tokens exceeds the model limit" })),
        )
        .await;

        let error = BedrockChat::new(crate::bedrock::client(&server), CLAUDE)
            .complete(request(None))
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains(CLAUDE) && report.contains("max_tokens exceeds the model limit"),
            "{report}"
        );
    }
}
