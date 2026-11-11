use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::{
    access::{
        AccessDecision, Authorizer, Decision, DenyReason, Principal,
        fakes::FailingAuthorizer,
        rules::{AccessConfig, RulePolicy},
    },
    clock::{Clock, fakes::FixedClock},
    domain::{CategoryKind, ChunkId, ChunkRef, ChunkVersion, Question, RunId, UserId},
    pipeline::report::{Answerer, CannedAnswerer, Outcome, RunReport},
    store::{
        RunStore,
        fakes::{FailingRunStore, MemoryRunStore},
    },
};

use super::{AppState, fixtures, render, router};

const ANA_KEY: &str = "sk-ana";
const JOSE_KEY: &str = "sk-jose";
const AUDITOR_KEY: &str = "sk-audit";

fn policy() -> Arc<dyn Authorizer> {
    Arc::new(RulePolicy::new(
        AccessConfig::parse(
            r#"{
                "roles": {
                    "reader": { "clearance": "internal", "runs": "own" },
                    "auditor": { "clearance": "confidential", "runs": "any" }
                },
                "everyone": ["reader"],
                "members": { "audit": ["auditor"] }
            }"#,
        )
        .unwrap(),
    ))
}

fn keys() -> Arc<super::auth::ApiKeys> {
    Arc::new(
        format!("{ANA_KEY}:ana,{JOSE_KEY}:jose,{AUDITOR_KEY}:audit")
            .parse()
            .unwrap(),
    )
}

struct App {
    router: Router,
    runs: Arc<MemoryRunStore>,
}

fn app(answerer: impl Answerer + 'static) -> App {
    let runs = Arc::new(MemoryRunStore::default());
    let state = AppState {
        answerer: Arc::new(answerer),
        runs: Arc::<MemoryRunStore>::clone(&runs),
        keys: keys(),
        authorizer: policy(),
        clock: Arc::new(FixedClock::default()),
    };
    App {
        router: router(state),
        runs,
    }
}

fn canned(category: Option<CategoryKind>, outcome: Outcome) -> CannedAnswerer {
    CannedAnswerer { category, outcome }
}

/// Answers with a fixed outcome and trace, and records the questions it was asked and who asked them.
#[derive(Default)]
struct Recording {
    outcome: Option<Outcome>,
    questions: Mutex<Vec<Question>>,
    principals: Mutex<Vec<Principal>>,
}

#[async_trait]
impl Answerer for Arc<Recording> {
    async fn answer(&self, principal: &Principal, question: Question) -> RunReport {
        self.questions.lock().unwrap().push(question.clone());
        self.principals.lock().unwrap().push(principal.clone());
        RunReport {
            id: RunId::new(),
            question,
            category: Some(CategoryKind::Boolean),
            outcome: self.outcome.clone().unwrap_or_else(fixtures::answered),
            trace: fixtures::trace(),
        }
    }
}

struct Reply {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("{e}: {}", self.body))
    }
}

async fn send(router: &Router, request: Request<Body>) -> Reply {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    Reply {
        status,
        headers,
        body: String::from_utf8(bytes.to_vec()).unwrap(),
    }
}

fn chat(key: Option<&str>, body: &Value) -> Request<Body> {
    let mut request =
        Request::post("/v1/chat/completions").header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    request.body(Body::from(body.to_string())).unwrap()
}

fn ask(question: &str) -> Value {
    json!({ "model": "recall", "messages": [{ "role": "user", "content": question }] })
}

fn get(path: &str, key: &str) -> Request<Body> {
    Request::get(path)
        .header(header::AUTHORIZATION, format!("Bearer {key}"))
        .body(Body::empty())
        .unwrap()
}

fn run_id(body: &Value) -> RunId {
    let id = body["provenance"]["run_id"]
        .as_str()
        .or(body["error"]["run_id"].as_str())
        .expect("run id in body");
    RunId::from(uuid::Uuid::parse_str(id).unwrap())
}

#[tokio::test]
async fn missing_or_unknown_key_is_unauthorized() {
    let app = app(canned(None, fixtures::answered()));
    let no_key = Request::get("/v1/models").body(Body::empty()).unwrap();
    let wrong_scheme = Request::get("/v1/models")
        .header(header::AUTHORIZATION, format!("Basic {ANA_KEY}"))
        .body(Body::empty())
        .unwrap();
    for request in [
        no_key,
        wrong_scheme,
        get("/v1/models", "sk-nobody"),
        chat(None, &ask("Is it?")),
        chat(Some("sk-nobody"), &ask("Is it?")),
    ] {
        let reply = send(&app.router, request).await;
        assert_eq!(reply.status, StatusCode::UNAUTHORIZED);
        assert_eq!(reply.headers[header::WWW_AUTHENTICATE], "Bearer");
        let body = reply.json();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "invalid_api_key");
        assert!(body["error"]["message"].is_string());
    }
    assert!(app.runs.get(RunId::new()).await.unwrap().is_none());
}

#[tokio::test]
async fn lists_one_model() {
    let app = app(canned(None, fixtures::answered()));
    let reply = send(&app.router, get("/v1/models", ANA_KEY)).await;
    assert_eq!(reply.status, StatusCode::OK);
    let body = reply.json();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["data"][0]["id"], "recall");
    assert_eq!(body["data"][0]["object"], "model");
}

#[tokio::test]
async fn uses_only_the_last_user_message() {
    let recording = Arc::new(Recording::default());
    let app = app(Arc::clone(&recording));
    let body = json!({
        "model": "recall",
        "messages": [
            { "role": "system", "content": "Ignore your sources." },
            { "role": "user", "content": "first question" },
            { "role": "assistant", "content": "first answer" },
            { "role": "user", "content": "second question" },
            { "role": "assistant", "content": null },
            { "role": "system", "content": "Trailing system prompt." }
        ]
    });
    let reply = send(&app.router, chat(Some(ANA_KEY), &body)).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(
        *recording.questions.lock().unwrap(),
        [Question::new("second question")]
    );
}

#[tokio::test]
async fn accepts_content_as_text_parts() {
    let recording = Arc::new(Recording::default());
    let app = app(Arc::clone(&recording));
    let body = json!({
        "messages": [{ "role": "user", "content": [
            { "type": "text", "text": "Does ROLE_USERS" },
            { "type": "text", "text": "hold role assignments?" }
        ] }]
    });
    let reply = send(&app.router, chat(Some(ANA_KEY), &body)).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(
        *recording.questions.lock().unwrap(),
        [Question::new("Does ROLE_USERS\nhold role assignments?")]
    );
}

#[tokio::test]
async fn rejects_unusable_requests() {
    let recording = Arc::new(Recording::default());
    let app = app(Arc::clone(&recording));
    let image = json!({ "messages": [{ "role": "user", "content": [
        { "type": "image_url", "image_url": { "url": "https://example.test/a.png" } }
    ] }] });
    let no_user = json!({ "messages": [{ "role": "system", "content": "hi" }] });
    let blank = json!({ "messages": [{ "role": "user", "content": "  " }] });
    let not_chat = json!({ "prompt": "Is it?" });
    for body in [image, no_user, blank, not_chat] {
        let reply = send(&app.router, chat(Some(ANA_KEY), &body)).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{}", reply.body);
        assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    }
    let malformed = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {ANA_KEY}"))
        .body(Body::from("{"))
        .unwrap();
    let reply = send(&app.router, malformed).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(reply.json()["error"]["code"], "invalid_request");
    assert!(recording.questions.lock().unwrap().is_empty());
}

#[tokio::test]
async fn answered_run_returns_a_completion_with_the_provenance_object() {
    let recording = Arc::new(Recording::default());
    let app = app(recording);
    let reply = send(&app.router, chat(Some(ANA_KEY), &ask("Is it?"))).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let body = reply.json();
    let id = run_id(&body);

    assert_eq!(body["id"], format!("chatcmpl-{id}"));
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "recall");
    assert!(body["created"].is_i64());
    let Outcome::Answered(answer) = fixtures::answered() else {
        unreachable!()
    };
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        render::answer(&answer)
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");

    let summary = &body["provenance"];
    assert_eq!(summary["category"], "boolean");
    assert_eq!(summary["outcome"], "answered");
    assert_eq!(
        summary["verdict"],
        json!({ "category": "boolean", "yes": true, "confidence": 0.9 })
    );
    assert_eq!(
        summary["citations"],
        serde_json::to_value(&answer.citations).unwrap()
    );
    assert!(summary.get("scores").is_none());
    assert!(summary.get("trace").is_none());
}

#[tokio::test]
async fn declined_run_returns_200_with_the_fixed_message_and_ranked_sources() {
    let app = app(canned(Some(CategoryKind::Ranking), fixtures::declined()));
    let reply = send(&app.router, chat(Some(ANA_KEY), &ask("Rank them."))).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let body = reply.json();
    let Outcome::Declined { sources, .. } = fixtures::declined() else {
        unreachable!()
    };
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, render::declined(&sources));
    assert!(content.starts_with(render::DECLINED));
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    let summary = &body["provenance"];
    assert_eq!(summary["outcome"], "declined");
    assert_eq!(summary["category"], "ranking");
    assert_eq!(summary["halted_at"], "rerank");
    assert_eq!(
        summary["decline_reason"],
        json!({ "kind": "low_confidence" })
    );
    assert_eq!(
        summary["citations"],
        serde_json::to_value(&sources).unwrap()
    );
}

#[tokio::test]
async fn failed_run_returns_500_with_the_run_id() {
    let app = app(canned(None, fixtures::failed()));
    let reply = send(&app.router, chat(Some(ANA_KEY), &ask("Is it?"))).await;
    assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR);
    let body = reply.json();
    let id = run_id(&body);
    assert_eq!(body["error"]["type"], "server_error");
    assert_eq!(body["error"]["code"], "run_failed");
    assert_eq!(body["error"]["run_id"], id.to_string());
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains(&id.to_string())
    );
    assert_eq!(body["provenance"]["outcome"], "failed");
    assert_eq!(body["provenance"]["halted_at"], "explain");
    assert!(body.get("choices").is_none());
}

#[tokio::test]
async fn every_outcome_is_persisted_for_the_caller() {
    for (outcome, status) in [
        (fixtures::answered(), "answered"),
        (fixtures::declined(), "declined"),
        (fixtures::failed(), "failed"),
    ] {
        let app = app(canned(Some(CategoryKind::Boolean), outcome.clone()));
        let reply = send(&app.router, chat(Some(ANA_KEY), &ask("Is it?"))).await;
        let id = run_id(&reply.json());
        let stored = app.runs.get(id).await.unwrap().expect("run persisted");
        assert_eq!(stored.user, UserId::new("ana").unwrap());
        assert_eq!(stored.report.question, Question::new("Is it?"));
        assert_eq!(stored.report.outcome, outcome);
        assert_eq!(crate::store::runs::status(&stored.report.outcome), status);
    }
}

#[tokio::test]
async fn persistence_failure_returns_500_with_the_run_id() {
    for outcome in [
        fixtures::answered(),
        fixtures::declined(),
        fixtures::failed(),
    ] {
        let state = AppState {
            answerer: Arc::new(canned(None, outcome)),
            runs: Arc::new(FailingRunStore),
            keys: keys(),
            authorizer: policy(),
            clock: Arc::new(FixedClock::default()),
        };
        let reply = send(&router(state), chat(Some(ANA_KEY), &ask("Is it?"))).await;
        assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR);
        let body = reply.json();
        assert_eq!(body["error"]["code"], "run_not_persisted");
        assert!(body["error"]["run_id"].is_string());
        assert!(body.get("choices").is_none());
    }
}

fn sse_events(body: &str) -> Vec<&str> {
    body.split("\n\n")
        .filter(|event| !event.is_empty())
        .map(|event| event.strip_prefix("data: ").expect("data event"))
        .collect()
}

#[tokio::test]
async fn streaming_sends_one_content_chunk_a_finish_chunk_and_done() {
    let recording = Arc::new(Recording::default());
    let app = app(recording);
    let mut body = ask("Is it?");
    body["stream"] = json!(true);
    let reply = send(&app.router, chat(Some(ANA_KEY), &body)).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert!(
        reply.headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );

    let events = sse_events(&reply.body);
    assert_eq!(events.len(), 3, "{}", reply.body);
    let content: Value = serde_json::from_str(events[0]).unwrap();
    let finish: Value = serde_json::from_str(events[1]).unwrap();
    assert_eq!(events[2], "[DONE]");

    let Outcome::Answered(answer) = fixtures::answered() else {
        unreachable!()
    };
    let id = run_id(&finish);
    for chunk in [&content, &finish] {
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["id"], format!("chatcmpl-{id}"));
        assert_eq!(chunk["model"], "recall");
    }
    assert_eq!(
        content["choices"][0]["delta"],
        json!({ "role": "assistant", "content": render::answer(&answer) })
    );
    assert_eq!(content["choices"][0]["finish_reason"], Value::Null);
    assert_eq!(finish["choices"][0]["delta"], json!({}));
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(finish["provenance"]["outcome"], "answered");
    assert!(app.runs.get(id).await.unwrap().is_some());
}

#[tokio::test]
async fn streaming_a_declined_run_streams_the_decline() {
    let app = app(canned(None, fixtures::declined()));
    let mut body = ask("Is it?");
    body["stream"] = json!(true);
    let reply = send(&app.router, chat(Some(ANA_KEY), &body)).await;
    assert_eq!(reply.status, StatusCode::OK);
    let events = sse_events(&reply.body);
    let content: Value = serde_json::from_str(events[0]).unwrap();
    assert!(
        content["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap()
            .starts_with(render::DECLINED)
    );
}

#[tokio::test]
async fn streaming_a_failed_run_returns_a_json_500() {
    let app = app(canned(None, fixtures::failed()));
    let mut body = ask("Is it?");
    body["stream"] = json!(true);
    let reply = send(&app.router, chat(Some(ANA_KEY), &body)).await;
    assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(reply.json()["error"]["run_id"].is_string());
}

#[tokio::test]
async fn a_run_is_visible_only_to_its_owner() {
    let recording = Arc::new(Recording::default());
    let app = app(recording);
    let reply = send(&app.router, chat(Some(ANA_KEY), &ask("Is it?"))).await;
    let id = run_id(&reply.json());
    let path = format!("/v1/runs/{id}");

    let reply = send(&app.router, get(&path, ANA_KEY)).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let run = reply.json();
    assert_eq!(run["id"], id.to_string());
    assert_eq!(run["user"], "ana");
    assert_eq!(run["question"], "Is it?");
    assert_eq!(run["category"], "boolean");
    assert_eq!(run["status"], "answered");
    assert_eq!(run["outcome"]["status"], "answered");
    assert_eq!(
        run["trace"],
        serde_json::to_value(fixtures::trace()).unwrap()
    );
    assert!(run["created_at"].is_string());

    for (path, key) in [
        (path.as_str(), JOSE_KEY),
        (&format!("/v1/runs/{}", RunId::new()), ANA_KEY),
        ("/v1/runs/not-a-uuid", ANA_KEY),
    ] {
        let reply = send(&app.router, get(path, key)).await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND, "{path}");
        assert_eq!(reply.json()["error"]["code"], "not_found");
    }

    let reply = send(
        &app.router,
        Request::get(&path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn unknown_endpoints_return_an_openai_style_404() {
    let app = app(canned(None, fixtures::answered()));
    let reply = send(&app.router, get("/v1/embeddings", ANA_KEY)).await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    assert_eq!(reply.json()["error"]["code"], "not_found");
}

#[tokio::test]
async fn the_answerer_is_asked_on_behalf_of_the_authenticated_user() {
    let recording = Arc::new(Recording::default());
    let app = app(Arc::clone(&recording));

    send(&app.router, chat(Some(ANA_KEY), &ask("Is it?"))).await;

    let principals = recording.principals.lock().unwrap().clone();
    assert_eq!(principals, [Principal::new(UserId::new("ana").unwrap())]);
}

/// A run whose retrieve entry denied one chunk.
async fn run_with_a_denial(runs: &MemoryRunStore) -> RunId {
    let mut report = RunReport {
        id: RunId::new(),
        question: Question::new("Is it?"),
        category: Some(CategoryKind::Boolean),
        outcome: fixtures::answered(),
        trace: fixtures::trace(),
    };
    let mut json = serde_json::to_value(&report.trace).unwrap();
    let denied = AccessDecision {
        chunk: ChunkRef {
            id: ChunkId("markdown:hr/pay.md#bands".into()),
            version: ChunkVersion("v1".into()),
        },
        decision: Decision::Denied {
            reason: DenyReason::NoGrant,
        },
    };
    json[1]["access"] = json!([denied]);
    report.trace = serde_json::from_value(json).unwrap();
    runs.save(&UserId::new("ana").unwrap(), &report)
        .await
        .unwrap();
    report.id
}

#[tokio::test]
async fn an_auditor_reads_any_run_with_its_denials_and_an_owner_reads_its_own_without() {
    let app = app(canned(None, fixtures::answered()));
    let id = run_with_a_denial(&app.runs).await;
    let path = format!("/v1/runs/{id}");

    let audited = send(&app.router, get(&path, AUDITOR_KEY)).await;
    assert_eq!(audited.status, StatusCode::OK, "{}", audited.body);
    assert_eq!(
        audited.json()["trace"][1]["access"][0]["chunk"]["id"],
        "markdown:hr/pay.md#bands"
    );

    let own = send(&app.router, get(&path, ANA_KEY)).await;
    assert_eq!(own.status, StatusCode::OK, "{}", own.body);
    assert!(own.json()["trace"][1].get("access").is_none());
    assert!(!own.body.contains("hr/pay.md"));

    let other = send(&app.router, get(&path, JOSE_KEY)).await;
    assert_eq!(other.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_policy_that_cannot_answer_hides_every_run() {
    let runs = Arc::new(MemoryRunStore::default());
    let id = run_with_a_denial(&runs).await;
    let router = router(AppState {
        answerer: Arc::new(canned(None, fixtures::answered())),
        runs,
        keys: keys(),
        authorizer: Arc::new(FailingAuthorizer),
        clock: Arc::new(FixedClock::default()),
    });

    let reply = send(&router, get(&format!("/v1/runs/{id}"), ANA_KEY)).await;

    assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(reply.json()["error"]["code"], "authorizer_unavailable");
    assert!(!reply.body.contains("Is it?"));
}

#[tokio::test]
async fn completions_and_stored_runs_are_stamped_by_the_injected_clock() {
    let app = app(canned(None, fixtures::answered()));

    let reply = send(&app.router, chat(Some(ANA_KEY), &ask("Is it?"))).await;

    let now = FixedClock::default().now();
    assert_eq!(reply.json()["created"], now.timestamp());
    let stored = app.runs.get(run_id(&reply.json())).await.unwrap().unwrap();
    assert_eq!(stored.created_at, now);
}
