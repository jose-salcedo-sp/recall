//! `drew models check` against wiremock backends: Workers AI's OpenAI-compatible routes for embeddings and Explain,
//! and Kev's `/v1/systemone` for System One, since `workers-ai` takes no URL. One call per port, each reported with
//! its model, endpoint and outcome, and the key never printed.

use drew::{
    config::ConfigArgs,
    llm::{
        EMBEDDING_DIM,
        backend::{
            ChatArgs, ChatBackendKind, EmbedArgs, EmbedBackendKind, SystemOneArgs,
            SystemOneBackendKind,
        },
        openai::ReasoningEffort,
    },
    models::{Outcome, Port, check},
    pipeline::answerer::PipelineArgs,
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

const TOKEN: &str = "cf-secret-token";
const ACCOUNT: &str = "/client/v4/accounts/acc-123/ai";

async fn servers(system_one: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("{ACCOUNT}/v1/embeddings")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "index": 0, "embedding": vec![0.5; EMBEDDING_DIM] }]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{ACCOUNT}/v1/chat/completions")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": "{\"ok\": true}", "reasoning": "a check" } }]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(system_one)
        .mount(&server)
        .await;
    server
}

fn kev_answer() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "answers": { "check": { "type": "noul", "noul": 0.97 } }
    }))
}

fn args(server: &MockServer, system_one: SystemOneBackendKind) -> PipelineArgs {
    let openai_base = Some(format!("{}{ACCOUNT}/v1", server.uri()).parse().unwrap());
    let kev = system_one == SystemOneBackendKind::SystemOne;
    PipelineArgs {
        config: ConfigArgs::default(),
        embed: EmbedArgs {
            embed_backend: EmbedBackendKind::OpenaiCompatible,
            embed_model: "@cf/baai/bge-m3".into(),
            embed_url: openai_base.clone(),
            embed_account_id: None,
            embed_key: Some(TOKEN.parse().unwrap()),
            embed_max_batch_tokens: None,
        },
        chat: ChatArgs {
            explain_backend: ChatBackendKind::OpenaiCompatible,
            explain_model: "@cf/qwen/qwen3.8-27b".into(),
            explain_url: openai_base,
            explain_account_id: None,
            explain_key: Some(TOKEN.parse().unwrap()),
            explain_reasoning_effort: Some(ReasoningEffort::Medium),
            explain_schema_format: None,
        },
        system_one: SystemOneArgs {
            system_one_backend: system_one,
            system_one_model: kev.then(|| "kev-latest".into()),
            system_one_url: kev.then(|| server.uri().parse().unwrap()),
            system_one_account_id: None,
            system_one_key: kev.then(|| TOKEN.parse().unwrap()),
        },
    }
}

#[tokio::test]
async fn every_backend_answers_once_and_is_reported_ok_with_its_endpoint() {
    let server = servers(kev_answer()).await;

    let check = check(&args(&server, SystemOneBackendKind::SystemOne)).await;

    assert_eq!(check.failed(), 0, "{check}");
    let ports: Vec<_> = check.probes.iter().map(|probe| probe.port).collect();
    assert_eq!(ports, [Port::Embeddings, Port::Explain, Port::SystemOne]);
    let report = check.to_string();
    let host = server.uri();
    for line in [
        format!("@cf/baai/bge-m3 (dimensions=1024)  {host}{ACCOUNT}/v1/embeddings  "),
        format!(
            "@cf/qwen/qwen3.8-27b (openai-compatible;reasoning_effort=medium)  {host}{ACCOUNT}/v1/chat/completions  "
        ),
        format!("kev-latest (system-one-v1)  {host}/v1/systemone  "),
    ] {
        assert!(report.contains(&line), "{report}");
    }
    assert!(!report.contains(TOKEN), "{report}");
    assert!(!report.contains("Bedrock"), "{report}");
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn a_failing_backend_is_reported_with_its_error_and_the_others_still_pass() {
    let server = servers(
        ResponseTemplate::new(401).set_body_json(json!({ "detail": "Authentication error" })),
    )
    .await;

    let check = check(&args(&server, SystemOneBackendKind::SystemOne)).await;

    assert_eq!(check.failed(), 1, "{check}");
    let failed = check.probes.iter().find(|probe| probe.failed()).unwrap();
    assert_eq!(failed.port, Port::SystemOne);
    let report = check.to_string();
    assert!(
        report.contains("system one  kev-latest (system-one-v1)")
            && report.contains("Authentication error")
            && report.contains("asking one yes/no question"),
        "{report}"
    );
    assert!(!report.contains(TOKEN), "{report}");
}

#[tokio::test]
async fn a_missing_setting_fails_its_port_by_name() {
    let server = servers(kev_answer()).await;
    let mut args = args(&server, SystemOneBackendKind::SystemOne);
    args.system_one.system_one_backend = SystemOneBackendKind::WorkersAi;
    args.system_one.system_one_model = Some("@cf/cloudflare/clef-flash".into());
    args.system_one.system_one_url = None;

    let check = check(&args).await;

    assert_eq!(check.failed(), 1, "{check}");
    assert!(matches!(
        check
            .probes
            .last()
            .map(|probe| (probe.port, &probe.outcome)),
        Some((Port::SystemOne, Outcome::NotBuilt(_)))
    ));
    let report = check.to_string();
    assert!(
        report.contains("system one  not built")
            && report.contains(
                "DREW_SYSTEM_ONE_ACCOUNT_ID is required with DREW_SYSTEM_ONE_BACKEND=workers-ai"
            ),
        "{report}"
    );
}

#[tokio::test]
async fn a_url_left_over_from_kev_fails_workers_ai_before_any_call() {
    let server = servers(kev_answer()).await;
    let mut args = args(&server, SystemOneBackendKind::SystemOne);
    args.system_one.system_one_backend = SystemOneBackendKind::WorkersAi;
    args.system_one.system_one_model = Some("@cf/cloudflare/clef-flash".into());
    args.system_one.system_one_account_id = Some("acc-123".parse().unwrap());

    let check = check(&args).await;

    assert_eq!(check.failed(), 1, "{check}");
    let report = check.to_string();
    assert!(
        report.contains("system one  not built")
            && report.contains(
                "DREW_SYSTEM_ONE_URL is set, but DREW_SYSTEM_ONE_BACKEND=workers-ai does not use it; unset it"
            ),
        "{report}"
    );
    let systemone_calls = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.url.path() == "/v1/systemone")
        .count();
    assert_eq!(systemone_calls, 0);
}

#[tokio::test]
async fn no_system_one_is_a_note_not_a_failure() {
    let server = servers(kev_answer()).await;

    let check = check(&args(&server, SystemOneBackendKind::None)).await;

    assert_eq!(check.failed(), 0, "{check}");
    assert_eq!(check.probes.len(), 2);
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("DREW_SYSTEM_ONE_BACKEND=none")),
        "{check}"
    );
}
