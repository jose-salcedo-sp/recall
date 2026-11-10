//! Ingest a page, then ask about it over HTTP, against a real Postgres and fake models.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use chrono::Utc;
use drew::{
    domain::{ChunkDraft, Document, Labels, Sensitivity, SourceKind},
    ingest::{Source, fakes::StaticSource, ingest},
    llm::fakes::{HashEmbedder, ScriptedChat},
    pipeline::answerer::PipelineAnswerer,
    retrieval::hybrid::{HybridConfig, HybridRetriever},
    server::{AppState, auth::AdminUsers, router},
    store::runs::PgRunStore,
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sqlx::PgPool;
use tower::ServiceExt;

fn roles_page() -> Document {
    Document {
        source: SourceKind::Markdown,
        external_id: "roles.md".into(),
        title: "Role tables".into(),
        url: "https://wiki.example.test/roles.md".into(),
        updated_at: Utc::now(),
        labels: Labels {
            acl: vec![],
            customer: None,
            sensitivity: Sensitivity::Internal,
        },
        chunks: vec![
            ChunkDraft {
                locator: "roles".into(),
                heading: Some("Role assignments".into()),
                url: "https://wiki.example.test/roles.md#roles".into(),
                text: "AGR_USERS stores which users are assigned to which roles.".into(),
                authoritative: true,
            },
            ChunkDraft {
                locator: "transport".into(),
                heading: Some("Transports".into()),
                url: "https://wiki.example.test/roles.md#transport".into(),
                text: "Roles move between systems in transports.".into(),
                authoritative: true,
            },
        ],
    }
}

/// Ingest, hybrid retrieval, the pipeline, the HTTP API, and the run store, with only the models faked.
#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn an_ingested_page_answers_a_question_over_http(pool: PgPool) -> eros::Result<()> {
    let sources: Vec<Box<dyn Source>> = vec![Box::new(StaticSource {
        kind: SourceKind::Markdown,
        documents: vec![roles_page()],
    })];
    ingest(&pool, &HashEmbedder, &sources, false).await?;

    let retriever = HybridRetriever::new(
        pool.clone(),
        Arc::new(HashEmbedder),
        HybridConfig::default(),
    );
    let explain = ScriptedChat::replying([json!({
        "claims": [{
            "text": "AGR_USERS holds the user-to-role assignments.",
            "chunk_ids": ["markdown:roles.md#roles"]
        }]
    })
    .to_string()]);
    let app = router(AppState {
        answerer: Arc::new(PipelineAnswerer::new(
            Arc::new(retriever),
            Arc::new(explain),
        )),
        runs: Arc::new(PgRunStore::new(pool)),
        keys: Arc::new("sk-jose:jose".parse().unwrap()),
        admins: Arc::new(AdminUsers::default()),
    });
    let send = |request: Request<Body>| {
        let app = app.clone();
        async move {
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, serde_json::from_slice::<Value>(&body).unwrap())
        }
    };

    let (status, completion) = send(
        Request::post("/v1/chat/completions")
            .header(header::AUTHORIZATION, "Bearer sk-jose")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "drew-ai",
                    "messages": [{ "role": "user", "content": "Which table has AGR_USERS role assignments?" }]
                })
                .to_string(),
            ))?,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{completion:#}");
    assert_eq!(completion["drew"]["outcome"], "answered");
    assert_eq!(completion["drew"]["category"], "open");
    assert_eq!(
        completion["drew"]["citations"][0]["chunk"]["id"],
        "markdown:roles.md#roles"
    );
    assert!(completion["drew"].get("trace").is_none());
    let content = completion["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(content.contains("[1]"), "{content}");

    let id = completion["drew"]["run_id"].as_str().unwrap();
    let (status, run) = send(
        Request::get(format!("/v1/runs/{id}"))
            .header(header::AUTHORIZATION, "Bearer sk-jose")
            .body(Body::empty())?,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(run["status"], "answered");
    let steps: Vec<_> = run["trace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["step"].as_str().unwrap())
        .collect();
    assert_eq!(
        steps,
        ["classify", "retrieve", "rerank", "explain", "verify"]
    );
    Ok(())
}
