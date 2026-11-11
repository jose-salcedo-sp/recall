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
use http_body_util::BodyExt;
use recall::{
    access::{
        Authorizer, ConnectorLabels,
        rules::{AccessConfig, RulePolicy},
    },
    clock::{SystemClock, UuidV7},
    domain::{ChunkDraft, Document, Labels, Sensitivity},
    ingest::{Sources, fakes::StaticSource, ingest, markdown},
    llm::{
        ChatModel,
        fakes::{HashEmbedder, ScriptedChat},
    },
    pipeline::{answerer::PipelineAnswerer, assembly::Ports},
    retrieval::hybrid::{HybridConfig, HybridRetriever},
    server::{AppState, router},
    store::{chunks::PgChunkStore, runs::PgRunStore},
};
use serde_json::{Value, json};
use sqlx::PgPool;
use tower::ServiceExt;

fn roles_page() -> Document {
    Document {
        source: markdown::NAME.clone(),
        external_id: "roles.md".into(),
        title: "Role tables".into(),
        url: "https://wiki.example.test/roles.md".into(),
        updated_at: Utc::now(),
        labels: Labels {
            acl: recall::access::Acl::anyone(),
            sensitivity: Sensitivity::Internal,
        },
        chunks: vec![
            ChunkDraft {
                locator: "roles".into(),
                heading: Some("Role assignments".into()),
                url: "https://wiki.example.test/roles.md#roles".into(),
                text: "ROLE_USERS stores which users are assigned to which roles.".into(),
                standing: recall::domain::Standing::Authoritative,
            },
            ChunkDraft {
                locator: "transport".into(),
                heading: Some("Transports".into()),
                url: "https://wiki.example.test/roles.md#transport".into(),
                text: "Roles move between systems in transports.".into(),
                standing: recall::domain::Standing::Authoritative,
            },
        ],
    }
}

fn ports(
    retriever: HybridRetriever,
    explain: Arc<dyn ChatModel>,
    authorizer: &Arc<dyn Authorizer>,
) -> Ports {
    Ports {
        retriever: Arc::new(retriever),
        embedding: None,
        explain,
        judge: None,
        authorizer: Arc::clone(authorizer),
        clock: Arc::new(SystemClock),
        ids: Arc::new(UuidV7),
    }
}

/// A policy whose baseline grant makes every user a `reader` of internal chunks and its own runs.
fn reader_policy() -> Arc<dyn Authorizer> {
    Arc::new(RulePolicy::new(
        AccessConfig::parse(
            r#"{
                "roles": { "reader": { "clearance": "internal", "runs": "own" } },
                "everyone": ["reader"]
            }"#,
        )
        .unwrap(),
    ))
}

/// Ingest, hybrid retrieval, the pipeline, the HTTP API, and the run store, with only the models faked.
#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn an_ingested_page_answers_a_question_over_http(pool: PgPool) -> eros::Result<()> {
    let sources = Sources::new(vec![Box::new(StaticSource {
        name: markdown::NAME.clone(),
        documents: vec![roles_page()],
    })])?;
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await?;

    let retriever = HybridRetriever::new(
        pool.clone(),
        Arc::new(HashEmbedder),
        HybridConfig::default(),
    );
    let explain = ScriptedChat::replying([json!({
        "claims": [{
            "text": "ROLE_USERS holds the user-to-role assignments.",
            "chunk_ids": ["markdown:roles.md#roles"]
        }]
    })
    .to_string()]);
    let authorizer = reader_policy();
    let app = router(AppState {
        answerer: Arc::new(PipelineAnswerer::new(ports(
            retriever,
            Arc::new(explain),
            &authorizer,
        ))),
        runs: Arc::new(PgRunStore::new(pool, Arc::new(SystemClock))),
        clock: Arc::new(SystemClock),
        keys: Arc::new("sk-jose:jose".parse().unwrap()),
        authorizer,
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
                    "model": "recall",
                    "messages": [{ "role": "user", "content": "Which table has ROLE_USERS role assignments?" }]
                })
                .to_string(),
            ))?,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{completion:#}");
    assert_eq!(completion["provenance"]["outcome"], "answered");
    assert_eq!(completion["provenance"]["category"], "open");
    assert_eq!(
        completion["provenance"]["citations"][0]["chunk"]["id"],
        "markdown:roles.md#roles"
    );
    assert!(completion["provenance"].get("trace").is_none());
    let content = completion["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(content.contains("[1]"), "{content}");

    let id = completion["provenance"]["run_id"].as_str().unwrap();
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
        [
            "authorize",
            "classify",
            "retrieve",
            "rerank",
            "explain",
            "verify"
        ]
    );
    assert_eq!(run["trace"][0]["record"]["principal"]["id"], "jose");
    Ok(())
}

fn pay_page() -> Document {
    Document {
        source: markdown::NAME.clone(),
        external_id: "hr/pay.md".into(),
        title: "Pay".into(),
        url: "https://wiki.example.test/hr/pay.md".into(),
        updated_at: Utc::now(),
        labels: roles_page().labels,
        chunks: vec![ChunkDraft {
            locator: "bands".into(),
            heading: Some("Role pay bands".into()),
            url: "https://wiki.example.test/hr/pay.md#bands".into(),
            text: "ROLE_USERS role assignments set each user's pay band.".into(),
            standing: recall::domain::Standing::Authoritative,
        }],
    }
}

/// A document the label rules make confidential is never retrieved, prompted, cited or listed for a reader, even
/// when the model cites it.
#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn a_relabelled_confidential_page_never_reaches_a_reader(pool: PgPool) -> eros::Result<()> {
    let sources = Sources::new(vec![Box::new(StaticSource {
        name: markdown::NAME.clone(),
        documents: vec![roles_page(), pay_page()],
    })])?;
    let rules = AccessConfig::parse(
        r#"{ "roles": {}, "labels": [{ "source": "markdown", "path": "hr/", "sensitivity": "confidential" }] }"#,
    )?;
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        rules.labels(),
        &sources,
        false,
    )
    .await?;

    let explain = Arc::new(ScriptedChat::replying([json!({
        "claims": [{
            "text": "ROLE_USERS sets pay bands.",
            "chunk_ids": ["markdown:hr/pay.md#bands"]
        }]
    })
    .to_string()]));
    let retriever = HybridRetriever::new(
        pool.clone(),
        Arc::new(HashEmbedder),
        HybridConfig::default(),
    );
    let authorizer = reader_policy();
    let app = router(AppState {
        answerer: Arc::new(PipelineAnswerer::new(ports(
            retriever,
            Arc::<ScriptedChat>::clone(&explain) as Arc<dyn ChatModel>,
            &authorizer,
        ))),
        runs: Arc::new(PgRunStore::new(pool, Arc::new(SystemClock))),
        clock: Arc::new(SystemClock),
        keys: Arc::new("sk-jose:jose".parse().unwrap()),
        authorizer,
    });

    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::AUTHORIZATION, "Bearer sk-jose")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "messages": [{ "role": "user", "content": "Do ROLE_USERS role assignments set pay bands?" }]
                    })
                    .to_string(),
                ))?,
        )
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let completion: Value = serde_json::from_slice(&body)?;

    assert_eq!(
        completion["provenance"]["outcome"], "declined",
        "{completion:#}"
    );
    assert!(!String::from_utf8_lossy(&body).contains("hr/pay.md"));
    let prompt = &explain.requests()[0].user;
    assert!(prompt.contains("markdown:roles.md#roles"));
    assert!(!prompt.contains("each user's pay band"), "{prompt}");
    Ok(())
}
