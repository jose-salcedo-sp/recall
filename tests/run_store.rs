//! Runs persisted in Postgres through the HTTP API.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use recall::{
    domain::UserId,
    pipeline::report::CannedAnswerer,
    server::{AppState, router},
    store::{RunStore, runs::PgRunStore},
};
use serde_json::{Value, json};
use sqlx::PgPool;
use tower::ServiceExt;

#[path = "common/runs.rs"]
mod runs;

use runs::reports;

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn the_status_column_follows_the_outcome(pool: PgPool) -> eros::Result<()> {
    let store = PgRunStore::new(pool.clone());
    for (report, status) in reports() {
        store.save(&UserId("ana".into()), &report).await?;

        let (column,): (String,) = sqlx::query_as("SELECT status FROM pipeline_runs WHERE id = $1")
            .bind(report.id.as_uuid())
            .fetch_one(&pool)
            .await?;
        assert_eq!(column, status);
    }
    Ok(())
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn served_runs_are_persisted_in_postgres(pool: PgPool) -> eros::Result<()> {
    let (report, _) = reports().remove(1);
    let state = AppState {
        answerer: Arc::new(CannedAnswerer {
            category: report.category,
            outcome: report.outcome,
        }),
        runs: Arc::new(PgRunStore::new(pool)),
        keys: Arc::new("sk-ana:ana".parse().unwrap()),
        authorizer: Arc::new(recall::access::rules::RulePolicy::new(
            recall::access::rules::AccessConfig::parse(include_str!("../config/access.json"))?,
        )),
    };
    let app = router(state);
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
            .header(header::AUTHORIZATION, "Bearer sk-ana")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({ "messages": [{ "role": "user", "content": "ROLEADM or USERADM?" }] })
                    .to_string(),
            ))?,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let id = completion["provenance"]["run_id"].as_str().unwrap();

    let (status, run) = send(
        Request::get(format!("/v1/runs/{id}"))
            .header(header::AUTHORIZATION, "Bearer sk-ana")
            .body(Body::empty())?,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(run["status"], "declined");
    assert_eq!(run["question"], "ROLEADM or USERADM?");
    assert_eq!(run["outcome"]["reason"]["kind"], "invalid_model_output");
    Ok(())
}
