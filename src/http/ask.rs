use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::StreamExt;
use serde_json::json;

use crate::ctx::Ctx;
use crate::error::{RecallError, Result};
use crate::pipeline::{self, Progress, ProgressSink};
use crate::store;
use crate::types::{AdmittedEvent, Ask, AskRequest, Stage, SyncResponse};

/// `POST /v1/ask/sync` — JSON, for tests and MCP.
pub async fn sync(
    State(ctx): State<Arc<Ctx>>,
    Json(req): Json<AskRequest>,
) -> Result<Json<SyncResponse>> {
    let _permit = ctx.try_admit_request()?;

    let ask = tokio::time::timeout(
        ctx.cfg.ask_timeout,
        pipeline::run(&ctx, req, &ProgressSink::silent()),
    )
    .await
    .map_err(|_| RecallError::Timeout {
        stage: Stage::Generate,
        ms: ctx.cfg.ask_timeout.as_millis() as u64,
    })??;

    store::save_ask(&ctx, &ask, None).await;

    let mut citations = ask.admitted.clone();
    citations.extend(ask.conflicts.clone());
    Ok(Json(SyncResponse {
        ask_id: ask.ask_id,
        text: ask.answer.clone().unwrap_or_default(),
        citations,
        empty: ask.empty,
        usage: json!({}),
    }))
}

/// `POST /v1/ask` — SSE, the product path.
///
/// The response headers matter as much as the body. RunPod publishes Pod HTTP ports
/// through Cloudflare, which buffers responses and closes a connection that has not
/// produced bytes within 100 seconds. So we disable proxy buffering explicitly and
/// emit a progress event per stage, which is what holds the connection open through a
/// slow admission.
pub async fn stream(
    State(ctx): State<Arc<Ctx>>,
    Json(req): Json<AskRequest>,
) -> Result<impl IntoResponse> {
    let permit = ctx.try_admit_request()?;

    let (sink, mut progress_rx) = ProgressSink::channel();
    let ctx_task = ctx.clone();
    let ping_interval = ctx.cfg.stage_ping_interval;

    // The pipeline runs in its own task so progress events can be forwarded while it
    // is still working, rather than all arriving at the end.
    let (done_tx, mut done_rx) = tokio::sync::oneshot::channel::<Result<Ask>>();
    let ask_timeout = ctx.cfg.ask_timeout;
    tokio::spawn(async move {
        // Bounds the stages before generation. Once tokens start flowing the stream
        // is self-limiting via max_tokens, and aborting mid-answer would be worse
        // than letting it finish.
        let result = match tokio::time::timeout(
            ask_timeout,
            pipeline::run_until_admit(&ctx_task, req, &sink),
        )
        .await
        {
            Ok(r) => r,
            Err(_) => Err(RecallError::Timeout {
                stage: Stage::Admit,
                ms: ask_timeout.as_millis() as u64,
            }),
        };
        let _ = done_tx.send(result);
    });

    let body = async_stream::stream! {
        // Phase 1: forward stage progress until the pre-generation pipeline finishes.
        let admitted_ask = loop {
            tokio::select! {
                Some(p) = progress_rx.recv() => {
                    if let Some(ev) = progress_event(&p) {
                        yield sse(ev);
                    }
                }
                res = &mut done_rx => {
                    match res {
                        Ok(Ok(ask)) => break ask,
                        Ok(Err(e)) => {
                            yield sse(error_event(e.code(), &e.to_string()));
                            return;
                        }
                        Err(_) => {
                            yield sse(error_event("internal", "pipeline task died"));
                            return;
                        }
                    }
                }
            }
        };

        // Drain any progress that landed in the same tick as completion.
        while let Ok(p) = progress_rx.try_recv() {
            if let Some(ev) = progress_event(&p) {
                yield sse(ev);
            }
        }

        let mut ask = admitted_ask;

        if ask.chitchat {
            let mut stream = match pipeline::generate::stream_chitchat(&ctx, &ask.question).await {
                Ok(s) => s,
                Err(e) => {
                    yield sse(error_event(e.code(), &e.to_string()));
                    store::save_ask(&ctx, &ask, Some(&e.to_string())).await;
                    drop(permit);
                    return;
                }
            };
            let gen_started = std::time::Instant::now();
            let mut answer = String::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(text) => {
                        answer.push_str(&text);
                        match Event::default().event("token").json_data(json!({ "text": text })) {
                            Ok(ev) => yield sse(ev),
                            Err(_) => continue,
                        }
                    }
                    Err(e) => {
                        yield sse(error_event(e.code(), &e.to_string()));
                        ask.answer = Some(answer);
                        store::save_ask(&ctx, &ask, Some(&e.to_string())).await;
                        drop(permit);
                        return;
                    }
                }
            }
            let gen_ms = gen_started.elapsed().as_millis() as u64;
            ask.stages.push(crate::types::StageRecord {
                stage: Stage::Generate,
                ms: gen_ms,
                ok: true,
            });
            tracing::info!(stage = "generate", ms = gen_ms, ask_id = %ask.ask_id, "stage ok");
            crate::pipeline::log_generated(ask.ask_id, true, 0, &answer);
            crate::pipeline::log_published(ask.ask_id, true, false, &answer);
            ask.answer = Some(answer);
            yield sse(done_event(&ask));
            store::save_ask(&ctx, &ask, None).await;
            drop(permit);
            return;
        }

        let cites = match ask.citations_for_generate() {
            None => {
                crate::pipeline::log_empty_admission(ask.ask_id, ask.candidates.len());
                let ask = ask.into_empty_admit();
                crate::pipeline::log_published(
                    ask.ask_id,
                    false,
                    true,
                    ask.answer.as_deref().unwrap_or(""),
                );
                yield sse(Event::default()
                    .event("empty")
                    .json_data(json!({ "reason": "no_admitted_citation" }))
                    .unwrap_or_else(|_| Event::default().event("empty").data("{}")));
                yield sse(done_event(&ask));
                store::save_ask(&ctx, &ask, None).await;
                drop(permit);
                return;
            }
            Some(c) => c,
        };

        let shown: Vec<_> = ask
            .admitted
            .iter()
            .chain(ask.conflicts.iter())
            .cloned()
            .collect();
        let admitted_payload = AdmittedEvent {
            citations: shown.as_slice(),
            candidate_count: ask.candidates.len(),
        };
        match Event::default().event("admitted").json_data(&admitted_payload) {
            Ok(ev) => yield sse(ev),
            Err(e) => {
                yield sse(error_event("internal", &format!("serialize admitted: {e}")));
                return;
            }
        }

        // Buffer generation and verify before any answer bytes reach the client.
        // Keep-alive pings hold the connection through this work; stage events for
        // generate/verify are recorded on `ask.stages` and appear in audit, not SSE.
        if let Err(e) = pipeline::generate_verify_publish(&ctx, &mut ask, &cites, &ProgressSink::silent()).await {
            yield sse(error_event(e.code(), &e.to_string()));
            store::save_ask(&ctx, &ask, Some(&e.to_string())).await;
            drop(permit);
            return;
        }

        if ask.empty {
            yield sse(Event::default()
                .event("empty")
                .json_data(json!({ "reason": "no_verified_claim" }))
                .unwrap_or_else(|_| Event::default().event("empty").data("{}")));
        }

        let published = ask.answer.clone().unwrap_or_default();
        match Event::default().event("token").json_data(json!({ "text": published })) {
            Ok(ev) => yield sse(ev),
            Err(e) => {
                yield sse(error_event("internal", &format!("serialize token: {e}")));
                drop(permit);
                return;
            }
        }

        yield sse(
            Event::default()
                .event("verified")
                .json_data(json!({ "claims": ask.verdicts }))
                .unwrap_or_else(|_| Event::default().event("verified").data("{}")),
        );

        yield sse(done_event(&ask));
        store::save_ask(&ctx, &ask, None).await;
        drop(permit);
    };

    // The keep-alive comment is the backstop for Cloudflare's 100-second cap when a
    // stage runs long enough to emit no progress of its own.
    let sse = Sse::new(body).keep_alive(KeepAlive::new().interval(ping_interval).text("ping"));

    Ok((
        [
            (header::CACHE_CONTROL, "no-cache"),
            (header::CONNECTION, "keep-alive"),
            // Without this, Cloudflare (and nginx) buffer the stream and the client
            // sees nothing until the response completes, defeating SSE entirely.
            (header::HeaderName::from_static("x-accel-buffering"), "no"),
        ],
        sse,
    ))
}

/// SSE events are infallible once built; this pins the stream's item type so the
/// `stream!` macro can infer it.
type SseItem = std::result::Result<Event, std::convert::Infallible>;

fn sse(event: Event) -> SseItem {
    Ok(event)
}

fn progress_event(p: &Progress) -> Option<Event> {
    let payload = match p {
        Progress::Entered(stage) => json!({ "stage": stage.as_str(), "status": "started" }),
        Progress::Finished { stage, ms, ok } => json!({
            "stage": stage.as_str(),
            "status": if *ok { "ok" } else { "failed" },
            "ms": ms,
        }),
    };
    Event::default().event("stage").json_data(payload).ok()
}

fn error_event(code: &str, message: &str) -> Event {
    Event::default()
        .event("error")
        .json_data(json!({ "code": code, "message": message }))
        .unwrap_or_else(|_| Event::default().event("error").data("{}"))
}

fn done_event(ask: &Ask) -> Event {
    Event::default()
        .event("done")
        .json_data(done_data(ask))
        .unwrap_or_else(|_| Event::default().event("done").data("{}"))
}

fn done_data(ask: &Ask) -> serde_json::Value {
    let citations: Vec<_> = ask.admitted.iter().chain(ask.conflicts.iter()).collect();
    json!({
        "ask_id": ask.ask_id,
        "text": ask.answer.as_deref().unwrap_or_default(),
        "citations": citations,
        "empty": ask.empty,
        "usage": {
            "candidates": ask.candidates.len(),
            "admitted": ask.admitted.len(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AskRequest, Citation, FilterRoute};
    use uuid::Uuid;

    #[test]
    fn done_data_contains_published_text_and_cited_citations_only() {
        let mut ask = Ask::new(AskRequest {
            question: "Who is on the team?".into(),
            as_of: None,
            brain_id: Uuid::new_v4(),
            trace_id: None,
        });
        let cited_id = Uuid::new_v4();
        let uncited_id = Uuid::new_v4();
        ask.answer = Some("Pepe is an infra engineer. [memory_0]".into());
        ask.admitted.push(Citation {
            id: cited_id,
            index: 0,
            statement: "Pepe is an infra engineer.".into(),
            noul: 0.9,
            origin: "granted".into(),
            grantor_name: Some("Pepe".into()),
            source: Some("onboarding-seed".into()),
            occurred_at: None,
            route: FilterRoute::Include,
            text: "Pepe is an infra engineer.".into(),
        });
        ask.admitted.push(Citation {
            id: uncited_id,
            index: 1,
            statement: "Ana is on the team.".into(),
            noul: 0.8,
            origin: "personal".into(),
            grantor_name: None,
            source: None,
            occurred_at: None,
            route: FilterRoute::Include,
            text: "Ana is on the team.".into(),
        });

        let verdicts = vec![crate::types::ClaimVerdict {
            claim: ask.answer.clone().unwrap(),
            verdict: "supports".into(),
            memory_index: Some(0),
        }];
        let outcome = pipeline::verify::apply_publish(
            ask.answer.as_deref().unwrap(),
            &verdicts,
            &mut ask.admitted,
            &mut ask.conflicts,
        );
        ask.answer = Some(outcome.text);

        let data = done_data(&ask);
        assert_eq!(data["text"], "Pepe is an infra engineer. [memory_0]");
        assert_eq!(data["citations"].as_array().unwrap().len(), 1);
        assert_eq!(data["citations"][0]["id"], cited_id.to_string());
        assert_eq!(data["citations"][0]["index"], 0);
        assert_eq!(data["citations"][0]["grantor_name"], "Pepe");
    }
}
