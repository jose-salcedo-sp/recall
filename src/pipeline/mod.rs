pub mod admit;
pub mod embed;
pub mod generate;
pub mod kind;
pub mod retrieve;
pub mod verify;

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::Instrument;
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::{RecallError, Result};
use crate::types::{AdmittedCitations, Ask, AskRequest, Stage, StageRecord};

#[derive(Debug, Clone)]
pub enum Progress {
    Entered(Stage),
    Finished { stage: Stage, ms: u64, ok: bool },
}

#[derive(Debug, Clone, Copy)]
pub struct AskIds {
    pub ask_id: Uuid,
    pub trace_id: Option<Uuid>,
}

impl From<&Ask> for AskIds {
    fn from(a: &Ask) -> Self {
        Self {
            ask_id: a.ask_id,
            trace_id: a.trace_id,
        }
    }
}

pub struct ProgressSink {
    tx: Option<mpsc::UnboundedSender<Progress>>,
}

impl ProgressSink {
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Progress>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx: Some(tx) }, rx)
    }

    pub fn silent() -> Self {
        Self { tx: None }
    }

    fn send(&self, p: Progress) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(p);
        }
    }

    pub async fn track<T, F>(&self, ids: AskIds, stage: Stage, fut: F) -> (Result<T>, StageRecord)
    where
        F: Future<Output = Result<T>>,
    {
        let span = tracing::info_span!(
            "stage",
            stage = stage.as_str(),
            ask_id = %ids.ask_id,
            trace_id = ?ids.trace_id,
        );

        self.send(Progress::Entered(stage));
        let started = Instant::now();
        let result = fut.instrument(span.clone()).await;
        let ms = started.elapsed().as_millis() as u64;
        let ok = result.is_ok();
        self.send(Progress::Finished { stage, ms, ok });

        let st = stage.as_str();
        span.in_scope(|| match &result {
            Ok(_) => tracing::info!(stage = st, ms, ask_id = %ids.ask_id, "stage ok"),
            Err(e) => tracing::warn!(
                stage = st, ms, ask_id = %ids.ask_id,
                error = %e, transient = e.is_transient(), "stage failed"
            ),
        });

        (result, StageRecord { stage, ms, ok })
    }
}

macro_rules! stage {
    ($sink:expr, $ask:expr, $stage:expr, $fut:expr) => {{
        let ids = AskIds {
            ask_id: $ask.ask_id,
            trace_id: $ask.trace_id,
        };
        let (res, rec) = $sink.track(ids, $stage, $fut).await;
        $ask.stages.push(rec);
        res?
    }};
}

pub async fn run_until_admit(ctx: &Arc<Ctx>, req: AskRequest, sink: &ProgressSink) -> Result<Ask> {
    req.validate()
        .map_err(|e| RecallError::BadRequest(e.into()))?;
    let mut ask = Ask::new(req);

    tracing::debug!(
        ask_id = %ask.ask_id,
        brain_id = %ask.brain_id,
        question = %ask.question,
        "ask started"
    );
    tracing::info!(
        ask_id = %ask.ask_id,
        brain_id = %ask.brain_id,
        question = %trunc_log(&ask.question, 4000),
        as_of = ?ask.as_of,
        "ask started"
    );

    let (kind, confidence) = stage!(
        sink,
        ask,
        Stage::Kind,
        kind::run(ctx, ask.ask_id, &ask.question)
    );
    ask.kind = Some(kind.clone());
    ask.kind_confidence = Some(confidence);
    if kind::is_chitchat(&kind, confidence, ctx.cfg.kind_confidence_min) {
        ask.chitchat = true;
        return Ok(ask);
    }

    let embedding = stage!(sink, ask, Stage::Embed, embed::run(ctx, &ask.question));
    tracing::info!(ask_id = %ask.ask_id, dims = embedding.len(), "embedded");
    ask.embedding = Some(embedding);

    ask.candidates = stage!(
        sink,
        ask,
        Stage::Retrieve,
        retrieve::run(
            ctx,
            ask.ask_id,
            ask.brain_id,
            ask.embedding.as_ref().expect("embedding set above"),
            &ask.question,
            ask.as_of,
        )
    );
    let top_rrf = ask.candidates.first().map(|c| c.rrf_score).unwrap_or(0.0);
    tracing::info!(
        ask_id = %ask.ask_id,
        merged = ask.candidates.len(),
        top_rrf,
        "retrieve merged"
    );

    let admission = stage!(
        sink,
        ask,
        Stage::Admit,
        admit::run(
            ctx,
            ask.ask_id,
            &ask.question,
            ask.as_of_or_now(),
            &ask.candidates
        )
    );

    for c in ask.candidates.iter_mut() {
        if let Some(s) = admission.scores.get(&c.id) {
            c.injection = Some(s.injection);
            c.contradicts = Some(s.contradicts);
            c.relevant = Some(s.relevant);
            c.evidence = Some(s.evidence);
            c.noul = Some(s.evidence);
            c.route = Some(admit::route(*s, &ctx.cfg));
        }
    }
    ask.admitted = admission.admitted;
    ask.conflicts = admission.conflicts;

    Ok(ask)
}

pub async fn run(ctx: &Arc<Ctx>, req: AskRequest, sink: &ProgressSink) -> Result<Ask> {
    let mut ask = run_until_admit(ctx, req, sink).await?;

    if ask.chitchat {
        let (text, _usage) = stage!(
            sink,
            ask,
            Stage::Generate,
            generate::run_chitchat(ctx, &ask.question)
        );
        log_generated(ask.ask_id, true, 0, &text);
        log_published(ask.ask_id, true, false, &text);
        ask.answer = Some(text);
        return Ok(ask);
    }

    match ask.citations_for_generate() {
        None => {
            log_empty_admission(ask.ask_id, ask.candidates.len());
            let ask = ask.into_empty_admit();
            log_published(ask.ask_id, false, true, ask.answer.as_deref().unwrap_or(""));
            Ok(ask)
        }
        Some(cites) => {
            generate_verify_publish(ctx, &mut ask, &cites, sink).await?;
            Ok(ask)
        }
    }
}

/// Generate, verify, optionally regenerate once, then publish verified text only.
pub async fn generate_verify_publish(
    ctx: &Arc<Ctx>,
    ask: &mut Ask,
    cites: &AdmittedCitations,
    sink: &ProgressSink,
) -> Result<()> {
    let mut attempts = 0u32;
    loop {
        let (text, _usage) = stage!(
            sink,
            ask,
            Stage::Generate,
            generate::run(ctx, &ask.question, cites)
        );
        log_generated(ask.ask_id, false, cites.len(), &text);
        let verdicts = stage!(sink, ask, Stage::Verify, verify::run(ctx, &text, cites));
        ask.verdicts = verdicts.clone();
        let all_ok = verdicts.iter().all(|v| v.verdict == "supports");
        let supported = verdicts.iter().filter(|v| v.verdict == "supports").count();
        let verdicts_json = serde_json::to_string(&verdicts).unwrap_or_else(|_| "[]".into());
        tracing::info!(
            ask_id = %ask.ask_id,
            claims = verdicts.len(),
            supported,
            all_ok,
            attempt = attempts + 1,
            verdicts = %verdicts_json,
            "verify verdicts"
        );
        if all_ok {
            publish_verified(ask, &text, &verdicts);
            return Ok(());
        }
        attempts += 1;
        if attempts > ctx.cfg.verify_regen_max {
            publish_verified(ask, &text, &verdicts);
            return Ok(());
        }
        tracing::info!(
            ask_id = %ask.ask_id,
            attempt = attempts,
            "verify failed; regenerating once"
        );
    }
}

fn publish_verified(ask: &mut Ask, text: &str, verdicts: &[crate::types::ClaimVerdict]) {
    let outcome = verify::apply_publish(text, verdicts, &mut ask.admitted, &mut ask.conflicts);
    log_published(ask.ask_id, false, outcome.empty, &outcome.text);
    ask.answer = Some(outcome.text);
    ask.empty = outcome.empty;
}

pub(crate) fn log_generated(ask_id: Uuid, chitchat: bool, citations: usize, answer: &str) {
    tracing::info!(
        ask_id = %ask_id,
        chitchat,
        citations,
        answer = %trunc_log(answer, 4000),
        "generated"
    );
}

pub(crate) fn log_published(ask_id: Uuid, chitchat: bool, empty: bool, answer: &str) {
    tracing::info!(
        ask_id = %ask_id,
        chitchat,
        empty,
        answer = %trunc_log(answer, 4000),
        "answer published"
    );
}

pub(crate) fn log_empty_admission(ask_id: Uuid, candidates: usize) {
    tracing::info!(
        ask_id = %ask_id,
        candidates,
        "empty admission; skipping generator"
    );
}

fn trunc_log(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}
