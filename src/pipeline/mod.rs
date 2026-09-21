pub mod admit;
pub mod embed;
pub mod generate;
pub mod retrieve;

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::Instrument;
use uuid::Uuid;

use crate::ctx::Ctx;
use crate::error::{RecallError, Result};
use crate::types::{AdmittedCitations, Ask, AskRequest, Stage, StageRecord};

/// What a stage transition reports outward while the ask is still in flight.
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

/// One mechanism, three consumers: the SSE stream shows the client where we are, a
/// tracing span gives the operator per-stage timings, and the returned records
/// accumulate on the ask row for `GET /v1/asks/{id}`.
///
/// The SSE half is not cosmetic. RunPod's ingress closes a connection that has not
/// produced bytes within 100 seconds, so emitting progress during a slow admission is
/// what keeps the stream alive.
pub struct ProgressSink {
    tx: Option<mpsc::UnboundedSender<Progress>>,
}

impl ProgressSink {
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<Progress>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx: Some(tx) }, rx)
    }

    /// For `/v1/ask/sync`, where nobody is watching mid-flight.
    pub fn silent() -> Self {
        Self { tx: None }
    }

    fn send(&self, p: Progress) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(p);
        }
    }

    /// Wrap one stage: time it, trace it, report it, and hand back its record.
    ///
    /// Returns the record alongside the result rather than mutating the `Ask`, so a
    /// stage future is free to borrow the `Ask` it reads from.
    pub async fn track<T, F>(
        &self,
        ids: AskIds,
        stage: Stage,
        fut: F,
    ) -> (Result<T>, StageRecord)
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

        // `.instrument` rather than holding an entered guard across the await, which
        // would attribute time to the wrong span whenever the task yields.
        let result = fut.instrument(span.clone()).await;

        let ms = started.elapsed().as_millis() as u64;
        let ok = result.is_ok();
        self.send(Progress::Finished { stage, ms, ok });

        span.in_scope(|| match &result {
            Ok(_) => tracing::info!(ms, "stage ok"),
            Err(e) => tracing::warn!(ms, error = %e, transient = e.is_transient(), "stage failed"),
        });

        (result, StageRecord { stage, ms, ok })
    }
}

/// Run one stage, record it on the ask, and short-circuit on failure.
macro_rules! stage {
    ($sink:expr, $ask:expr, $stage:expr, $fut:expr) => {{
        let ids = AskIds::from(&$ask);
        let (res, rec) = $sink.track(ids, $stage, $fut).await;
        $ask.stages.push(rec);
        res?
    }};
}

/// The ask pipeline up to and including admission.
///
/// Stops before generation because the two entry points diverge there: `/v1/ask`
/// streams tokens, `/v1/ask/sync` blocks for the whole answer. Everything before that
/// point is identical, so it lives here once.
///
/// Four stages in a straight line, so there is no framework: a sequence of async fns
/// over a state struct that moves from one to the next is the pipeline. The `embed`
/// and `admit` stages await queue-backed workers, but the `Ask` never leaves this
/// process, which is why `Stage` is runtime data rather than a type parameter.
pub async fn run_until_admit(
    ctx: &Arc<Ctx>,
    req: AskRequest,
    sink: &ProgressSink,
) -> Result<Ask> {
    req.validate().map_err(|e| RecallError::BadRequest(e.into()))?;
    let mut ask = Ask::new(req);

    let embedding = stage!(sink, ask, Stage::Embed, embed::run(ctx, &ask.question));
    ask.embedding = Some(embedding);

    ask.candidates = stage!(
        sink,
        ask,
        Stage::Retrieve,
        retrieve::run(
            ctx,
            ask.brain_id,
            ask.embedding.as_ref().expect("embedding set above"),
            &ask.question,
            ask.as_of,
        )
    );

    let admission = stage!(
        sink,
        ask,
        Stage::Admit,
        admit::run(ctx, &ask.question, ask.as_of_or_now(), &ask.candidates)
    );

    // Write every score back, admitted or not, so the ask record carries the
    // negatives that a calibration fit needs.
    for c in ask.candidates.iter_mut() {
        c.noul = admission.scores.get(&c.id).copied();
    }
    ask.admitted = admission.admitted;

    Ok(ask)
}

/// The full synchronous ask, for `/v1/ask/sync`.
pub async fn run(ctx: &Arc<Ctx>, req: AskRequest, sink: &ProgressSink) -> Result<Ask> {
    let mut ask = run_until_admit(ctx, req, sink).await?;

    // The one branch that matters. `AdmittedCitations` cannot hold an empty set, so
    // there is no path from here to the generator without evidence.
    match AdmittedCitations::new(ask.admitted.clone()) {
        None => {
            tracing::info!(
                candidates = ask.candidates.len(),
                "empty admission; skipping generator"
            );
            Ok(ask.into_empty_admit())
        }
        Some(cites) => {
            let (text, _usage) = stage!(
                sink,
                ask,
                Stage::Generate,
                generate::run(ctx, &ask.question, &cites)
            );
            ask.answer = Some(text);
            Ok(ask)
        }
    }
}
