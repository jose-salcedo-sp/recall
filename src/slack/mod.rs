//! Developer-facing Slack bot over Socket Mode, sharing the HTTP API's answerer and run store.
//!
//! This module owns the Slack side only: the Web API client ([`api`]), the Socket Mode session ([`socket`]) and
//! mrkdwn rendering ([`render`]). Runs are stored under `slack:{user_id}`. Slack is never a corpus source.

pub mod api;
/// In-memory [`api::SlackApi`] and [`socket::Connector`] for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes;
pub mod render;
pub mod socket;

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use futures::FutureExt;
use serde_json::Value;
use tokio::{
    sync::{Semaphore, watch},
    task::{self, JoinError, JoinSet},
};

use crate::{
    domain::{Question, UserId},
    pipeline::report::{Outcome, RunReport},
    server::AppState,
};

use self::{
    api::{SlackApi, SlackClient},
    socket::{Backoff, Connector, Ending, WebSocketConnector},
};

/// Slack credentials. Both set turns the bot on; both unset or empty leaves it off.
#[derive(Debug, clap::Args)]
pub struct SlackArgs {
    /// App-level token (`xapp-`), for opening Socket Mode connections.
    #[arg(long, env = "SLACK_APP_TOKEN", hide_env_values = true)]
    pub app_token: Option<String>,
    /// Bot token (`xoxb-`), for posting and updating messages.
    #[arg(long, env = "SLACK_BOT_TOKEN", hide_env_values = true)]
    pub bot_token: Option<String>,
}

/// Answers mentions and direct messages in a thread, persisting each run like the HTTP API does.
pub struct Bot {
    api: Arc<dyn SlackApi>,
    connector: Arc<dyn Connector>,
}

impl Bot {
    /// `None` when both tokens are unset. An empty token counts as unset, as in `.env.example`.
    ///
    /// # Errors
    ///
    /// When only one token is set, or the HTTP client cannot be built.
    pub fn from_args(args: &SlackArgs) -> eros::Result<Option<Self>> {
        let set = |token: &Option<String>| token.clone().filter(|t| !t.trim().is_empty());
        match (set(&args.app_token), set(&args.bot_token)) {
            (None, None) => Ok(None),
            (Some(app), Some(bot)) => Ok(Some(Self::new(
                Arc::new(SlackClient::new(app, bot)?),
                Arc::new(WebSocketConnector),
            ))),
            _ => {
                eros::bail!("SLACK_APP_TOKEN and SLACK_BOT_TOKEN must both be set or both unset")
            }
        }
    }

    /// A bot over the given ports; tests pass fakes.
    pub fn new(api: Arc<dyn SlackApi>, connector: Arc<dyn Connector>) -> Self {
        Self { api, connector }
    }

    /// Reconnects until `shutdown` turns true or its sender is dropped, then waits up to
    /// `DRAIN_TIMEOUT` for in-flight events and tells the users of any still unanswered.
    pub async fn run(self, state: AppState, mut shutdown: watch::Receiver<bool>) {
        let mut events = EventTasks::new(Arc::clone(&self.api));
        let mut backoff = Backoff::default();
        loop {
            let on_event = |event: Value| match Ask::from_event(&event) {
                Ok(Some(ask)) => events.spawn(state.clone(), ask),
                Ok(None) => {}
                Err(error) => tracing::warn!(error = format!("{error:#?}"), "slack event failed"),
            };
            let ended = tokio::select! {
                () = stopped(&mut shutdown) => break,
                ended = self.session(&mut backoff, on_event) => ended,
            };
            events.reap();
            match &ended {
                Ok(Ending::Refresh) => tracing::info!("slack refreshed the socket"),
                Ok(Ending::LinkDisabled) => tracing::error!(
                    "slack disabled socket mode for this app; enable it in the app settings. \
                     Retrying every {:?}",
                    Backoff::MAX
                ),
                Err(error) => {
                    tracing::warn!(error = format!("{error:#?}"), "slack socket reconnecting");
                }
            }
            let delay = backoff.after(&ended);
            tokio::select! {
                () = stopped(&mut shutdown) => break,
                () = tokio::time::sleep(delay) => {}
            }
        }
        events.drain(DRAIN_TIMEOUT).await;
    }

    async fn session(
        &self,
        backoff: &mut Backoff,
        on_event: impl FnMut(Value) + Send,
    ) -> eros::Result<Ending> {
        let url = self.api.open_connection().await?;
        let connection = self.connector.connect(&url).await?;
        socket::run_session(connection, backoff, on_event).await
    }
}

async fn stopped(shutdown: &mut watch::Receiver<bool>) {
    // An `Err` means the sender is gone, which also stops the bot.
    let _ = shutdown.wait_for(|stop| *stop).await;
}

/// Each event runs a whole pipeline, which shares the database pool with the HTTP API.
const MAX_CONCURRENT_EVENTS: usize = 4;
/// Events past this many waiting for a pipeline get `BUSY_REPLY`, so a burst cannot queue without end.
const MAX_QUEUED_EVENTS: usize = 16;
/// Busy replies in flight at once; events past them are dropped, so a burst cannot grow the Slack
/// calls without end either.
const MAX_BUSY_REPLIES: usize = 4;
/// With `RESTART_NOTICE_TIMEOUT` and the HTTP drain alongside it, fits in Kubernetes' default 30 s
/// termination grace period.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(20);
/// How long shutdown spends replacing unanswered placeholders with `RESTARTED_REPLY`.
const RESTART_NOTICE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a panicked event spends replacing its placeholder with `FAILED_REPLY`.
const FAILURE_NOTICE_TIMEOUT: Duration = Duration::from_secs(5);

const PLACEHOLDER: &str = "Looking that up...";
const BUSY_REPLY: &str =
    "I'm busy with other questions right now. Please try again in a few minutes.";
const FAILED_REPLY: &str = "Something went wrong while I was answering. Please ask again.";
const RESTARTED_REPLY: &str = "I restarted before I could answer. Please ask again.";

/// A question asked in Slack, and where to answer it.
#[derive(Debug)]
struct Ask {
    question: Question,
    user: String,
    channel: String,
    thread_ts: Option<String>,
}

impl Ask {
    /// `None` for events the bot does not answer: its own messages, edits, and empty text.
    fn from_event(event: &Value) -> eros::Result<Option<Self>> {
        if event.get("bot_id").is_some() || event.get("subtype").is_some() {
            return Ok(None);
        }
        let Some(kind) = event.get("type").and_then(Value::as_str) else {
            return Ok(None);
        };
        let (question, channel, thread_ts) = match kind {
            "app_mention" => {
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                let question = strip_mentions(text);
                if question.is_empty() {
                    return Ok(None);
                }
                let channel = event
                    .get("channel")
                    .and_then(Value::as_str)
                    .ok_or_else(|| eros::error!("app_mention missing channel"))?;
                let thread = event
                    .get("thread_ts")
                    .or_else(|| event.get("ts"))
                    .and_then(Value::as_str);
                (question, channel, thread)
            }
            "message" if event.get("channel_type").and_then(Value::as_str) == Some("im") => {
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                if text.trim().is_empty() {
                    return Ok(None);
                }
                let channel = event
                    .get("channel")
                    .and_then(Value::as_str)
                    .ok_or_else(|| eros::error!("dm missing channel"))?;
                let thread = event.get("thread_ts").and_then(Value::as_str);
                (text.trim().to_owned(), channel, thread)
            }
            _ => return Ok(None),
        };
        let user = event
            .get("user")
            .and_then(Value::as_str)
            .ok_or_else(|| eros::error!("event missing user"))?;
        Ok(Some(Self {
            question: Question::new(question),
            user: user.to_owned(),
            channel: channel.to_owned(),
            thread_ts: thread_ts.map(str::to_owned),
        }))
    }

    async fn post_placeholder(&self, api: &dyn SlackApi) -> eros::Result<Placeholder> {
        let ts = api
            .post_message(&self.channel, self.thread_ts.as_deref(), PLACEHOLDER)
            .await?;
        Ok(Placeholder {
            channel: self.channel.clone(),
            ts,
        })
    }
}

/// A posted "Looking that up..." message, which the answer replaces.
#[derive(Debug, Clone)]
struct Placeholder {
    channel: String,
    ts: String,
}

impl Placeholder {
    async fn replace(&self, api: &dyn SlackApi, text: &str) -> eros::Result<()> {
        api.update_message(&self.channel, &self.ts, text).await
    }
}

/// Event tasks: every one is tracked, at most `MAX_CONCURRENT_EVENTS` run at once, at most
/// `MAX_QUEUED_EVENTS` more wait, and at most `MAX_BUSY_REPLIES` busy replies are in flight.
/// Each admitted task posts its placeholder before it waits.
struct EventTasks {
    api: Arc<dyn SlackApi>,
    tasks: JoinSet<eros::Result<()>>,
    running: Arc<Semaphore>,
    admitted: Arc<Semaphore>,
    busy_replies: Arc<Semaphore>,
    /// Placeholders not yet replaced by an answer, by the task that posted them.
    unanswered: Arc<Mutex<HashMap<task::Id, Placeholder>>>,
}

impl EventTasks {
    fn new(api: Arc<dyn SlackApi>) -> Self {
        Self {
            api,
            tasks: JoinSet::new(),
            running: Arc::new(Semaphore::new(MAX_CONCURRENT_EVENTS)),
            admitted: Arc::new(Semaphore::new(MAX_CONCURRENT_EVENTS + MAX_QUEUED_EVENTS)),
            busy_replies: Arc::new(Semaphore::new(MAX_BUSY_REPLIES)),
            unanswered: Arc::default(),
        }
    }

    fn spawn(&mut self, state: AppState, ask: Ask) {
        self.reap();
        let api = Arc::clone(&self.api);
        let Ok(admitted) = Arc::clone(&self.admitted).try_acquire_owned() else {
            let Ok(replying) = Arc::clone(&self.busy_replies).try_acquire_owned() else {
                tracing::warn!(
                    queued = MAX_QUEUED_EVENTS,
                    "slack event queue and busy replies are full; dropped the event"
                );
                return;
            };
            tracing::warn!(
                queued = MAX_QUEUED_EVENTS,
                "slack event queue is full; replied busy"
            );
            self.tasks.spawn(async move {
                let _replying = replying;
                api.post_message(&ask.channel, ask.thread_ts.as_deref(), BUSY_REPLY)
                    .await?;
                Ok(())
            });
            return;
        };
        let running = Arc::clone(&self.running);
        let unanswered = Arc::clone(&self.unanswered);
        self.tasks.spawn(async move {
            let _admitted = admitted;
            // A post that fails, or that is still in flight at the drain's abort, leaves no ts to
            // edit, so its user gets no restart notice. The client's 30 s request timeout exceeds
            // `DRAIN_TIMEOUT`, so a slow post can be cut off.
            let placeholder = ask.post_placeholder(api.as_ref()).await?;
            lock(&unanswered).insert(task::id(), placeholder.clone());
            let _running = running
                .acquire_owned()
                .await
                .expect("the event semaphore is never closed");
            // Caught here rather than in `reap`, which runs only when the next event arrives or the
            // session ends.
            let text = match AssertUnwindSafe(answer_text(&state, ask))
                .catch_unwind()
                .await
            {
                Ok(text) => text,
                Err(panic) => {
                    lock(&unanswered).remove(&task::id());
                    if let Err(error) = notify_failed(api.as_ref(), &placeholder).await {
                        tracing::warn!(
                            error = format!("{error:#?}"),
                            "slack failure notice failed"
                        );
                    }
                    std::panic::resume_unwind(panic);
                }
            };
            // Removed by hand, not by a drop guard, so an aborted task stays listed for the restart
            // notice; and before the update, since once its request may have reached Slack, a
            // restart notice could overwrite the answer.
            lock(&unanswered).remove(&task::id());
            placeholder.replace(api.as_ref(), &text).await
        });
    }

    fn reap(&mut self) {
        while let Some(finished) = self.tasks.try_join_next() {
            report(finished);
        }
    }

    async fn drain(mut self, within: Duration) {
        let all_done = async {
            while let Some(finished) = self.tasks.join_next().await {
                report(finished);
            }
        };
        if tokio::time::timeout(within, all_done).await.is_err() {
            tracing::warn!(
                abandoned = self.tasks.len(),
                "slack events still running at shutdown were abandoned after {within:?}"
            );
            self.tasks.abort_all();
            while let Some(finished) = self.tasks.join_next().await {
                if !finished.as_ref().is_err_and(JoinError::is_cancelled) {
                    report(finished);
                }
            }
        }
        self.notify_restarted(RESTART_NOTICE_TIMEOUT).await;
    }

    /// Best effort: replaces each unanswered placeholder with `RESTARTED_REPLY`, for at most `within`.
    async fn notify_restarted(&self, within: Duration) {
        let unanswered = std::mem::take(&mut *lock(&self.unanswered));
        if unanswered.is_empty() {
            return;
        }
        let notices = unanswered.values().map(|placeholder| async {
            if let Err(error) = placeholder
                .replace(self.api.as_ref(), RESTARTED_REPLY)
                .await
            {
                tracing::warn!(error = format!("{error:#?}"), "slack restart notice failed");
            }
        });
        if tokio::time::timeout(within, futures::future::join_all(notices))
            .await
            .is_err()
        {
            tracing::warn!("slack restart notices still unsent after {within:?} were abandoned");
        }
    }
}

#[eros::context("replace a failed slack event's placeholder")]
async fn notify_failed(api: &dyn SlackApi, placeholder: &Placeholder) -> eros::Result<()> {
    let notice = placeholder.replace(api, FAILED_REPLY);
    let Ok(replaced) = tokio::time::timeout(FAILURE_NOTICE_TIMEOUT, notice).await else {
        eros::bail!("timed out after {FAILURE_NOTICE_TIMEOUT:?}");
    };
    replaced
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .expect("lock poisoned: no code panics while holding the placeholder lock")
}

fn report(finished: Result<eros::Result<()>, JoinError>) {
    match finished {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::warn!(error = format!("{error:#?}"), "slack event failed"),
        Err(error) if error.is_panic() => {
            tracing::error!(error = format!("{error:#?}"), "slack event task panicked");
        }
        Err(error) => tracing::warn!(error = format!("{error:#?}"), "slack event task cancelled"),
    }
}

/// Answers `event` in Slack at once, with no queue.
#[cfg(test)]
async fn handle_event(state: &AppState, api: &dyn SlackApi, event: &Value) -> eros::Result<()> {
    let Some(ask) = Ask::from_event(event)? else {
        return Ok(());
    };
    let placeholder = ask.post_placeholder(api).await?;
    let text = answer_text(state, ask).await;
    placeholder.replace(api, &text).await
}

async fn answer_text(state: &AppState, ask: Ask) -> String {
    let report = state.answerer.answer(ask.question).await;
    let user = UserId(format!("slack:{}", ask.user));
    persist_and_render(state, &user, &report).await
}

/// Slack replies go to shared channels, so a failure shows only the run id; the detail goes to the log.
async fn persist_and_render(state: &AppState, user: &UserId, report: &RunReport) -> String {
    let run_id = report.id;
    if let Err(error) = state.runs.save(user, report).await {
        tracing::error!(%run_id, error = format!("{error:#?}"), "slack run not persisted");
        return format!("Run {run_id} could not be completed.");
    }
    let id = run_id.to_string();
    match &report.outcome {
        Outcome::Answered(answer) => render::answer(answer, &id),
        Outcome::Declined { sources, .. } => render::declined(sources, &id),
        Outcome::Failed { at, error } => {
            tracing::error!(%run_id, at, error, "slack run failed");
            format!("Run {id} failed at {at}.")
        }
    }
}

/// Removes Slack user mentions (`<@U…>`) from mention text.
pub fn strip_mentions(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            let mut token = String::from("<");
            while let Some(&next) = chars.peek() {
                token.push(next);
                chars.next();
                if next == '>' {
                    break;
                }
            }
            if token.starts_with("<@") {
                continue;
            }
            out.push_str(&token);
        } else {
            out.push(c);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
