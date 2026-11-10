use std::{
    io::Write,
    sync::{Arc, Mutex},
};

use crate::{
    domain::{CategoryKind, Probability, Question},
    pipeline::report::{Answer, Answerer, CannedAnswerer, Outcome, Verdict},
    server::{AppState, auth::AdminUsers},
    slack::{Bot, SlackArgs, fakes::FakeSlack, strip_mentions},
    store::{
        RunStore,
        fakes::{FailingRunStore, MemoryRunStore},
    },
};

fn answered() -> Outcome {
    Outcome::Answered(Answer {
        verdict: Some(Verdict::Boolean {
            yes: true,
            confidence: Probability::new(0.9).unwrap(),
        }),
        why: vec![],
        citations: vec![],
    })
}

fn failed() -> Outcome {
    Outcome::Failed {
        at: "answer".into(),
        error: "POST https://system-one.internal/v1/decide: 502 body=<html>upstream</html>".into(),
    }
}

fn state(answerer: Arc<dyn Answerer>, runs: Arc<dyn RunStore>) -> AppState {
    AppState {
        answerer,
        runs,
        keys: Arc::new("k:u".parse().unwrap()),
        admins: Arc::new(AdminUsers::default()),
    }
}

fn canned(outcome: Outcome) -> Arc<dyn Answerer> {
    Arc::new(CannedAnswerer {
        category: Some(CategoryKind::Boolean),
        outcome,
    })
}

fn mention() -> serde_json::Value {
    serde_json::json!({
        "type": "app_mention",
        "user": "UJOSE",
        "text": "<@UBOT> What is AGR_USERS?",
        "channel": "C1",
        "ts": "100.001"
    })
}

/// Collects the log lines written on this thread while the guard lives.
#[derive(Clone, Default)]
pub(super) struct Logs(Arc<Mutex<Vec<u8>>>);

impl Logs {
    pub(super) fn capture() -> (Self, tracing::subscriber::DefaultGuard) {
        let logs = Self::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        (logs, tracing::subscriber::set_default(subscriber))
    }

    pub(super) fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl Write for Logs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn empty_tokens_leave_the_bot_off() {
    let args = |app: Option<&str>, bot: Option<&str>| SlackArgs {
        app_token: app.map(Into::into),
        bot_token: bot.map(Into::into),
    };
    assert!(Bot::from_args(&args(None, None)).unwrap().is_none());
    assert!(
        Bot::from_args(&args(Some(""), Some(" ")))
            .unwrap()
            .is_none()
    );
    assert!(Bot::from_args(&args(Some("xapp-1"), Some(""))).is_err());
    assert!(
        Bot::from_args(&args(Some("xapp-1"), Some("xoxb-1")))
            .unwrap()
            .is_some()
    );
}

#[test]
fn strip_mention_removes_bot_tag() {
    assert_eq!(
        strip_mentions("<@U123ABC> What is AGR_USERS?"),
        "What is AGR_USERS?"
    );
}

#[tokio::test]
async fn app_mention_posts_placeholder_then_updates_with_answer() {
    let slack = Arc::new(FakeSlack::default());
    let runs = Arc::new(MemoryRunStore::default());
    let state = state(canned(answered()), Arc::<MemoryRunStore>::clone(&runs));
    super::handle_event(&state, slack.as_ref(), &mention())
        .await
        .unwrap();

    let posts = slack.posts.lock().unwrap();
    assert_eq!(posts.len(), 1);
    assert_eq!(posts[0].0, "C1");
    assert_eq!(posts[0].1.as_deref(), Some("100.001"));
    assert_eq!(posts[0].2, "Looking that up...");

    let updates = slack.updates.lock().unwrap();
    assert_eq!(updates.len(), 1);
    assert!(updates[0].2.contains("Run "));
    assert_eq!(updates[0].0, "C1");

    let stored = runs.stored();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].user.0, "slack:UJOSE");
    assert_eq!(
        stored[0].report.question,
        Question::new("What is AGR_USERS?")
    );
}

#[tokio::test]
async fn a_failed_run_posts_only_its_id_and_step_and_logs_the_error() {
    let (logs, _guard) = Logs::capture();
    let slack = Arc::new(FakeSlack::default());
    let runs = Arc::new(MemoryRunStore::default());
    let state = state(canned(failed()), Arc::<MemoryRunStore>::clone(&runs));
    super::handle_event(&state, slack.as_ref(), &mention())
        .await
        .unwrap();

    let run_id = runs.stored()[0].report.id;
    let updates = slack.updates.lock().unwrap();
    assert_eq!(updates[0].2, format!("Run {run_id} failed at answer."));
    let logs = logs.text();
    assert!(logs.contains("system-one.internal"), "{logs}");
    assert_eq!(logs.matches("system-one.internal").count(), 1, "{logs}");
}

#[tokio::test]
async fn an_unsaved_run_posts_only_its_id_and_logs_the_error() {
    let (logs, _guard) = Logs::capture();
    let slack = Arc::new(FakeSlack::default());
    let state = state(canned(failed()), Arc::new(FailingRunStore));
    super::handle_event(&state, slack.as_ref(), &mention())
        .await
        .unwrap();

    let updates = slack.updates.lock().unwrap();
    let text = &updates[0].2;
    assert!(text.starts_with("Run "), "{text}");
    assert!(text.ends_with(" could not be completed."), "{text}");
    assert!(!text.contains("postgres"), "{text}");
    assert!(!text.contains("system-one"), "{text}");
    let logs = logs.text();
    assert_eq!(logs.matches("db.internal").count(), 1, "{logs}");
}

mod socket {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use async_trait::async_trait;
    use futures::{Sink, SinkExt, Stream, StreamExt};
    use serde_json::json;
    use tokio::{sync::watch, time::Instant};
    use tokio_tungstenite::tungstenite::{self, Message};

    use super::*;
    use crate::{
        pipeline::report::RunReport,
        slack::{
            BUSY_REPLY, DRAIN_TIMEOUT, FAILED_REPLY, MAX_BUSY_REPLIES, MAX_CONCURRENT_EVENTS,
            MAX_QUEUED_EVENTS, PLACEHOLDER, RESTART_NOTICE_TIMEOUT, RESTARTED_REPLY,
            api::SlackApi,
            fakes::{FakeSocket, connection_pair},
            socket::{Backoff, Connector, Ending, IDLE_TIMEOUT, WebSocketConnector, run_session},
        },
    };

    fn envelope(json: &serde_json::Value) -> Message {
        Message::text(json.to_string())
    }

    fn hello() -> Message {
        envelope(&json!({ "type": "hello" }))
    }

    fn disconnect(reason: &str) -> Message {
        envelope(&json!({ "type": "disconnect", "reason": reason }))
    }

    /// Runs the bot over `socket` until `stop_after`, and returns when each connect started,
    /// measured from the start.
    async fn connect_offsets(socket: Arc<FakeSocket>, stop_after: Duration) -> Vec<Duration> {
        let bot = Bot::new(
            Arc::new(FakeSlack::default()),
            Arc::<FakeSocket>::clone(&socket),
        );
        let (stop, shutdown) = watch::channel(false);
        let started = Instant::now();
        tokio::join!(
            bot.run(
                state(canned(answered()), Arc::new(MemoryRunStore::default())),
                shutdown
            ),
            async {
                tokio::time::sleep(stop_after).await;
                stop.send(true).unwrap();
            }
        );
        socket
            .connects()
            .iter()
            .map(|at| at.duration_since(started))
            .collect()
    }

    fn gaps(offsets: &[Duration]) -> Vec<Duration> {
        offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Each gap is a failure's jittered delay: the `i`th lies in `[INITIAL * 2^i / 2, INITIAL * 2^i]`.
    fn assert_backs_off(gaps: &[Duration]) {
        for (i, gap) in gaps.iter().enumerate() {
            let full = Backoff::INITIAL * 2u32.pow(u32::try_from(i).unwrap());
            assert!(*gap >= full / 2 && *gap <= full, "gap {i} of {gaps:?}");
        }
    }

    async fn carries_frames_both_ways<S>(
        connector: &dyn Connector,
        url: &str,
        slack: impl Future<Output = S>,
    ) where
        S: Stream<Item = Result<Message, tungstenite::Error>>
            + Sink<Message, Error = tungstenite::Error>
            + Unpin,
    {
        let (bot, mut slack) = tokio::join!(connector.connect(url), slack);
        let mut bot = bot.unwrap();
        slack.send(hello()).await.unwrap();
        assert_eq!(bot.next().await.unwrap().unwrap(), hello());
        bot.send(Message::text("ack")).await.unwrap();
        assert_eq!(slack.next().await.unwrap().unwrap(), Message::text("ack"));
    }

    #[tokio::test]
    async fn fake_connector_carries_frames_both_ways() {
        let socket = FakeSocket::default();
        let slack = socket.accept().await;
        carries_frames_both_ways(&socket, "", async { slack }).await;
    }

    #[tokio::test]
    async fn websocket_connector_carries_frames_both_ways() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let slack = async move {
            let (tcp, _) = listener.accept().await.unwrap();
            tokio_tungstenite::accept_async(tcp).await.unwrap()
        };
        carries_frames_both_ways(&WebSocketConnector, &url, slack).await;
    }

    #[tokio::test]
    async fn fake_connector_refusal_is_an_error() {
        let socket = FakeSocket::default();
        socket.refuse();
        assert!(socket.connect("").await.is_err());
    }

    #[tokio::test]
    async fn websocket_connector_refusal_is_an_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        drop(listener);
        assert!(WebSocketConnector.connect(&url).await.is_err());
    }

    async fn session_ending(frames: Vec<Message>) -> eros::Result<Ending> {
        let (bot, mut slack) = connection_pair().await;
        for frame in frames {
            slack.send(frame).await.unwrap();
        }
        run_session(bot, &mut Backoff::default(), |_| {}).await
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_requested_and_a_close_after_hello_are_routine() {
        let refreshed = session_ending(vec![hello(), disconnect("refresh_requested")]).await;
        assert_eq!(refreshed.unwrap(), Ending::Refresh);
        let closed = session_ending(vec![hello(), Message::Close(None)]).await;
        assert_eq!(closed.unwrap(), Ending::Refresh);
    }

    #[tokio::test(start_paused = true)]
    async fn link_disabled_ends_the_session_as_disabled() {
        let ended = session_ending(vec![hello(), disconnect("link_disabled")]).await;
        assert_eq!(ended.unwrap(), Ending::LinkDisabled);
    }

    #[tokio::test(start_paused = true)]
    async fn too_many_websockets_and_unknown_reasons_are_failures() {
        for reason in ["too_many_websockets", "something_new"] {
            let error = session_ending(vec![hello(), disconnect(reason)])
                .await
                .unwrap_err();
            assert!(format!("{error:#?}").contains(reason), "{error:#?}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_warning_keeps_the_session_serving() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        slack.send(disconnect("warning")).await.unwrap();
        let event = json!({ "type": "app_mention" });
        slack
            .send(envelope(&json!({
                "type": "events_api",
                "envelope_id": "E1",
                "payload": { "event": event },
            })))
            .await
            .unwrap();
        slack.send(disconnect("refresh_requested")).await.unwrap();
        let mut events = Vec::new();
        let ended = run_session(bot, &mut Backoff::default(), |event| events.push(event)).await;
        assert_eq!(ended.unwrap(), Ending::Refresh);
        assert_eq!(events, [event]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_close_before_hello_is_a_failure() {
        assert!(session_ending(vec![Message::Close(None)]).await.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn a_malformed_envelope_is_skipped_and_the_next_event_still_arrives() {
        let (logs, _guard) = Logs::capture();
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        slack.send(Message::text("{not json")).await.unwrap();
        let event = json!({ "type": "app_mention" });
        slack
            .send(envelope(&json!({
                "type": "events_api",
                "envelope_id": "E1",
                "payload": { "event": event },
            })))
            .await
            .unwrap();
        slack.send(Message::Close(None)).await.unwrap();

        let mut events = Vec::new();
        run_session(bot, &mut Backoff::default(), |event| events.push(event))
            .await
            .unwrap();

        assert_eq!(events, [event]);
        assert_eq!(
            slack.next().await.unwrap().unwrap(),
            envelope(&json!({ "envelope_id": "E1" }))
        );
        assert!(logs.text().contains("malformed slack envelope"));
    }

    /// Delivers `frames`, then nothing; a send never completes, like a stalled link.
    struct StalledSends(std::collections::VecDeque<Message>);

    impl Stream for StalledSends {
        type Item = Result<Message, tungstenite::Error>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            match self.0.pop_front() {
                Some(frame) => std::task::Poll::Ready(Some(Ok(frame))),
                None => std::task::Poll::Pending,
            }
        }
    }

    impl Sink<Message> for StalledSends {
        type Error = tungstenite::Error;

        fn poll_ready(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }

        fn start_send(self: std::pin::Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
            Ok(())
        }

        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }

        fn poll_close(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }
    }

    #[tokio::test(start_paused = true)]
    async fn an_event_is_queued_even_when_its_ack_never_finishes() {
        let event = json!({ "type": "app_mention" });
        let connection = Box::pin(StalledSends(
            [
                hello(),
                envelope(&json!({
                    "type": "events_api",
                    "envelope_id": "E1",
                    "payload": { "event": event },
                })),
            ]
            .into(),
        ));
        let mut events = Vec::new();
        let mut backoff = Backoff::default();
        tokio::select! {
            _ = run_session(connection, &mut backoff, |event| events.push(event)) => {
                panic!("the session cannot end while its ack is stalled")
            }
            () = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
        assert_eq!(events, [event]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_silent_link_ends_the_session_after_the_idle_timeout() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        let started = Instant::now();
        let error = run_session(bot, &mut Backoff::default(), |_| {})
            .await
            .unwrap_err();
        let waited = started.elapsed();
        assert!(waited >= IDLE_TIMEOUT && waited < IDLE_TIMEOUT + Duration::from_secs(1));
        assert!(format!("{error:#?}").contains("no message from slack"));
    }

    #[tokio::test(start_paused = true)]
    async fn any_frame_restarts_the_idle_timeout() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        let started = Instant::now();
        let ping_at = Duration::from_secs(50);
        let mut backoff = Backoff::default();
        let (ended, ()) = tokio::join!(run_session(bot, &mut backoff, |_| {}), async {
            tokio::time::sleep(ping_at).await;
            slack.send(Message::Ping(Vec::new().into())).await.unwrap();
        });
        assert!(ended.is_err());
        let waited = started.elapsed();
        assert!(waited >= ping_at + IDLE_TIMEOUT, "{waited:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_healthy_session_resets_the_backoff_and_its_refresh_waits_only_the_floor() {
        let socket = Arc::new(FakeSocket::default());
        socket.refuse();
        socket.refuse();
        let mut slack = socket.accept().await;
        slack.send(hello()).await.unwrap();
        socket.refuse();
        let refresh_at = Duration::from_secs(45);
        let (offsets, ()) = tokio::join!(
            connect_offsets(Arc::clone(&socket), Duration::from_secs(60)),
            async {
                tokio::time::sleep(refresh_at).await;
                slack.send(disconnect("refresh_requested")).await.unwrap();
            }
        );

        assert_eq!(offsets.len(), 5, "{offsets:?}");
        let gaps = gaps(&offsets);
        // Two refusals back off, then the session is up from about 3 s to `refresh_at`.
        assert_backs_off(&gaps[..2]);
        assert_eq!(offsets[3], refresh_at + Backoff::FLOOR);
        // The refusal after it waits about `INITIAL` again, not the 4 s it had grown to.
        assert_backs_off(&gaps[3..]);
    }

    #[tokio::test(start_paused = true)]
    async fn sessions_slack_ends_soon_after_hello_back_off() {
        for ending in [
            Message::Close(None),
            disconnect("refresh_requested"),
            disconnect("too_many_websockets"),
            disconnect("something_new"),
        ] {
            let socket = Arc::new(FakeSocket::default());
            let mut ends = Vec::new();
            for _ in 0..5 {
                let mut slack = socket.accept().await;
                slack.send(hello()).await.unwrap();
                slack.send(ending.clone()).await.unwrap();
                ends.push(slack);
            }
            let offsets = connect_offsets(socket, Duration::from_secs(60)).await;
            assert_eq!(offsets.len(), 6, "{ending:?}: {offsets:?}");
            assert_backs_off(&gaps(&offsets));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn link_disabled_logs_an_error_and_retries_after_the_max_delay() {
        let (logs, _guard) = Logs::capture();
        let socket = Arc::new(FakeSocket::default());
        let mut slack = socket.accept().await;
        slack.send(hello()).await.unwrap();
        slack.send(disconnect("link_disabled")).await.unwrap();
        let offsets = connect_offsets(socket, Backoff::MAX * 2).await;
        assert_eq!(offsets, [Duration::ZERO, Backoff::MAX]);
        let logs = logs.text();
        assert!(logs.contains("ERROR"), "{logs}");
        assert!(logs.contains("disabled socket mode"), "{logs}");
    }

    /// Answers after `delay`, recording the most answers running at once.
    struct SlowAnswerer {
        delay: Duration,
        running: AtomicUsize,
        peak: AtomicUsize,
    }

    impl SlowAnswerer {
        fn new(delay: Duration) -> Self {
            Self {
                delay,
                running: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Answerer for SlowAnswerer {
        async fn answer(&self, question: Question) -> RunReport {
            let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            self.running.fetch_sub(1, Ordering::SeqCst);
            CannedAnswerer {
                category: Some(CategoryKind::Boolean),
                outcome: answered(),
            }
            .answer(question)
            .await
        }
    }

    struct PanickingAnswerer;

    #[async_trait]
    impl Answerer for PanickingAnswerer {
        async fn answer(&self, _: Question) -> RunReport {
            panic!("answerer bug")
        }
    }

    /// Runs the bot over one session that delivers `events`, and asks it to stop after `stop_after`.
    /// Returns the fake Web API and how long `Bot::run` took.
    async fn serve_events(
        answerer: Arc<dyn Answerer>,
        events: &[serde_json::Value],
        stop_after: Duration,
    ) -> (Arc<FakeSlack>, Duration) {
        let slack = Arc::new(FakeSlack::default());
        let (ran, ()) = serve_events_while(
            Arc::<FakeSlack>::clone(&slack),
            answerer,
            events,
            stop_after,
            async {},
        )
        .await;
        (slack, ran)
    }

    /// `serve_events` over `slack`, running `alongside` at the same time and returning its output.
    async fn serve_events_while<T>(
        slack: Arc<dyn SlackApi>,
        answerer: Arc<dyn Answerer>,
        events: &[serde_json::Value],
        stop_after: Duration,
        alongside: impl Future<Output = T>,
    ) -> (Duration, T) {
        let socket = Arc::new(FakeSocket::default());
        let mut slack_end = socket.accept().await;
        slack_end.send(hello()).await.unwrap();
        for (i, event) in events.iter().enumerate() {
            slack_end
                .send(envelope(&json!({
                    "type": "events_api",
                    "envelope_id": format!("E{i}"),
                    "payload": { "event": event },
                })))
                .await
                .unwrap();
        }
        let bot = Bot::new(slack, socket);
        let (stop, shutdown) = watch::channel(false);
        let started = Instant::now();
        let ((), (), alongside) = tokio::join!(
            bot.run(
                state(answerer, Arc::new(MemoryRunStore::default())),
                shutdown
            ),
            async {
                tokio::time::sleep(stop_after).await;
                stop.send(true).unwrap();
            },
            alongside
        );
        (started.elapsed(), alongside)
    }

    /// Posts like `FakeSlack`; an update reaches it, but the reply never comes back.
    struct HangingUpdates(Arc<FakeSlack>);

    #[async_trait]
    impl SlackApi for HangingUpdates {
        async fn open_connection(&self) -> eros::Result<String> {
            self.0.open_connection().await
        }

        async fn post_message(
            &self,
            channel: &str,
            thread_ts: Option<&str>,
            text: &str,
        ) -> eros::Result<String> {
            self.0.post_message(channel, thread_ts, text).await
        }

        async fn update_message(&self, channel: &str, ts: &str, text: &str) -> eros::Result<()> {
            self.0.update_message(channel, ts, text).await?;
            std::future::pending().await
        }
    }

    /// Like `FakeSlack`, but each post takes `POST_TAKES`.
    struct SlowPosts(Arc<FakeSlack>);

    const POST_TAKES: Duration = Duration::from_secs(1);

    #[async_trait]
    impl SlackApi for SlowPosts {
        async fn open_connection(&self) -> eros::Result<String> {
            self.0.open_connection().await
        }

        async fn post_message(
            &self,
            channel: &str,
            thread_ts: Option<&str>,
            text: &str,
        ) -> eros::Result<String> {
            tokio::time::sleep(POST_TAKES).await;
            self.0.post_message(channel, thread_ts, text).await
        }

        async fn update_message(&self, channel: &str, ts: &str, text: &str) -> eros::Result<()> {
            self.0.update_message(channel, ts, text).await
        }
    }

    #[tokio::test(start_paused = true)]
    async fn events_past_the_busy_reply_limit_are_dropped_and_logged() {
        let (logs, _guard) = Logs::capture();
        let slack = Arc::new(FakeSlack::default());
        let admitted = MAX_CONCURRENT_EVENTS + MAX_QUEUED_EVENTS;
        let events = vec![mention(); admitted + MAX_BUSY_REPLIES + 3];
        serve_events_while(
            Arc::new(SlowPosts(Arc::clone(&slack))),
            Arc::new(SlowAnswerer::new(Duration::from_secs(5))),
            &events,
            Duration::from_secs(60),
            async {},
        )
        .await;
        let posts = slack.posts.lock().unwrap();
        assert_eq!(texts(&posts, PLACEHOLDER), admitted);
        assert_eq!(texts(&posts, BUSY_REPLY), MAX_BUSY_REPLIES);
        let logs = logs.text();
        assert_eq!(logs.matches("dropped the event").count(), 3, "{logs}");
    }

    #[tokio::test(start_paused = true)]
    async fn an_answer_already_sent_at_abort_is_not_overwritten_by_a_restart_notice() {
        let slack = Arc::new(FakeSlack::default());
        serve_events_while(
            Arc::new(HangingUpdates(Arc::clone(&slack))),
            canned(answered()),
            &[mention()],
            Duration::from_secs(1),
            async {},
        )
        .await;
        let updates = slack.updates.lock().unwrap();
        assert_eq!(updates.len(), 1, "{updates:?}");
        assert!(updates[0].2.contains("Run "), "{updates:?}");
    }

    fn texts(messages: &[(String, Option<String>, String)], text: &str) -> usize {
        messages.iter().filter(|(_, _, t)| t == text).count()
    }

    #[tokio::test(start_paused = true)]
    async fn queued_events_post_their_placeholder_before_they_wait() {
        let slack = Arc::new(FakeSlack::default());
        let events = vec![mention(); MAX_CONCURRENT_EVENTS + 2];
        let (_, posted_early) = serve_events_while(
            Arc::<FakeSlack>::clone(&slack),
            Arc::new(SlowAnswerer::new(Duration::from_secs(10))),
            &events,
            Duration::from_secs(30),
            async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                slack.posts.lock().unwrap().clone()
            },
        )
        .await;
        assert_eq!(texts(&posted_early, PLACEHOLDER), events.len());
        assert_eq!(slack.updates.lock().unwrap().len(), events.len());
    }

    #[tokio::test(start_paused = true)]
    async fn events_past_the_queue_limit_get_a_busy_reply() {
        let answerer = Arc::new(SlowAnswerer::new(Duration::from_secs(5)));
        let admitted = MAX_CONCURRENT_EVENTS + MAX_QUEUED_EVENTS;
        let events = vec![mention(); admitted + 2];
        let (slack, _) = serve_events(
            Arc::<SlowAnswerer>::clone(&answerer),
            &events,
            Duration::from_secs(60),
        )
        .await;
        let posts = slack.posts.lock().unwrap();
        assert_eq!(texts(&posts, PLACEHOLDER), admitted);
        assert_eq!(texts(&posts, BUSY_REPLY), 2);
        assert!(posts.iter().all(|(channel, thread, _)| {
            channel == "C1" && thread.as_deref() == Some("100.001")
        }));
        assert_eq!(slack.updates.lock().unwrap().len(), admitted);
        assert_eq!(answerer.peak.load(Ordering::SeqCst), MAX_CONCURRENT_EVENTS);
    }

    #[tokio::test(start_paused = true)]
    async fn at_most_the_limit_of_events_run_at_once() {
        let answerer = Arc::new(SlowAnswerer::new(Duration::from_secs(5)));
        let events = vec![mention(); 3 * MAX_CONCURRENT_EVENTS];
        let (slack, _) = serve_events(
            Arc::<SlowAnswerer>::clone(&answerer),
            &events,
            Duration::from_secs(50),
        )
        .await;
        assert_eq!(answerer.peak.load(Ordering::SeqCst), MAX_CONCURRENT_EVENTS);
        assert_eq!(slack.updates.lock().unwrap().len(), events.len());
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_waits_for_in_flight_events() {
        let answer_takes = Duration::from_secs(10);
        let (slack, ran) = serve_events(
            Arc::new(SlowAnswerer::new(answer_takes)),
            &[mention()],
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(slack.updates.lock().unwrap().len(), 1);
        assert!(
            ran >= answer_takes && ran < answer_takes + Duration::from_secs(1),
            "{ran:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_abandons_events_still_running_after_the_drain_timeout() {
        let (logs, _guard) = Logs::capture();
        let stop_after = Duration::from_secs(1);
        let (slack, ran) = serve_events(
            Arc::new(SlowAnswerer::new(Duration::from_secs(3600))),
            &[mention()],
            stop_after,
        )
        .await;
        let deadline = stop_after + DRAIN_TIMEOUT;
        assert!(
            ran >= deadline && ran < deadline + Duration::from_secs(1),
            "{ran:?}"
        );
        assert_eq!(slack.updates.lock().unwrap().len(), 1);
        let logs = logs.text();
        assert!(logs.contains("abandoned=1"), "{logs}");
        assert!(!logs.contains("cancelled"), "{logs}");
    }

    #[tokio::test(start_paused = true)]
    async fn abandoned_events_get_a_restart_notice_in_place_of_their_placeholder() {
        let events = vec![mention(); MAX_CONCURRENT_EVENTS + 1];
        let (slack, _) = serve_events(
            Arc::new(SlowAnswerer::new(Duration::from_secs(3600))),
            &events,
            Duration::from_secs(1),
        )
        .await;
        let mut placeholders: Vec<_> = slack
            .posts
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, (channel, _, _))| (channel.clone(), format!("{}.0001", i + 1)))
            .collect();
        let mut notices: Vec<_> = slack
            .updates
            .lock()
            .unwrap()
            .iter()
            .map(|(channel, ts, text)| {
                assert_eq!(text, RESTARTED_REPLY);
                (channel.clone(), ts.clone())
            })
            .collect();
        placeholders.sort();
        notices.sort();
        assert_eq!(placeholders.len(), events.len());
        assert_eq!(notices, placeholders);
    }

    #[tokio::test(start_paused = true)]
    async fn restart_notices_give_up_after_their_timeout() {
        let (logs, _guard) = Logs::capture();
        let stop_after = Duration::from_secs(1);
        let (ran, ()) = serve_events_while(
            Arc::new(HangingUpdates(Arc::default())),
            Arc::new(SlowAnswerer::new(Duration::from_secs(3600))),
            &[mention(), mention()],
            stop_after,
            async {},
        )
        .await;
        let deadline = stop_after + DRAIN_TIMEOUT + RESTART_NOTICE_TIMEOUT;
        assert!(
            ran >= deadline && ran < deadline + Duration::from_secs(1),
            "{ran:?}"
        );
        let logs = logs.text();
        assert_eq!(
            logs.matches("restart notices still unsent").count(),
            1,
            "{logs}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_panicked_event_replaces_its_placeholder_at_once() {
        let slack = Arc::new(FakeSlack::default());
        let (_, updated_early) = serve_events_while(
            Arc::<FakeSlack>::clone(&slack),
            Arc::new(PanickingAnswerer),
            &[mention()],
            Duration::from_secs(30),
            async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                slack.updates.lock().unwrap().clone()
            },
        )
        .await;
        let placeholder_ts = slack.posts.lock().unwrap().len().to_string() + ".0001";
        assert_eq!(
            updated_early,
            [("C1".to_owned(), placeholder_ts, FAILED_REPLY.to_owned())]
        );
        assert_eq!(slack.updates.lock().unwrap().len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn event_panics_and_errors_reach_the_log() {
        let (logs, _guard) = Logs::capture();
        let no_user = json!({ "type": "app_mention", "text": "hi", "channel": "C1" });
        serve_events(
            Arc::new(PanickingAnswerer),
            &[mention(), no_user],
            Duration::from_secs(5),
        )
        .await;
        let logs = logs.text();
        assert!(logs.contains("slack event task panicked"), "{logs}");
        assert!(logs.contains("answerer bug"), "{logs}");
        assert!(logs.contains("slack event failed"), "{logs}");
        assert!(logs.contains("event missing user"), "{logs}");
    }
}

#[tokio::test]
async fn bot_messages_are_ignored() {
    let slack = Arc::new(FakeSlack::default());
    let state = state(canned(failed()), Arc::new(MemoryRunStore::default()));
    super::handle_event(
        &state,
        slack.as_ref(),
        &serde_json::json!({ "type": "message", "bot_id": "B1", "text": "hi" }),
    )
    .await
    .unwrap();
    assert!(slack.posts.lock().unwrap().is_empty());
}
