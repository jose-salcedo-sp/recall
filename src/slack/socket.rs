//! Socket Mode: ack envelopes, dispatch events, end the session on Slack's request or a silent link.

use std::{
    hash::{BuildHasher, Hasher, RandomState},
    pin::Pin,
    time::Duration,
};

use async_trait::async_trait;
use futures::{Sink, SinkExt, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::time::Instant;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
};

/// Covers DNS, TCP, and the TLS and WebSocket handshakes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Slack pings a healthy connection well within this; silence this long means a half-open link.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// A WebSocket-shaped message stream and sink; lets tests use an in-memory duplex in place of a socket.
pub trait Transport:
    Stream<Item = Result<Message, tungstenite::Error>>
    + Sink<Message, Error = tungstenite::Error>
    + Send
{
}

impl<T> Transport for T where
    T: Stream<Item = Result<Message, tungstenite::Error>>
        + Sink<Message, Error = tungstenite::Error>
        + Send
{
}

/// An open Socket Mode link.
pub type Connection = Pin<Box<dyn Transport>>;

/// Opens the WebSocket at the URL `apps.connections.open` returned.
#[async_trait]
pub trait Connector: Send + Sync {
    /// # Errors
    ///
    /// When the handshake fails or takes longer than its connect timeout.
    async fn connect(&self, url: &str) -> eros::Result<Connection>;
}

/// The production [`Connector`], over TLS.
pub struct WebSocketConnector;

#[async_trait]
impl Connector for WebSocketConnector {
    async fn connect(&self, url: &str) -> eros::Result<Connection> {
        connect(url).await
    }
}

/// The URL carries a one-time ticket, so it stays out of the error context.
#[eros::context("connect the Slack Socket Mode websocket")]
async fn connect(url: &str) -> eros::Result<Connection> {
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded by CONNECT_TIMEOUT below"
    )]
    let connecting = connect_async(url);
    let Ok(connected) = tokio::time::timeout(CONNECT_TIMEOUT, connecting).await else {
        eros::bail!("timed out after {CONNECT_TIMEOUT:?}");
    };
    let (ws, _) = connected?;
    Ok(Box::pin(ws))
}

/// How Slack ended a session without it failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// A routine refresh: `disconnect` with `refresh_requested`, or a close after `hello`.
    Refresh,
    /// `disconnect` with `link_disabled`: Socket Mode is off in the app settings, and only an
    /// operator can turn it back on.
    LinkDisabled,
}

/// Reconnect delays. Each failure doubles the delay from `INITIAL` up to `MAX`, with jitter.
/// A session that stays up `HEALTHY_AFTER` past `hello` resets the delay, and its routine refresh
/// reconnects after `FLOOR`.
#[derive(Debug)]
pub struct Backoff {
    next: Duration,
    greeted_at: Option<Instant>,
}

impl Backoff {
    /// The delay after the first failure.
    pub const INITIAL: Duration = Duration::from_secs(1);
    /// The longest delay.
    pub const MAX: Duration = Duration::from_secs(60);
    /// The least wait between connects, so even sessions that each count as healthy cannot loop.
    pub const FLOOR: Duration = Duration::from_millis(500);
    /// A session that Slack ends sooner than this after `hello` counts as a failure.
    pub const HEALTHY_AFTER: Duration = Duration::from_secs(30);

    /// Records that Slack sent `hello` on the current session.
    pub fn greeted(&mut self) {
        self.greeted_at = Some(Instant::now());
    }

    /// The wait before the next connect, after a session that ended with `ended`.
    pub fn after<E>(&mut self, ended: &Result<Ending, E>) -> Duration {
        let healthy = self
            .greeted_at
            .take()
            .is_some_and(|at| at.elapsed() >= Self::HEALTHY_AFTER);
        if healthy {
            self.next = Self::INITIAL;
        }
        match ended {
            Ok(Ending::Refresh) if healthy => Self::FLOOR,
            Ok(Ending::LinkDisabled) => {
                self.next = Self::MAX;
                Self::MAX
            }
            Ok(Ending::Refresh) | Err(_) => {
                let delay = self.next;
                self.next = (delay * 2).min(Self::MAX);
                jitter(delay).max(Self::FLOOR)
            }
        }
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            next: Self::INITIAL,
            greeted_at: None,
        }
    }
}

/// Spreads `delay` over `[delay / 2, delay]`, so bots that failed together do not reconnect together.
fn jitter(delay: Duration) -> Duration {
    // Each `RandomState` has fresh random keys, so hashing nothing still yields a random value.
    let random = RandomState::new().build_hasher().finish();
    let fraction = random as f64 / u64::MAX as f64;
    delay.mul_f64(1.0 - fraction / 2.0)
}

/// Serves one connection until it fails or Slack ends it, and tells `backoff` when `hello` arrives.
/// `on_event` must not block: it runs on the session loop, before the event's ack.
///
/// A `disconnect` with `too_many_websockets` or an unknown reason is a failure.
#[eros::context("serve a Slack Socket Mode session")]
pub async fn run_session(
    connection: Connection,
    backoff: &mut Backoff,
    on_event: impl FnMut(Value) + Send,
) -> eros::Result<Ending> {
    let (mut connection, mut on_event) = (connection, on_event);
    let mut greeted = false;
    loop {
        let Ok(message) = tokio::time::timeout(IDLE_TIMEOUT, connection.next()).await else {
            eros::bail!("no message from slack within {IDLE_TIMEOUT:?}");
        };
        match message {
            Some(Ok(Message::Text(text))) => {
                let envelope: Value = match serde_json::from_str(&text) {
                    Ok(envelope) => envelope,
                    Err(error) => {
                        tracing::warn!(
                            error = format!("{error:#?}"),
                            "skipped a malformed slack envelope"
                        );
                        continue;
                    }
                };
                match envelope.get("type").and_then(Value::as_str) {
                    Some("hello") => {
                        greeted = true;
                        backoff.greeted();
                    }
                    Some("disconnect") => {
                        match envelope.get("reason").and_then(Value::as_str) {
                            // Slack closes about 10 s later; events still arrive until then.
                            Some("warning") => tracing::info!("slack will refresh the socket soon"),
                            Some("refresh_requested") => return Ok(Ending::Refresh),
                            Some("link_disabled") => return Ok(Ending::LinkDisabled),
                            reason => eros::bail!("slack disconnected the socket: {reason:?}"),
                        }
                    }
                    Some("events_api") => {
                        // Before the ack: shutdown can cancel the send after Slack has the ack,
                        // and Slack does not resend an acked event.
                        if let Some(event) = envelope.pointer("/payload/event").cloned() {
                            on_event(event);
                        }
                        if let Some(id) = envelope.get("envelope_id").and_then(Value::as_str) {
                            let ack = json!({ "envelope_id": id }).to_string();
                            connection.send(Message::Text(ack.into())).await?;
                        }
                    }
                    Some(other) => tracing::debug!(ty = other, "ignored slack envelope"),
                    None => {}
                }
            }
            Some(Ok(Message::Close(_))) | None if greeted => return Ok(Ending::Refresh),
            Some(Ok(Message::Close(_))) | None => {
                eros::bail!("slack closed the websocket before hello")
            }
            Some(Err(error)) => return Err(error.into()),
            Some(Ok(_)) => {}
        }
    }
}
