use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use tokio::{io::DuplexStream, time::Instant};
use tokio_tungstenite::{WebSocketStream, tungstenite::protocol::Role};

use super::{
    api::SlackApi,
    socket::{Connection, Connector},
};

/// Records every post and update and hands out a fixed WebSocket URL.
#[derive(Default)]
pub struct FakeSlack {
    /// `(channel, thread_ts, text)` per post, in order.
    pub posts: Mutex<Vec<(String, Option<String>, String)>>,
    /// `(channel, ts, text)` per update, in order.
    pub updates: Mutex<Vec<(String, String, String)>>,
    /// Returned by every `open_connection`.
    pub ws_url: String,
}

#[async_trait]
impl SlackApi for FakeSlack {
    async fn open_connection(&self) -> eros::Result<String> {
        Ok(self.ws_url.clone())
    }

    async fn post_message(
        &self,
        channel: &str,
        thread_ts: Option<&str>,
        text: &str,
    ) -> eros::Result<String> {
        self.posts
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push((
                channel.to_owned(),
                thread_ts.map(str::to_owned),
                text.to_owned(),
            ));
        Ok(format!(
            "{}.0001",
            self.posts
                .lock()
                .expect("lock poisoned: a test thread panicked while holding it")
                .len()
        ))
    }

    async fn update_message(&self, channel: &str, ts: &str, text: &str) -> eros::Result<()> {
        self.updates
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push((channel.to_owned(), ts.to_owned(), text.to_owned()));
        Ok(())
    }
}

/// Slack's end of an in-memory Socket Mode connection.
pub type SlackEnd = WebSocketStream<DuplexStream>;

/// An in-memory WebSocket: the bot's end as a `Connection`, and Slack's end.
pub async fn connection_pair() -> (Connection, SlackEnd) {
    let (bot, slack) = tokio::io::duplex(64 * 1024);
    let bot = WebSocketStream::from_raw_socket(bot, Role::Client, None).await;
    let slack = WebSocketStream::from_raw_socket(slack, Role::Server, None).await;
    (Box::pin(bot), slack)
}

/// Hands out queued connections in order. With none queued, `connect` waits forever.
#[derive(Default)]
pub struct FakeSocket {
    queued: Mutex<VecDeque<Option<Connection>>>,
    connects: Mutex<Vec<Instant>>,
}

impl FakeSocket {
    /// Queues a connection and returns Slack's end of it.
    pub async fn accept(&self) -> SlackEnd {
        let (bot, slack) = connection_pair().await;
        self.queue(Some(bot));
        slack
    }

    /// Queues a failed connect.
    pub fn refuse(&self) {
        self.queue(None);
    }

    /// When each `connect` call started.
    pub fn connects(&self) -> Vec<Instant> {
        self.connects
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .clone()
    }

    fn queue(&self, connection: Option<Connection>) {
        self.queued
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push_back(connection);
    }
}

#[async_trait]
impl Connector for FakeSocket {
    async fn connect(&self, _url: &str) -> eros::Result<Connection> {
        self.connects
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .push(Instant::now());
        let next = self
            .queued
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .pop_front();
        match next {
            Some(Some(connection)) => Ok(connection),
            Some(None) => eros::bail!("connection refused"),
            None => std::future::pending().await,
        }
    }
}
