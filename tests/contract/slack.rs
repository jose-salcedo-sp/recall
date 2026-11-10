use std::sync::Arc;

use drew::slack::api::SlackApi;
use futures::future::BoxFuture;

const SOCKET_URL: &str = "wss://wss-primary.slack.com/link/?ticket=test";

/// Messages as they reached Slack: posts as (channel, thread_ts, text), updates as (channel, ts, text).
#[derive(Debug, Default, PartialEq)]
pub struct Sent {
    pub posts: Vec<(String, Option<String>, String)>,
    pub updates: Vec<(String, String, String)>,
}

/// An adapter whose Socket Mode URL is `SOCKET_URL`, and the messages it sent.
pub struct Adapter {
    pub slack: Arc<dyn SlackApi>,
    pub sent: Box<dyn Fn() -> BoxFuture<'static, Sent> + Send + Sync>,
}

pub async fn opening_a_connection_returns_the_socket_url(
    make: impl AsyncFnOnce() -> Adapter,
) -> eros::Result<()> {
    let adapter = make().await;

    assert_eq!(adapter.slack.open_connection().await?, SOCKET_URL);
    Ok(())
}

pub async fn a_posted_message_can_be_updated_by_its_ts(
    make: impl AsyncFnOnce() -> Adapter,
) -> eros::Result<()> {
    let adapter = make().await;

    let ts = adapter
        .slack
        .post_message("C1", Some("1700000000.000100"), "Thinking…")
        .await?;
    adapter.slack.update_message("C1", &ts, "Done").await?;

    assert!(!ts.is_empty());
    let sent = (adapter.sent)().await;
    assert_eq!(sent.updates, [("C1".into(), ts, "Done".into())]);
    Ok(())
}

pub async fn messages_reach_slack_as_given(
    make: impl AsyncFnOnce() -> Adapter,
) -> eros::Result<()> {
    let adapter = make().await;

    adapter
        .slack
        .post_message("C1", Some("1700000000.000100"), "In a thread & <escaped>")
        .await?;
    adapter.slack.post_message("D2", None, "A DM").await?;

    let sent = (adapter.sent)().await;
    assert_eq!(
        sent.posts,
        [
            (
                "C1".into(),
                Some("1700000000.000100".into()),
                "In a thread & <escaped>".into()
            ),
            ("D2".into(), None, "A DM".into()),
        ]
    );
    Ok(())
}

macro_rules! slack_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn opening_a_connection_returns_the_socket_url() -> eros::Result<()> {
            crate::slack::opening_a_connection_returns_the_socket_url($make).await
        }

        #[tokio::test]
        async fn a_posted_message_can_be_updated_by_its_ts() -> eros::Result<()> {
            crate::slack::a_posted_message_can_be_updated_by_its_ts($make).await
        }

        #[tokio::test]
        async fn messages_reach_slack_as_given() -> eros::Result<()> {
            crate::slack::messages_reach_slack_as_given($make).await
        }
    };
}

mod fake {
    use std::sync::Arc;

    use drew::slack::fakes::FakeSlack;

    use super::{Adapter, SOCKET_URL, Sent};

    async fn fake() -> Adapter {
        let fake = Arc::new(FakeSlack {
            ws_url: SOCKET_URL.into(),
            ..FakeSlack::default()
        });
        Adapter {
            slack: Arc::<FakeSlack>::clone(&fake),
            sent: Box::new(move || {
                let fake = Arc::clone(&fake);
                Box::pin(async move {
                    Sent {
                        posts: fake.posts.lock().unwrap().clone(),
                        updates: fake.updates.lock().unwrap().clone(),
                    }
                })
            }),
        }
    }

    slack_contract!(fake);
}

mod web_api {
    use std::{collections::HashMap, sync::Arc};

    use drew::slack::api::{SlackApi, SlackClient};
    use reqwest::Url;
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{header, method, path},
    };

    use super::{Adapter, SOCKET_URL, Sent};

    fn client(server: &MockServer) -> SlackClient {
        SlackClient::new("xapp-test".into(), "xoxb-test".into())
            .unwrap()
            .with_base_url(format!("{}/api/", server.uri()))
    }

    fn form(request: &Request) -> HashMap<String, String> {
        let body = String::from_utf8(request.body.clone()).unwrap();
        Url::parse(&format!("http://form.invalid/?{body}"))
            .unwrap()
            .query_pairs()
            .into_owned()
            .collect()
    }

    async fn reply(server: &MockServer, api_method: &str, token: &str, body: serde_json::Value) {
        Mock::given(method("POST"))
            .and(path(format!("/api/{api_method}")))
            .and(header("authorization", format!("Bearer {token}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(server)
            .await;
    }

    async fn slack() -> Adapter {
        let server = MockServer::start().await;
        reply(
            &server,
            "apps.connections.open",
            "xapp-test",
            json!({ "ok": true, "url": SOCKET_URL }),
        )
        .await;
        reply(
            &server,
            "chat.postMessage",
            "xoxb-test",
            json!({ "ok": true, "channel": "C1", "ts": "1700000001.000200" }),
        )
        .await;
        reply(
            &server,
            "chat.update",
            "xoxb-test",
            json!({ "ok": true, "channel": "C1", "ts": "1700000001.000200" }),
        )
        .await;
        let slack = Arc::new(client(&server));
        let server = Arc::new(server);
        Adapter {
            slack,
            sent: Box::new(move || {
                let server = Arc::clone(&server);
                Box::pin(async move {
                    let mut sent = Sent::default();
                    for request in server.received_requests().await.unwrap() {
                        let mut form = form(&request);
                        let mut take = |key: &str| form.remove(key);
                        match request.url.path() {
                            "/api/chat.postMessage" => sent.posts.push((
                                take("channel").unwrap(),
                                take("thread_ts"),
                                take("text").unwrap(),
                            )),
                            "/api/chat.update" => sent.updates.push((
                                take("channel").unwrap(),
                                take("ts").unwrap(),
                                take("text").unwrap(),
                            )),
                            _ => {}
                        }
                    }
                    sent
                })
            }),
        }
    }

    slack_contract!(slack);

    async fn responding(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn ok_false_names_the_method_and_slacks_error() {
        let server = responding(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "ok": false, "error": "channel_not_found" })),
        )
        .await;
        let slack = client(&server);

        let errors = [
            slack.open_connection().await.unwrap_err(),
            slack.post_message("C1", None, "hi").await.unwrap_err(),
            slack.update_message("C1", "1.0", "hi").await.unwrap_err(),
        ];

        for (error, api_method) in
            errors
                .iter()
                .zip(["apps.connections.open", "chat.postMessage", "chat.update"])
        {
            let report = format!("{error:#?}");
            assert!(
                report.contains(api_method) && report.contains("channel_not_found"),
                "{report}"
            );
        }
    }

    #[tokio::test]
    async fn an_error_status_with_a_non_json_body_keeps_the_status() {
        let server = responding(
            ResponseTemplate::new(503).set_body_string("<html>Service Unavailable</html>"),
        )
        .await;

        let error = client(&server)
            .post_message("C1", None, "hi")
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains("chat.postMessage") && report.contains("503"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn a_reply_without_its_value_is_an_error() {
        let server =
            responding(ResponseTemplate::new(200).set_body_json(json!({ "ok": true }))).await;
        let slack = client(&server);

        assert!(slack.open_connection().await.is_err());
        assert!(slack.post_message("C1", None, "hi").await.is_err());
    }
}
