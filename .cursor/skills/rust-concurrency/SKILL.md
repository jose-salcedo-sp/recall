---
name: rust-concurrency
description: >-
  Rust concurrency and async: channels versus Arc<Mutex>, lock poisoning,
  bounded channels, network and idle timeouts, reconnect backoff, blocking I/O,
  and mutex guards across .await. Use when writing Rust threads, channels,
  shared state, network clients, tokio, or other async code.
---

# Rust concurrency

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

Prefer a channel when ownership of the data can move. `Arc<Mutex<T>>` is for state that stays shared. A poisoned `std` lock means another thread panicked while holding it; `expect` with that reason is appropriate. Any other lock failure is a `Result`.

```rust
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));
let handles: Vec<_> = (0..10).map(|_| {
    let counter = Arc::clone(&counter);
    std::thread::spawn(move || {
        let mut num = counter.lock().expect("mutex poisoned");
        *num += 1;
    })
}).collect();

for handle in handles {
    handle.join().expect("worker thread panicked");
}
```

Use a bounded channel when the receiver must apply backpressure. Drop the original sender so the receiver's iterator ends.

```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::sync_channel(16);

for i in 0..5 {
    let tx = tx.clone();
    std::thread::spawn(move || {
        tx.send(format!("message {i}")).expect("receiver disconnected");
    });
}
drop(tx);

for msg in rx {
    println!("{msg}");
}
```

## Async

Use that runtime's sleep, timeout, and mutex. A `std::sync::Mutex` guard does not cross `.await`. Synchronous file and network I/O belongs outside the async fn, on a blocking pool the runtime provides. `std::thread::sleep` blocks the executor; the runtime's sleep awaits.

Every network call has a deadline. Set it on the client where it is built (`reqwest::ClientBuilder::timeout` and `connect_timeout`, an SDK's timeout config), so no call site can forget it. Wrap a call the client does not cover, such as a WebSocket connect, in the runtime's `timeout`. A long-lived connection also has an idle timeout: no frame or ping within the window means the peer is gone, so reconnect. Reconnect backoff resets once a connection has been healthy, so a routine server-side close does not inherit the delay of earlier failures.

The error types below follow [`rust-errors`](../rust-errors/SKILL.md).

```rust
use eros::Context;
use tokio::time::Duration;

fn http_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .build()
}

async fn fetch(client: &reqwest::Client, url: &str) -> eros::Result<String> {
    let response = client.get(url).send().await.context("request failed")?;
    response.text().await.context("failed to read body")
}

async fn fetch_all(client: &reqwest::Client, urls: Vec<String>) -> Vec<eros::Result<String>> {
    let handles: Vec<_> = urls
        .into_iter()
        .map(|url| {
            let client = client.clone();
            tokio::spawn(async move { fetch(&client, &url).await })
        })
        .collect();
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(handle.await.expect("spawned task panicked"));
    }
    results
}
```

## Before finishing

- Async code does not block the runtime, and a `std` mutex guard does not cross `.await`.
- Each network call has a deadline, each long-lived connection has an idle timeout, and reconnect backoff resets after a healthy connection.
