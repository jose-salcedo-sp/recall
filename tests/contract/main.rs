//! One contract suite per port. Each suite is a set of `async fn`s over the port and a macro that turns them into
//! tests; every adapter of the port, its fake included, calls the macro once.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

#[path = "../common/runs.rs"]
mod runs;

mod bedrock;
mod chat_model;
mod chunk_store;
mod embedder;
mod judge;
mod models_check;
mod retriever;
mod run_store;
mod source;
mod system_one;
