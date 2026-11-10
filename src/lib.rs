//! Drew AI answers questions about SAP and ARM internals from Drew Steinfatt's pre-audited markdown documentation.
//! A small System One model answers the "what" (yes or no, a choice, a ranking) with calibrated probabilities; an
//! LLM answers the "why". Every claim cites its source, and every run, answered, declined or failed, is persisted
//! with its full trace. The design is in [`docs/design.md`]; read the section for a module before changing it.
//!
//! One binary serves four commands: `drew ingest` loads the corpus, `drew serve` runs the OpenAI-compatible HTTP
//! API and the Slack bot, `drew eval` measures each step against a golden set, and `drew models check` calls each
//! model backend once.
//!
//! # Pipeline
//!
//! A question becomes a [`pipeline::Run`] whose category and state are types, so a step applies only where its
//! input matches and invalid pipelines do not compile. [`pipeline::Run::then`] times each step, appends its trace
//! entry and checks its threshold; a decline or failure becomes [`pipeline::Halted`], which keeps the trace.
//!
//! ```text
//! Classify ∥ Retrieve → Rerank → Answer → Explain → Verify    boolean, choice, ranking
//! Classify ∥ Retrieve → Rerank →          Explain → Verify    open
//! ```
//!
//! The steps are in [`pipeline::steps`]; [`pipeline::answerer::PipelineAnswerer`] drives a question through them
//! and returns a [`pipeline::report::RunReport`]. [`pipeline::steps::VerifiedAnswer`] can only be built by
//! `Verify`, which rejects any claim without a citation and any citation outside the run's sources.
//!
//! # Ports
//!
//! Every external service sits behind a trait, so tests swap in the fakes from `*/fakes.rs` (built with the
//! `testing` feature) and run offline:
//!
//! | Port | Production adapter |
//! |---|---|
//! | [`ingest::Source`] | [`ingest::markdown::Connector`], the markdown directory, the only corpus source |
//! | [`llm::Embedder`] | Bedrock or an OpenAI-compatible server (Ollama, Workers AI), chosen by [`llm::embedder`] |
//! | [`llm::ChatModel`] | Bedrock or an OpenAI-compatible server (Ollama, Workers AI), chosen by [`llm::explain_model`] |
//! | [`llm::system_one::SystemOne`] | [`llm::system_one::HttpSystemOne`] (Kev, system-one.dev) or [`llm::workers_ai::WorkersAiSystemOne`], chosen by [`llm::system_one_client`] |
//! | [`retrieval::Retriever`] | [`retrieval::hybrid::HybridRetriever`], full-text and vector search in Postgres |
//! | [`store::RunStore`] | [`store::runs::PgRunStore`] |
//! | [`pipeline::report::Answerer`] | [`pipeline::answerer::PipelineAnswerer`] |
//! | [`slack::api::SlackApi`] | [`slack::api::SlackClient`] |
//!
//! # Where to start
//!
//! - [`domain`]: the shared types: questions, chunks, provenance, [`domain::Probability`].
//! - [`config`]: every pipeline tunable, loaded from `config/drew.json`.
//! - [`ingest`] and [`store`]: chunking, versioning and storing the corpus.
//! - [`retrieval`], [`llm`] and [`pipeline`]: answering a question.
//! - [`server`] and [`slack`]: the two front ends.
//! - [`eval`]: golden-set metrics and threshold tuning.
//!
//! [`docs/design.md`]: https://github.com/jose-salcedo-sp/drew-ai/blob/master/docs/design.md

pub mod config;
pub mod domain;
pub mod eval;
pub mod ingest;
pub mod llm;
pub mod models;
pub mod pipeline;
pub mod retrieval;
pub mod server;
pub mod slack;
pub mod store;
