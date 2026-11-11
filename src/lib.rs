//! recall: a backend-agnostic provenance system. It answers questions from an ingested corpus so that every claim
//! cites the exact chunk versions it came from, is checked against them before it is shown, and is recorded with the
//! full trace of how it was produced. A [`llm::judge::Judge`] answers the "what" (the question's category, yes or
//! no) with calibrated probabilities, orders matches, admits sources and checks each claim's support; an LLM answers
//! the "why". System One is the shipped judge backend. Every model, store and source sits behind a port, so a domain application supplies
//! its corpus, its answering rules (`explain.instructions`) and its front end as adapters and config. The design is
//! in [`docs/design.md`]; read the section for a module before changing it.
//!
//! One binary serves four commands: `recall ingest` loads the corpus, `recall serve` runs the OpenAI-compatible HTTP
//! API, `recall eval` measures each step against a golden set, and `recall models check` calls each model backend
//! once. The binary is one caller of this library: its clap options and the adapters they select live in `cli`
//! (the `cli` feature, on by default), and nothing else in the crate reads the command line or the environment.
//!
//! # Library interface
//!
//! A domain application builds [`Ports`] from its adapters and parses a [`PipelineConfig`], then
//! [`PipelineAnswerer::assemble`] gives the [`pipeline::report::Answerer`] it asks through. Ingestion is
//! [`ingest::ingest`] over a [`store::chunks::ChunkStore`], an [`llm::Embedder`], an [`access::Labeler`] and its
//! [`ingest::Sources`]. Every run starts at [`pipeline::Run::authorize`], so every trace opens with the policy's
//! decision; eval's runs too.
//!
//! # Pipeline
//!
//! A question becomes a [`pipeline::Run`] whose category and state are types, so a step applies only where its
//! input matches and invalid pipelines do not compile. [`pipeline::Run::then`] times each step, appends its trace
//! entry and checks its threshold; a decline or failure becomes [`pipeline::Halted`], which keeps the trace.
//!
//! ```text
//! Classify ∥ Retrieve → Rerank [→ Admit] → Answer → Explain → Verify    boolean, choice, ranking
//! Classify ∥ Retrieve → Rerank [→ Admit] →          Explain → Verify    open
//! ```
//!
//! The steps are in [`pipeline::steps`]; [`pipeline::answerer::PipelineAnswerer`] drives a question through them
//! and returns a [`pipeline::report::RunReport`]. `Admit`, when configured, screens the sources for injection,
//! relevance, contradiction and evidence. [`pipeline::steps::VerifiedAnswer`] can only be built by `Verify`, which
//! rejects any claim without a citation or citing outside the run's sources and, when configured, any claim whose
//! quoted spans or judged support its cited chunks do not back.
//!
//! # Ports
//!
//! Every external service sits behind a trait, so tests swap in the fakes from `*/fakes.rs` (built with the
//! `testing` feature) and run offline:
//!
//! | Port | Production adapter |
//! |---|---|
//! | [`ingest::Source`] | [`ingest::markdown::Connector`], the markdown directory; a domain passes its own connectors to [`ingest::ingest`] |
//! | [`access::Authorizer`] | [`access::rules::RulePolicy`], roles from `config/access.json`; asked once per run, enforced at every step |
//! | [`access::Labeler`] | [`access::rules::LabelRules`], tightening by source and path prefix; [`access::ConnectorLabels`] keeps the connector's |
//! | [`llm::Embedder`] | Bedrock or an OpenAI-compatible server (Ollama, Workers AI), chosen by `recall`'s `cli` options |
//! | [`llm::ChatModel`] | Bedrock or an OpenAI-compatible server (Ollama, Workers AI), chosen by `recall`'s `cli` options |
//! | [`llm::system_one::SystemOne`] | [`llm::system_one::HttpSystemOne`] (Kev, system-one.dev) or [`llm::workers_ai::WorkersAiSystemOne`], chosen by `recall`'s `cli` options |
//! | [`llm::judge::Judge`] | [`llm::judge::SystemOneJudge`], over whichever System One is configured, every judged step asks through it |
//! | [`retrieval::Retriever`] | [`retrieval::hybrid::HybridRetriever`], full-text and vector search in Postgres |
//! | [`store::RunStore`] | [`store::runs::PgRunStore`] |
//! | [`store::chunks::ChunkStore`] | [`store::chunks::PgChunkStore`]; [`ingest::ingest`] writes through it |
//! | [`clock::Clock`], [`clock::RunIds`] | [`clock::SystemClock`], [`clock::UuidV7`]: trace, run and eval timestamps and run ids |
//! | [`eval::files::Files`] | [`eval::files::LocalFiles`]: golden sets, baselines and results |
//! | [`pipeline::report::Answerer`] | [`pipeline::answerer::PipelineAnswerer`] |
//!
//! # Where to start
//!
//! - [`domain`]: the shared types: questions, chunks, provenance, [`domain::Probability`].
//! - [`access`]: who is asking and what they may read.
//! - [`config`]: every pipeline tunable, loaded from `config/recall.json`.
//! - [`ingest`] and [`store`]: chunking, versioning and storing the corpus.
//! - [`retrieval`], [`llm`] and [`pipeline`]: answering a question.
//! - [`server`]: the OpenAI-compatible HTTP front end.
//! - [`eval`]: golden-set metrics and threshold tuning.
//! - `cli`: the `recall` command line, the environment, and the production adapters it selects.
//!
//! [`docs/design.md`]: https://github.com/jose-salcedo-sp/recall/blob/main/docs/design.md

pub mod access;
#[cfg(feature = "cli")]
pub mod cli;
pub mod clock;
pub mod config;
pub mod domain;
pub mod eval;
pub mod ingest;
pub mod llm;
pub mod pipeline;
pub mod retrieval;
pub mod server;
pub mod store;

pub use config::PipelineConfig;
pub use pipeline::{answerer::PipelineAnswerer, assembly::Ports};
