//! recall: a backend-agnostic provenance system. It answers questions from an ingested corpus so that every claim
//! cites the exact chunk versions it came from, is checked against them before it is shown, and is recorded with the
//! full trace of how it was produced. A decision model ("System One") answers the "what" (yes or no, a choice, a
//! ranking) with calibrated probabilities; an LLM answers the "why"; a [`llm::judge::Judge`] admits sources and
//! checks each claim's support. Every model, store and source sits behind a port, so a domain application supplies
//! its corpus, its answering rules (`explain.instructions`) and its front end as adapters and config. The design is
//! in [`docs/design.md`]; read the section for a module before changing it.
//!
//! One binary serves four commands: `recall ingest` loads the corpus, `recall serve` runs the OpenAI-compatible HTTP
//! API, `recall eval` measures each step against a golden set, and `recall models check` calls each model backend
//! once.
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
//! | [`ingest::Source`] | [`ingest::markdown::Connector`], the markdown directory; a domain passes its own connectors to [`ingest::run_with`] |
//! | [`llm::Embedder`] | Bedrock or an OpenAI-compatible server (Ollama, Workers AI), chosen by [`llm::embedder`] |
//! | [`llm::ChatModel`] | Bedrock or an OpenAI-compatible server (Ollama, Workers AI), chosen by [`llm::explain_model`] |
//! | [`llm::system_one::SystemOne`] | [`llm::system_one::HttpSystemOne`] (Kev, system-one.dev) or [`llm::workers_ai::WorkersAiSystemOne`], chosen by [`llm::system_one_client`] |
//! | [`llm::judge::Judge`] | [`llm::judge::SystemOneJudge`], over whichever System One is configured |
//! | [`retrieval::Retriever`] | [`retrieval::hybrid::HybridRetriever`], full-text and vector search in Postgres |
//! | [`store::RunStore`] | [`store::runs::PgRunStore`] |
//! | [`pipeline::report::Answerer`] | [`pipeline::answerer::PipelineAnswerer`] |
//!
//! # Where to start
//!
//! - [`domain`]: the shared types: questions, chunks, provenance, [`domain::Probability`].
//! - [`config`]: every pipeline tunable, loaded from `config/recall.json`.
//! - [`ingest`] and [`store`]: chunking, versioning and storing the corpus.
//! - [`retrieval`], [`llm`] and [`pipeline`]: answering a question.
//! - [`server`]: the OpenAI-compatible HTTP front end.
//! - [`eval`]: golden-set metrics and threshold tuning.
//!
//! [`docs/design.md`]: https://github.com/jose-salcedo-sp/recall/blob/main/docs/design.md

pub mod config;
pub mod domain;
pub mod eval;
pub mod ingest;
pub mod llm;
pub mod models;
pub mod pipeline;
pub mod retrieval;
pub mod server;
pub mod store;
