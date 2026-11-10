//! OpenAI-compatible HTTP API (`drew serve`): `/v1/models`, `/v1/chat/completions`, `/v1/runs/{id}`. See
//! `docs/design.md` section 4.
//!
//! This module owns HTTP: authentication, wire types, error bodies and markdown rendering. It answers through the
//! [`Answerer`] port and persists every run through [`RunStore`] before responding. `drew serve` also starts the
//! Slack bot ([`crate::slack`]) when it is configured.

pub mod auth;
pub mod error;
pub mod openai;
pub mod render;

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;

use std::{convert::Infallible, future::IntoFuture, sync::Arc};

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use eros::Context;
use serde::Serialize;
use uuid::Uuid;

use crate::{
    domain::{RunId, UserId},
    pipeline::{
        answerer::{PipelineAnswerer, PipelineArgs},
        report::{Answerer, Outcome, RunReport},
    },
    store::{self, DbArgs, RunStore, StoredRun, runs::PgRunStore},
};

use self::{
    auth::{AdminUsers, ApiKeys, User},
    error::ApiError,
    openai::{ChatRequest, Chunk, Completion, Drew, ModelList},
};

/// `drew serve` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Database connection.
    #[command(flatten)]
    pub db: DbArgs,
    /// Socket address to listen on.
    #[arg(long, env = "DREW_BIND", default_value = "0.0.0.0:8080")]
    pub bind: String,
    /// Per-user bearer keys: `key:user_id` pairs, comma-separated.
    #[arg(long, env = "DREW_API_KEYS", hide_env_values = true)]
    pub api_keys: ApiKeys,
    /// User ids that may read any persisted run, comma-separated.
    #[arg(long, env = "DREW_ADMIN_USERS", default_value = "")]
    pub admin_users: AdminUsers,
    /// Models and tunables.
    #[command(flatten)]
    pub pipeline: PipelineArgs,
    /// The Slack bot; off unless both tokens are set.
    #[command(flatten)]
    pub slack: crate::slack::SlackArgs,
}

/// What every handler shares; the Slack bot uses the same answerer and run store.
#[derive(Clone)]
pub struct AppState {
    /// Runs the pipeline.
    pub answerer: Arc<dyn Answerer>,
    /// Persists every run before it is returned.
    pub runs: Arc<dyn RunStore>,
    /// Bearer keys and the users they map to.
    pub keys: Arc<ApiKeys>,
    /// Users who may read any run.
    pub admins: Arc<AdminUsers>,
}

/// The API routes, each behind a bearer key. An unknown path gets an OpenAI-style 404 without authentication.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/runs/{id}", get(get_run))
        .fallback(unknown_endpoint)
        .with_state(state)
}

/// Runs `drew serve` until SIGINT or SIGTERM, then shuts the HTTP server and the Slack bot down gracefully.
///
/// # Errors
///
/// When the database, pipeline or Slack bot cannot be set up, the address cannot be bound, or the server fails.
pub async fn run(args: Args) -> eros::Result<()> {
    let pool = store::connect(&args.db).await?;
    let answerer = PipelineAnswerer::from_args(pool.clone(), &args.pipeline).await?;
    let state = AppState {
        answerer: Arc::new(answerer),
        runs: Arc::new(PgRunStore::new(pool.clone())),
        keys: Arc::new(args.api_keys),
        admins: Arc::new(args.admin_users),
    };
    let bot = crate::slack::Bot::from_args(&args.slack)?;
    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    tracing::info!(addr = %listener.local_addr()?, "drew serve listening");
    let (stop, shutdown) = tokio::sync::watch::channel(false);
    let serve = axum::serve(listener, router(state.clone())).with_graceful_shutdown(async move {
        shutdown_signal().await;
        // Fails only when the bot is off, so nothing is listening.
        let _ = stop.send(true);
    });
    let slack = async {
        if let Some(bot) = bot {
            bot.run(state, shutdown).await;
        }
    };
    let (served, ()) = tokio::join!(serve.into_future(), slack);
    served?;
    Ok(())
}

/// As PID 1 in a container the process gets no default SIGTERM handler, so it has to exit on SIGTERM itself.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::warn!(%error, "no SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

async fn models(_: User) -> Json<ModelList> {
    Json(ModelList::drew())
}

async fn chat_completions(
    State(state): State<AppState>,
    User(user): User,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = body.map_err(|rejection| ApiError::bad_request(rejection.body_text()))?;
    let question = request.question()?;

    let report = state.answerer.answer(question).await;
    let run_id = report.id;
    if let Err(error) = state.runs.save(&user, &report).await {
        tracing::error!(%run_id, error = format!("{error:#?}"), "run not persisted");
        return Err(ApiError::internal(
            "run_not_persisted",
            format!("Run {run_id} could not be persisted."),
        )
        .with_run(run_id));
    }

    let drew = Drew::new(&report);
    tracing::info!(%run_id, user = user.0, outcome = drew.outcome, "run finished");
    let content = match &report.outcome {
        Outcome::Answered(answer) => render::answer(answer),
        Outcome::Declined { sources, .. } => render::declined(sources),
        Outcome::Failed { at, error } => {
            tracing::error!(%run_id, at, error, "run failed");
            return Err(
                ApiError::internal("run_failed", format!("Run {run_id} failed at {at}."))
                    .with_run(run_id)
                    .with_drew(drew),
            );
        }
    };

    let created = Utc::now().timestamp();
    if request.stream != Some(true) {
        return Ok(Json(Completion::new(drew, content, created)).into_response());
    }
    let events: Vec<Result<Event, Infallible>> = Chunk::pair(drew, content, created)
        .iter()
        .map(|chunk| {
            let data = serde_json::to_string(chunk)
                .expect("a completion chunk is plain data, which always serializes");
            Ok(Event::default().data(data))
        })
        .chain([Ok(Event::default().data("[DONE]"))])
        .collect();
    Ok(Sse::new(futures::stream::iter(events)).into_response())
}

/// A stored run as returned by `GET /v1/runs/{id}`.
#[derive(Serialize)]
struct RunView {
    #[serde(flatten)]
    report: RunReport,
    status: &'static str,
    user: UserId,
    created_at: DateTime<Utc>,
}

impl From<StoredRun> for RunView {
    fn from(run: StoredRun) -> Self {
        Self {
            status: store::runs::status(&run.report.outcome),
            report: run.report,
            user: run.user,
            created_at: run.created_at,
        }
    }
}

async fn get_run(
    State(state): State<AppState>,
    User(user): User,
    Path(id): Path<String>,
) -> Result<Json<RunView>, ApiError> {
    let not_found = || ApiError::not_found("Run not found.");
    let run_id = Uuid::parse_str(&id)
        .map(RunId::from)
        .map_err(|_| not_found())?;
    let run = state.runs.get(run_id).await.map_err(|error| {
        tracing::error!(%run_id, error = format!("{error:#?}"), "run not loaded");
        ApiError::internal("run_store_unavailable", "The run could not be loaded.")
    })?;
    match run {
        Some(run) if run.user == user || state.admins.contains(&user) => Ok(Json(run.into())),
        _ => Err(not_found()),
    }
}

async fn unknown_endpoint() -> ApiError {
    ApiError::not_found("Unknown endpoint.")
}
