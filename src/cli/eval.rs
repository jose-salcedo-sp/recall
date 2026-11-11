//! `recall eval`: the golden set against the configured pipeline, retrieval alone against the corpus, and threshold
//! tuning over a results file, on the local file system and the system clock.

use std::{
    num::NonZeroUsize,
    path::PathBuf,
    process::{Command as Process, Stdio},
    sync::Arc,
};

use crate::{
    access::{Principal, rules::RulePolicy},
    clock::{SystemClock, UuidV7},
    domain::UserId,
    eval::{
        EvalPorts, RunOptions, TuneOptions,
        files::LocalFiles,
        metrics::RunSetup,
        retrieval::{RetrievalOptions, RetrievalSetup, run_retrieval},
        run_golden, run_tune,
    },
    retrieval::hybrid::HybridRetriever,
};

use super::{
    args::{AccessArgs, ConfigArgs, DbArgs},
    backend::{self, EmbedArgs},
    pipeline::PipelineArgs,
};

/// `recall eval` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// `run`, `retrieval` or `tune`.
    #[command(subcommand)]
    pub command: Command,
}

/// The `recall eval` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Answer every golden question and report per-step metrics.
    Run(Box<RunCommand>),
    /// Run only Retrieve on every golden question and report recall, per question against a baseline.
    Retrieval(Box<RetrievalCommand>),
    /// Re-apply candidate thresholds to a results file's recorded scores.
    Tune(TuneArgs),
}

/// Who eval asks as. There is no implicit superuser: every question runs with this principal's access, so a golden
/// set measures what that principal can be answered.
#[derive(Clone, Debug, clap::Args)]
pub struct AsArgs {
    /// The user id every golden question is asked as; the access policy decides what it may read.
    #[arg(long = "as", env = "RECALL_EVAL_AS", value_name = "USER")]
    pub user: UserId,
}

impl AsArgs {
    /// The principal, with no asserted roles.
    pub fn principal(&self) -> Principal {
        Principal::new(self.user.clone())
    }
}

/// `recall eval run` options.
#[derive(Debug, clap::Args)]
pub struct RunCommand {
    /// Inputs, outputs and selection.
    #[command(flatten)]
    pub files: RunArgs,
    /// Database connection.
    #[command(flatten)]
    pub db: DbArgs,
    /// Models and tunables.
    #[command(flatten)]
    pub pipeline: PipelineArgs,
}

/// What `recall eval run` reads and writes, independent of the pipeline it runs.
#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Golden set, JSONL.
    #[arg(long)]
    pub golden: PathBuf,
    /// Where to write the results JSON.
    #[arg(long)]
    pub out: PathBuf,
    /// Results JSON (optionally with `tolerances`) to gate against; any regression exits non-zero.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Questions answered at once.
    #[arg(long, default_value = "4")]
    pub concurrency: NonZeroUsize,
    /// Run only these golden question ids (repeatable, or comma-separated). With `--baseline`, the baseline's
    /// metrics are recomputed from its runs of the same questions, so a subset is compared like for like.
    #[arg(long = "question", value_name = "ID", value_delimiter = ',')]
    pub questions: Vec<String>,
    /// Who the questions are asked as.
    #[command(flatten)]
    pub asking: AsArgs,
}

impl From<RunArgs> for RunOptions {
    fn from(args: RunArgs) -> Self {
        Self {
            principal: args.asking.principal(),
            golden: args.golden,
            out: args.out,
            baseline: args.baseline,
            concurrency: args.concurrency,
            questions: args.questions,
        }
    }
}

/// `recall eval retrieval` options.
#[derive(Debug, clap::Args)]
pub struct RetrievalCommand {
    /// Inputs, outputs and selection.
    #[command(flatten)]
    pub files: RetrievalArgs,
    /// Database connection.
    #[command(flatten)]
    pub db: DbArgs,
    /// The tunables file; only `retrieval` is used.
    #[command(flatten)]
    pub config: ConfigArgs,
    /// The query embedder; must match the corpus's model.
    #[command(flatten)]
    pub embed: EmbedArgs,
    /// The access policy retrieval runs under.
    #[command(flatten)]
    pub access: AccessArgs,
}

/// What `recall eval retrieval` reads and writes, independent of the retriever it runs.
#[derive(Debug, clap::Args)]
pub struct RetrievalArgs {
    /// Golden set, JSONL.
    #[arg(long)]
    pub golden: PathBuf,
    /// Where to write the results JSON.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// A `recall eval run` or `recall eval retrieval` results file to compare each question with; any question that
    /// loses recall exits non-zero.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Questions retrieved at once.
    #[arg(long, default_value = "4")]
    pub concurrency: NonZeroUsize,
    /// Run only these golden question ids (repeatable, or comma-separated).
    #[arg(long = "question", value_name = "ID", value_delimiter = ',')]
    pub questions: Vec<String>,
    /// Who the questions are asked as.
    #[command(flatten)]
    pub asking: AsArgs,
}

impl From<RetrievalArgs> for RetrievalOptions {
    fn from(args: RetrievalArgs) -> Self {
        Self {
            principal: args.asking.principal(),
            golden: args.golden,
            out: args.out,
            baseline: args.baseline,
            concurrency: args.concurrency,
            questions: args.questions,
        }
    }
}

/// `recall eval tune` options.
#[derive(Debug, clap::Args)]
pub struct TuneArgs {
    /// Results JSON written by `recall eval run`.
    #[arg(long)]
    pub results: PathBuf,
    /// JSON array of candidates: `{"name", "thresholds": {step: min}, "per_category": {category: {step: min}}}`.
    #[arg(long)]
    pub candidates: PathBuf,
    /// Where to write the tuning report JSON.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

impl From<TuneArgs> for TuneOptions {
    fn from(args: TuneArgs) -> Self {
        Self {
            results: args.results,
            candidates: args.candidates,
            out: args.out,
        }
    }
}

/// Runs `recall eval` on the local file system and the system clock.
///
/// # Errors
///
/// As [`run_golden`], [`run_retrieval`] or [`run_tune`], or when the database or pipeline cannot be set up.
pub async fn run(args: Args) -> eros::Result<()> {
    let ports = EvalPorts {
        files: Arc::new(LocalFiles),
        clock: Arc::new(SystemClock),
        ids: Arc::new(UuidV7),
    };
    match args.command {
        Command::Run(args) => {
            let pool = args.db.connect().await?;
            let answerer = args.pipeline.answerer(pool).await?;
            let setup = RunSetup {
                started_at: ports.clock.now(),
                finished_at: None,
                golden: args.files.golden.display().to_string(),
                git: git_describe(),
                config: answerer.config().clone(),
                models: answerer.models(),
                plan: Some(answerer.plan()),
            };
            run_golden(Arc::new(answerer), &ports, args.files.into(), Some(setup)).await
        }
        Command::Retrieval(args) => {
            let pool = args.db.connect().await?;
            let config = args.config.load()?;
            let embedder = backend::embedder(&args.embed).await?;
            let setup = RetrievalSetup {
                started_at: ports.clock.now(),
                finished_at: None,
                golden: args.files.golden.display().to_string(),
                git: git_describe(),
                embedding: embedder.model(),
                retrieval: config.retrieval.clone(),
            };
            let retriever = HybridRetriever::new(pool, embedder, config.retrieval);
            let authorizer = RulePolicy::new(args.access.load()?);
            run_retrieval(
                Arc::new(retriever),
                &authorizer,
                &ports,
                args.files.into(),
                Some(setup),
            )
            .await
        }
        Command::Tune(args) => run_tune(ports.files.as_ref(), &args.into()),
    }
}

fn git_describe() -> Option<String> {
    let output = Process::new("git")
        .args(["describe", "--always", "--dirty"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
