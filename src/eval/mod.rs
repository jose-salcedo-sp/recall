//! `drew eval`: golden-set metrics per step, and threshold tuning over persisted runs. See `docs/design.md`
//! section 6.
//!
//! `drew eval run` answers every [`golden`] question through an [`Answerer`], scores each run from its trace
//! ([`metrics`]), and gates against a baseline. `drew eval tune` re-applies candidate thresholds to a results file's
//! recorded scores without calling any model ([`tune`]). `drew eval retrieval` runs only `Retrieve` and reports
//! recall ([`retrieval`]). Eval reads runs; it never writes the database.

pub mod golden;
pub mod metrics;
pub mod progress;
pub mod retrieval;
pub mod tune;

use std::{
    fs,
    io::Write as _,
    path::PathBuf,
    process::{Command as Process, Stdio},
    sync::Arc,
};

use chrono::Utc;

use crate::{
    config::ConfigArgs,
    llm::{self, backend::EmbedArgs},
    pipeline::{
        answerer::{PipelineAnswerer, PipelineArgs},
        report::Answerer,
    },
    retrieval::hybrid::HybridRetriever,
    store::{self, DbArgs},
};

use self::{
    golden::GoldenQuestion,
    metrics::{Baseline, Results, RunSetup},
    retrieval::{RetrievalArgs, RetrievalSetup, run_retrieval},
    tune::Candidate,
};

/// `drew eval` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// `run`, `retrieval` or `tune`.
    #[command(subcommand)]
    pub command: Command,
}

/// The `drew eval` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Answer every golden question and report per-step metrics.
    Run(Box<RunCommand>),
    /// Run only Retrieve on every golden question and report recall, per question against a baseline.
    Retrieval(Box<RetrievalCommand>),
    /// Re-apply candidate thresholds to a results file's recorded scores.
    Tune(TuneArgs),
}

/// `drew eval retrieval` options.
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
}

/// `drew eval run` options.
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

/// What `drew eval run` reads and writes, independent of the pipeline it runs.
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
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Run only these golden question ids (repeatable, or comma-separated). With `--baseline`, the baseline's
    /// metrics are recomputed from its runs of the same questions, so a subset is compared like for like.
    #[arg(long = "question", value_name = "ID", value_delimiter = ',')]
    pub questions: Vec<String>,
}

/// `drew eval tune` options.
#[derive(Debug, clap::Args)]
pub struct TuneArgs {
    /// Results JSON written by `drew eval run`.
    #[arg(long)]
    pub results: PathBuf,
    /// JSON array of candidates: `{"name", "thresholds": {step: min}, "per_category": {category: {step: min}}}`.
    #[arg(long)]
    pub candidates: PathBuf,
    /// Where to write the tuning report JSON.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// Runs `drew eval`.
///
/// # Errors
///
/// As [`run_golden`] or [`run_tune`], or when the database or pipeline cannot be set up.
pub async fn run(args: Args) -> eros::Result<()> {
    match args.command {
        Command::Run(args) => {
            let pool = store::connect(&args.db).await?;
            let answerer = PipelineAnswerer::from_args(pool, &args.pipeline).await?;
            let setup = RunSetup {
                started_at: Utc::now(),
                finished_at: None,
                golden: args.files.golden.display().to_string(),
                git: git_describe(),
                config: answerer.config().clone(),
                models: answerer.models(),
            };
            run_golden(Arc::new(answerer), args.files, Some(setup)).await
        }
        Command::Retrieval(args) => {
            let pool = store::connect(&args.db).await?;
            let config = args.config.load()?;
            let embedder = llm::embedder(&args.embed).await?;
            let setup = RetrievalSetup {
                started_at: Utc::now(),
                finished_at: None,
                golden: args.files.golden.display().to_string(),
                git: git_describe(),
                embedding: embedder.model(),
                retrieval: config.retrieval.clone(),
            };
            let retriever = HybridRetriever::new(pool, embedder, config.retrieval);
            run_retrieval(Arc::new(retriever), args.files, Some(setup)).await
        }
        Command::Tune(args) => run_tune(&args),
    }
}

/// Answers the golden set, writes the results file, and compares it with the baseline when one is given.
/// `setup`, when known, is printed before the first question and recorded in the results file.
///
/// # Errors
///
/// When a file cannot be read, parsed or written, a selected question id is unknown, the baseline cannot be
/// restricted to the selection, or any metric regressed beyond its tolerance.
pub async fn run_golden(
    answerer: Arc<dyn Answerer>,
    args: RunArgs,
    setup: Option<RunSetup>,
) -> eros::Result<()> {
    let golden = select(read_golden(&args.golden)?, &args.questions)?;
    let mut baseline: Option<Baseline> = args.baseline.as_deref().map(read_json).transpose()?;
    if !args.questions.is_empty() {
        baseline = baseline
            .map(|baseline| baseline.restricted_to(&args.questions))
            .transpose()?;
    }

    if let Some(setup) = &setup {
        print_setup(&serde_json::json!({
            "golden": setup.golden,
            "git": setup.git,
            "models": setup.models,
            "config": setup.config,
        }));
    }
    let bar = progress::Bar::new(golden.len(), args.concurrency);
    let mut results = metrics::evaluate(answerer.as_ref(), golden, args.concurrency, &bar).await;
    bar.finish();
    results.setup = setup.map(|setup| RunSetup {
        finished_at: Some(Utc::now()),
        ..setup
    });
    write_json(&args.out, &results)?;
    println!("{}", results.metrics);
    println!("Results written to {}", args.out.display());

    let Some(baseline) = baseline else {
        return Ok(());
    };
    let regressions = metrics::regressions(&baseline, &results.metrics);
    if regressions.is_empty() {
        println!("No regressions against the baseline.");
        return Ok(());
    }
    println!("Regressions against the baseline:");
    for regression in &regressions {
        println!("  {regression}");
    }
    eros::bail!("{} metric(s) regressed beyond tolerance", regressions.len())
}

/// Replays every candidate over the results file, prints the report, and writes it when `out` is set.
///
/// # Errors
///
/// When a file cannot be read, parsed or written.
pub fn run_tune(args: &TuneArgs) -> eros::Result<()> {
    let results: Results = read_json(&args.results)?;
    let candidates: Vec<Candidate> = read_json(&args.candidates)?;
    let report = tune::tune(&results, &candidates);
    println!("{report}");
    if let Some(out) = &args.out {
        write_json(out, &report)?;
        println!("Tuning report written to {}", out.display());
    }
    Ok(())
}

/// Through `jq -C` when it is installed, so the config reads like any other JSON in the terminal.
fn print_setup(setup: &serde_json::Value) {
    let pretty = serde_json::to_string_pretty(setup)
        .expect("the setup is plain data, which always serializes");
    println!("{}", console::style("    Setup").green().bold());
    let colour = if console::colors_enabled() {
        "-C"
    } else {
        "-M"
    };
    let jq = Process::new("jq")
        .args([colour, "."])
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::null())
        .spawn()
        .and_then(|mut jq| {
            if let Some(mut stdin) = jq.stdin.take() {
                stdin.write_all(pretty.as_bytes())?;
            }
            jq.wait()
        });
    if !jq.is_ok_and(|status| status.success()) {
        println!("{pretty}");
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

/// Every golden question when `ids` is empty, else exactly those, in golden-set order.
#[eros::context("selecting golden questions {:?}", ids)]
fn select(golden: Vec<GoldenQuestion>, ids: &[String]) -> eros::Result<Vec<GoldenQuestion>> {
    if ids.is_empty() {
        return Ok(golden);
    }
    let unknown: Vec<&str> = ids
        .iter()
        .filter(|id| !golden.iter().any(|question| question.id == **id))
        .map(String::as_str)
        .collect();
    eros::ensure!(unknown.is_empty(), "no golden question with id {unknown:?}");
    Ok(golden
        .into_iter()
        .filter(|question| ids.contains(&question.id))
        .collect())
}

#[eros::context("reading golden set {}", path.display())]
fn read_golden(path: &std::path::Path) -> eros::Result<Vec<GoldenQuestion>> {
    let text = fs::read_to_string(path)?;
    Ok(golden::parse(&text)?)
}

#[eros::context("reading {}", path.display())]
fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> eros::Result<T> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

#[eros::context("writing {}", path.display())]
fn write_json(path: &std::path::Path, value: &impl serde::Serialize) -> eros::Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    Ok(fs::write(path, json + "\n")?)
}

#[cfg(test)]
mod tests;
