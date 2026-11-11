//! `recall eval`: golden-set metrics per step, and threshold tuning over persisted runs. See `docs/design.md`
//! section 6.
//!
//! [`run_golden`] (`recall eval run`) answers every [`golden`] question through an [`Answerer`], scores each run from
//! its trace ([`metrics`]), and gates against a baseline. [`run_tune`] (`recall eval tune`) re-applies candidate
//! thresholds to a results file's recorded scores without calling any model ([`tune`]).
//! [`retrieval::run_retrieval`] (`recall eval retrieval`) authorizes each question and runs only `Retrieve`, then
//! reports recall. Eval reads runs; it never writes the database.

pub mod files;
pub mod golden;
pub mod metrics;
pub(crate) mod progress;
pub mod retrieval;
pub mod tune;

use std::{
    io::Write as _,
    num::NonZeroUsize,
    path::PathBuf,
    process::{Command as Process, Stdio},
    sync::Arc,
};

use crate::{
    access::Principal,
    clock::{Clock, RunIds},
    pipeline::report::Answerer,
};

use self::{
    files::Files,
    golden::GoldenQuestion,
    metrics::{Baseline, Results, RunSetup},
    tune::Candidate,
};

/// What [`run_golden`] reads and writes, independent of the pipeline it runs.
#[derive(Clone, Debug)]
pub struct RunOptions {
    /// Golden set, JSONL.
    pub golden: PathBuf,
    /// Where to write the results JSON.
    pub out: PathBuf,
    /// Results JSON (optionally with `tolerances`) to gate against; any regression fails the run.
    pub baseline: Option<PathBuf>,
    /// Questions answered at once.
    pub concurrency: NonZeroUsize,
    /// Run only these golden question ids; every question when empty. With a baseline, the baseline's metrics are
    /// recomputed from its runs of the same questions, so a subset is compared like for like.
    pub questions: Vec<String>,
    /// Who every question is asked as. There is no implicit superuser: every question runs with this principal's
    /// access, so a golden set measures what that principal can be answered.
    pub principal: Principal,
}

/// What [`run_tune`] reads and writes.
#[derive(Clone, Debug)]
pub struct TuneOptions {
    /// Results JSON written by [`run_golden`].
    pub results: PathBuf,
    /// JSON array of candidates: `{"name", "thresholds": {step: min}, "per_category": {category: {step: min}}}`.
    pub candidates: PathBuf,
    /// Where to write the tuning report JSON.
    pub out: Option<PathBuf>,
}

/// What eval reads and writes its files through, and stamps its results with.
#[derive(Clone)]
pub struct EvalPorts {
    /// Golden sets, baselines and candidates in; results and reports out.
    pub files: Arc<dyn Files>,
    /// Stamps `started_at` and `finished_at`, and every trace entry of the runs eval starts itself.
    pub clock: Arc<dyn Clock>,
    /// Names the runs eval starts itself, such as `recall eval retrieval`'s.
    pub ids: Arc<dyn RunIds>,
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
    ports: &EvalPorts,
    args: RunOptions,
    setup: Option<RunSetup>,
) -> eros::Result<()> {
    let files = ports.files.as_ref();
    let golden = select(read_golden(files, &args.golden)?, &args.questions)?;
    let mut baseline: Option<Baseline> = args
        .baseline
        .as_deref()
        .map(|path| read_json(files, path))
        .transpose()?;
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
            "plan": setup.plan,
            "config": setup.config,
        }));
    }
    let bar = progress::Bar::new(golden.len(), args.concurrency.get());
    let mut results = metrics::evaluate(
        answerer.as_ref(),
        &args.principal,
        golden,
        args.concurrency.get(),
        &bar,
    )
    .await;
    bar.finish();
    results.setup = setup.map(|setup| RunSetup {
        finished_at: Some(ports.clock.now()),
        ..setup
    });
    write_json(files, &args.out, &results)?;
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
pub fn run_tune(files: &dyn Files, args: &TuneOptions) -> eros::Result<()> {
    let results: Results = read_json(files, &args.results)?;
    let candidates: Vec<Candidate> = read_json(files, &args.candidates)?;
    let report = tune::tune(&results, &candidates);
    println!("{report}");
    if let Some(out) = &args.out {
        write_json(files, out, &report)?;
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
fn read_golden(files: &dyn Files, path: &std::path::Path) -> eros::Result<Vec<GoldenQuestion>> {
    let text = files.read(path)?;
    Ok(golden::parse(&text)?)
}

#[eros::context("parsing {}", path.display())]
fn read_json<T: serde::de::DeserializeOwned>(
    files: &dyn Files,
    path: &std::path::Path,
) -> eros::Result<T> {
    let text = files.read(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_json(
    files: &dyn Files,
    path: &std::path::Path,
    value: &impl serde::Serialize,
) -> eros::Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    files.write(path, &(json + "\n"))
}

#[cfg(test)]
mod tests;
