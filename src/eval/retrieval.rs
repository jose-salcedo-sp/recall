//! `drew eval retrieval`: runs only [`Retrieve`] (embedder and database; no System One, no LLM) on every golden
//! question, reports recall over the retrieved matches, and compares each question with a baseline.

use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use chrono::{DateTime, Utc};
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use crate::{
    domain::CategoryKind,
    llm::ModelRef,
    pipeline::{
        Asked, Step, StepErrorKind,
        report::RunReport,
        steps::Retrieve,
        trace::{RetrieveRecord, TraceRecord},
    },
    retrieval::{Retriever, hybrid::HybridConfig},
};

use super::{
    golden::GoldenQuestion,
    metrics::{self, CATEGORIES, Metric, RECALL_K, cell, percent, scope_name},
};

/// Retrieved matches that count for recall@10.
pub const SHORT_RECALL_K: usize = 10;

/// What `drew eval retrieval` reads and writes, independent of the retriever it runs.
#[derive(Debug, clap::Args)]
pub struct RetrievalArgs {
    /// Golden set, JSONL.
    #[arg(long)]
    pub golden: PathBuf,
    /// Where to write the results JSON.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// A `drew eval run` or `drew eval retrieval` results file to compare each question with; any question that
    /// loses recall exits non-zero.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Questions retrieved at once.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Run only these golden question ids (repeatable, or comma-separated).
    #[arg(long = "question", value_name = "ID", value_delimiter = ',')]
    pub questions: Vec<String>,
}

/// The `drew eval retrieval` output file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResults {
    /// What the runs ran with; `None` for retrievers built outside the command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<RetrievalSetup>,
    /// Computed from `runs`.
    pub metrics: RetrievalReport,
    /// The comparison with `--baseline`, when given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Comparison>,
    /// One per golden question, in golden-set order.
    pub runs: Vec<RetrievalRun>,
}

/// The retriever behind a retrieval results file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalSetup {
    /// When the first question was sent.
    pub started_at: DateTime<Utc>,
    /// When the last retrieval finished; `None` until then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    /// Path of the golden set, as given.
    pub golden: String,
    /// `git describe --always --dirty` of the working tree, when it is a git checkout.
    pub git: Option<String>,
    /// The query embedder.
    pub embedding: ModelRef,
    /// The hybrid search tunables.
    pub retrieval: HybridConfig,
}

/// One golden question and what `Retrieve` returned for it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalRun {
    /// The question and its relevant chunks.
    pub golden: GoldenQuestion,
    /// Derived from `golden` and `outcome`.
    pub scores: RecallScores,
    /// Wall-clock time of the retrieval, in milliseconds.
    pub duration_ms: u64,
    /// The retrieved chunks, or the error.
    pub outcome: RetrievalOutcome,
}

/// How one retrieval ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RetrievalOutcome {
    /// `Retrieve`'s record; empty when it found nothing.
    Retrieved(RetrieveRecord),
    /// The retriever errored.
    Failed {
        /// The error rendered with `{:#?}`.
        error: String,
    },
}

/// One question's recall. `None` when the golden question lists no relevant chunk or retrieval failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallScores {
    /// Fraction of the relevant chunks among the first [`SHORT_RECALL_K`] matches.
    pub recall_at_10: Option<f64>,
    /// Fraction of the relevant chunks among the first [`RECALL_K`] matches.
    pub recall_at_50: Option<f64>,
}

impl RecallScores {
    /// Scores `record` against `golden`'s relevant chunks; context chunks do not count.
    pub fn of(golden: &GoldenQuestion, record: Option<&RetrieveRecord>) -> Self {
        let at =
            |k| record.and_then(|record| metrics::recall_at(k, &golden.relevant_chunks, record));
        Self {
            recall_at_10: at(SHORT_RECALL_K),
            recall_at_50: at(RECALL_K),
        }
    }

    /// Better on recall@50, then on recall@10, is a gain; a question that lost its score is a loss.
    fn change_from(self, baseline: Self) -> Change {
        let current = (self.recall_at_50, self.recall_at_10);
        let before = (baseline.recall_at_50, baseline.recall_at_10);
        match current.partial_cmp(&before) {
            Some(Ordering::Greater) => Change::Gained,
            Some(Ordering::Less) => Change::Lost,
            _ => Change::Unchanged,
        }
    }
}

impl RetrievalOutcome {
    /// The record recall is computed from; `None` when retrieval failed.
    pub fn record(&self) -> Option<&RetrieveRecord> {
        match self {
            Self::Retrieved(record) => Some(record),
            Self::Failed { .. } => None,
        }
    }
}

/// Recall overall and per golden category.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalReport {
    /// Over every question.
    pub overall: RetrievalMetrics,
    /// Grouped by the golden category, in [`CATEGORIES`] order.
    pub per_category: Vec<CategoryRetrievalMetrics>,
}

/// The recall of the questions whose golden category is `category`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CategoryRetrievalMetrics {
    /// The golden category.
    pub category: CategoryKind,
    /// Its recall.
    #[serde(flatten)]
    pub metrics: RetrievalMetrics,
}

/// Means of [`RecallScores`] over a set of questions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    /// Questions in the set.
    pub runs: usize,
    /// Mean of [`RecallScores::recall_at_10`].
    pub recall_at_10: Metric,
    /// Mean of [`RecallScores::recall_at_50`].
    pub recall_at_50: Metric,
}

impl RetrievalReport {
    /// Groups the scores by golden category.
    pub fn of(scores: &[(CategoryKind, RecallScores)]) -> Self {
        let of = |category: Option<CategoryKind>| {
            let scores = scores
                .iter()
                .filter(move |(c, _)| category.is_none_or(|category| *c == category))
                .map(|(_, scores)| scores);
            RetrievalMetrics {
                runs: scores.clone().count(),
                recall_at_10: Metric::mean(scores.clone().filter_map(|s| s.recall_at_10)),
                recall_at_50: Metric::mean(scores.filter_map(|s| s.recall_at_50)),
            }
        };
        Self {
            overall: of(None),
            per_category: CATEGORIES
                .into_iter()
                .map(|category| CategoryRetrievalMetrics {
                    category,
                    metrics: of(Some(category)),
                })
                .collect(),
        }
    }

    fn rows(&self) -> impl Iterator<Item = (Option<CategoryKind>, &RetrievalMetrics)> {
        std::iter::once((None, &self.overall)).chain(
            self.per_category
                .iter()
                .map(|entry| (Some(entry.category), &entry.metrics)),
        )
    }
}

/// The same questions scored on the baseline's retrieved chunks, against the current golden set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    /// Path of the baseline file, as given.
    pub file: String,
    /// The baseline's recall on the same questions.
    pub metrics: RetrievalReport,
    /// One per question, in golden-set order.
    pub questions: Vec<QuestionChange>,
}

/// One question's recall in the baseline and now.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuestionChange {
    /// The golden question id.
    pub id: String,
    /// Recall on the baseline's chunks.
    pub baseline: RecallScores,
    /// Recall now.
    pub current: RecallScores,
    /// Derived from the two.
    pub change: Change,
}

/// How a question's recall moved against the baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Change {
    /// Higher recall@50, or equal recall@50 and higher recall@10.
    Gained,
    /// Lower, by the same order.
    Lost,
    /// Same recall@50 and recall@10.
    Unchanged,
}

impl Comparison {
    fn ids(&self, change: Change) -> Vec<&str> {
        self.questions
            .iter()
            .filter(|question| question.change == change)
            .map(|question| question.id.as_str())
            .collect()
    }
}

/// Runs `Retrieve` on every golden question, at most `concurrency` at a time, and scores the results.
pub async fn evaluate(
    retriever: Arc<dyn Retriever>,
    golden: Vec<GoldenQuestion>,
    concurrency: usize,
) -> Vec<RetrievalRun> {
    let retrieve = Retrieve::new(retriever);
    let retrieve = &retrieve;
    let mut runs: Vec<(usize, RetrievalRun)> = stream::iter(golden.into_iter().enumerate())
        .map(|(index, golden)| async move { (index, retrieve_one(retrieve, golden).await) })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    runs.sort_by_key(|(index, _)| *index);
    runs.into_iter().map(|(_, run)| run).collect()
}

async fn retrieve_one(retrieve: &Retrieve, golden: GoldenQuestion) -> RetrievalRun {
    let start = Instant::now();
    let outcome = match retrieve.run(&golden.question, Asked).await {
        Ok(scored) => RetrievalOutcome::Retrieved(scored.record),
        Err(error) => match (error.kind, error.record) {
            (_, Some(TraceRecord::Retrieve(record))) => RetrievalOutcome::Retrieved(record),
            (StepErrorKind::Failed(error), _) => RetrievalOutcome::Failed {
                error: format!("{error:#?}"),
            },
            (StepErrorKind::Decline(reason), _) => RetrievalOutcome::Failed {
                error: format!("declined without a record: {reason:?}"),
            },
        },
    };
    RetrievalRun {
        scores: RecallScores::of(&golden, outcome.record()),
        golden,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        outcome,
    }
}

/// Compares `runs` with the baseline's retrieved chunks for the same questions.
///
/// # Errors
///
/// When the baseline did not run every question in `runs`.
#[eros::context("comparing with baseline {}", file)]
pub fn compare(
    runs: &[RetrievalRun],
    baseline: &BaselineRecords,
    file: &str,
) -> eros::Result<Comparison> {
    let missing: Vec<&str> = runs
        .iter()
        .map(|run| run.golden.id.as_str())
        .filter(|id| !baseline.contains(id))
        .collect();
    eros::ensure!(missing.is_empty(), "the baseline did not run {missing:?}");
    let questions: Vec<QuestionChange> = runs
        .iter()
        .map(|run| {
            let before = RecallScores::of(&run.golden, baseline.record(&run.golden.id));
            QuestionChange {
                id: run.golden.id.clone(),
                baseline: before,
                current: run.scores,
                change: run.scores.change_from(before),
            }
        })
        .collect();
    let scores: Vec<(CategoryKind, RecallScores)> = runs
        .iter()
        .zip(&questions)
        .map(|(run, question)| (run.golden.expected.category(), question.baseline))
        .collect();
    Ok(Comparison {
        file: file.to_owned(),
        metrics: RetrievalReport::of(&scores),
        questions,
    })
}

/// Each baseline question's `Retrieve` record by golden id; `None` when its retrieval failed or left no record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BaselineRecords(HashMap<String, Option<RetrieveRecord>>);

/// A run of either results file: `drew eval run` records a `report`, `drew eval retrieval` an `outcome`.
#[derive(Deserialize)]
struct BaselineRun {
    golden: BaselineGolden,
    #[serde(default)]
    report: Option<RunReport>,
    #[serde(default)]
    outcome: Option<RetrievalOutcome>,
}

#[derive(Deserialize)]
struct BaselineGolden {
    id: String,
}

#[derive(Deserialize)]
struct BaselineFile {
    runs: Vec<BaselineRun>,
}

impl BaselineRecords {
    /// Whether the baseline ran the golden question `id`.
    pub fn contains(&self, id: &str) -> bool {
        self.0.contains_key(id)
    }

    /// The `Retrieve` record of question `id`; `None` when the baseline did not run it or its retrieval failed.
    pub fn record(&self, id: &str) -> Option<&RetrieveRecord> {
        self.0.get(id).and_then(Option::as_ref)
    }

    /// Parses a `drew eval run` or `drew eval retrieval` results file.
    ///
    /// # Errors
    ///
    /// When the JSON is not a results file, or a run has neither a report nor a retrieval outcome.
    pub fn parse(json: &str) -> eros::Result<Self> {
        let file: BaselineFile = serde_json::from_str(json)?;
        let records = file
            .runs
            .into_iter()
            .map(|run| {
                let record = match (run.outcome, run.report) {
                    (Some(RetrievalOutcome::Retrieved(record)), _) => Some(record),
                    (Some(RetrievalOutcome::Failed { .. }), _) => None,
                    (None, Some(report)) => metrics::retrieve_record(&report).cloned(),
                    (None, None) => eros::bail!(
                        "baseline run {} has neither a report nor a retrieval outcome",
                        run.golden.id
                    ),
                };
                Ok((run.golden.id, record))
            })
            .collect::<eros::Result<_>>()?;
        Ok(Self(records))
    }
}

/// Retrieves the golden set, prints and writes the results, and fails when a question lost recall against the
/// baseline. `setup`, when known, is printed first and recorded in the results file.
///
/// # Errors
///
/// When a file cannot be read, parsed or written, a selected question id is unknown, the baseline did not run a
/// selected question, or any question lost recall against it.
pub async fn run_retrieval(
    retriever: Arc<dyn Retriever>,
    args: RetrievalArgs,
    setup: Option<RetrievalSetup>,
) -> eros::Result<()> {
    let golden = super::select(super::read_golden(&args.golden)?, &args.questions)?;
    let baseline = match args.baseline.as_deref() {
        Some(path) => Some((path, read_baseline(path)?)),
        None => None,
    };

    if let Some(setup) = &setup {
        super::print_setup(&serde_json::json!({
            "golden": setup.golden,
            "git": setup.git,
            "embedding": setup.embedding,
            "retrieval": setup.retrieval,
        }));
    }
    let start = Instant::now();
    let runs = evaluate(retriever, golden, args.concurrency).await;
    let elapsed = start.elapsed();
    let comparison = baseline
        .map(|(path, records)| compare(&runs, &records, &path.display().to_string()))
        .transpose()?;
    let scores: Vec<(CategoryKind, RecallScores)> = runs
        .iter()
        .map(|run| (run.golden.expected.category(), run.scores))
        .collect();
    let results = RetrievalResults {
        setup: setup.map(|setup| RetrievalSetup {
            finished_at: Some(Utc::now()),
            ..setup
        }),
        metrics: RetrievalReport::of(&scores),
        baseline: comparison,
        runs,
    };

    println!("{results}");
    println!(
        "Retrieved {} questions in {:.1}s",
        results.runs.len(),
        elapsed.as_secs_f64()
    );
    if let Some(out) = &args.out {
        super::write_json(out, &results)?;
        println!("Results written to {}", out.display());
    }

    let lost = results
        .baseline
        .as_ref()
        .map(|comparison| comparison.ids(Change::Lost))
        .unwrap_or_default();
    eros::ensure!(
        lost.is_empty(),
        "{} question(s) lost recall against the baseline: {}",
        lost.len(),
        lost.join(", ")
    );
    Ok(())
}

#[eros::context("reading baseline {}", path.display())]
fn read_baseline(path: &Path) -> eros::Result<BaselineRecords> {
    BaselineRecords::parse(&std::fs::read_to_string(path)?)
}

fn score(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{:.0}%", value * 100.0))
}

fn change_label(change: Change) -> &'static str {
    match change {
        Change::Gained => "gained",
        Change::Lost => "LOST",
        Change::Unchanged => "",
    }
}

impl fmt::Display for RetrievalResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let changes: HashMap<&str, &QuestionChange> = self
            .baseline
            .iter()
            .flat_map(|comparison| &comparison.questions)
            .map(|question| (question.id.as_str(), question))
            .collect();
        let width = self
            .runs
            .iter()
            .map(|run| run.golden.id.len())
            .chain(["question".len()])
            .max()
            .unwrap_or_default();

        write!(
            f,
            "{:<width$} {:<8} {:>9} {:>9}",
            "question", "category", "recall@10", "recall@50"
        )?;
        if self.baseline.is_some() {
            write!(f, " {:>11} {:>11}  change", "baseline@10", "baseline@50")?;
        }
        writeln!(f)?;
        for run in &self.runs {
            write!(
                f,
                "{:<width$} {:<8} {:>9} {:>9}",
                run.golden.id,
                scope_name(Some(run.golden.expected.category())),
                score(run.scores.recall_at_10),
                score(run.scores.recall_at_50),
            )?;
            if let Some(change) = changes.get(run.golden.id.as_str()) {
                write!(
                    f,
                    " {:>11} {:>11}  {}",
                    score(change.baseline.recall_at_10),
                    score(change.baseline.recall_at_50),
                    change_label(change.change),
                )?;
            }
            writeln!(f)?;
            if let RetrievalOutcome::Failed { error } = &run.outcome {
                let first = error.lines().next().unwrap_or_default();
                writeln!(f, "{:>width$} failed: {first}", "")?;
            }
        }
        writeln!(f)?;

        write!(
            f,
            "{:<8} {:>5} {:>14} {:>14}",
            "scope", "runs", "recall@10", "recall@50"
        )?;
        if self.baseline.is_some() {
            write!(f, " {:>14} {:>14}", "baseline@10", "baseline@50")?;
        }
        writeln!(f)?;
        let baseline_rows: Vec<Option<&RetrievalMetrics>> = match &self.baseline {
            Some(comparison) => comparison.metrics.rows().map(|(_, m)| Some(m)).collect(),
            None => vec![None; 1 + CATEGORIES.len()],
        };
        for ((category, metrics), baseline) in self.metrics.rows().zip(baseline_rows) {
            write!(
                f,
                "{:<8} {:>5} {:>14} {:>14}",
                scope_name(category),
                metrics.runs,
                cell(metrics.recall_at_10),
                cell(metrics.recall_at_50),
            )?;
            if let Some(baseline) = baseline {
                write!(
                    f,
                    " {:>14} {:>14}",
                    cell(baseline.recall_at_10),
                    cell(baseline.recall_at_50),
                )?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Each cell is value (runs counted). Recall@{SHORT_RECALL_K} and recall@{RECALL_K} count the matches \
             retrieval ranked, not the context chunks next to them, over questions with relevant chunks."
        )?;
        let failed = self
            .runs
            .iter()
            .filter(|run| run.outcome.record().is_none())
            .count();
        if failed > 0 {
            write!(f, "\n{failed} retrieval(s) failed.")?;
        }
        if let Some(comparison) = &self.baseline {
            let gained = comparison.ids(Change::Gained);
            let lost = comparison.ids(Change::Lost);
            let unchanged = comparison.ids(Change::Unchanged);
            write!(
                f,
                "\nAgainst {} ({} -> {} recall@{RECALL_K}): gained {} {:?}, lost {} {:?}, unchanged {}",
                comparison.file,
                percent(comparison.metrics.overall.recall_at_50.value),
                percent(self.metrics.overall.recall_at_50.value),
                gained.len(),
                gained,
                lost.len(),
                lost,
                unchanged.len(),
            )?;
        }
        Ok(())
    }
}
