//! Per-step metrics over golden-set runs, and the baseline regression gate.

use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use crate::{
    config::PipelineConfig,
    domain::{CategoryKind, ChunkId, ChunkRef, Provenance},
    pipeline::{
        answerer::Models,
        report::{Answer, Answerer, Outcome, RunReport, Verdict},
        trace::{ClassifyRecord, RerankRecord, RetrieveRecord, TraceRecord},
    },
};

use super::golden::GoldenQuestion;

/// Retrieved chunks that count for recall.
pub const RECALL_K: usize = 50;
/// Reranked matches that count for a hit.
pub const HIT_K: usize = 5;

/// The order per-category metrics are reported in.
pub const CATEGORIES: [CategoryKind; 4] = [
    CategoryKind::Boolean,
    CategoryKind::Choice,
    CategoryKind::Ranking,
    CategoryKind::Open,
];

/// The `drew eval run` output file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Results {
    /// What the runs ran with. `None` in files written before it was recorded, and for scripted answerers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<RunSetup>,
    /// Computed from `runs`.
    pub metrics: MetricsReport,
    /// One per golden question, in golden-set order.
    pub runs: Vec<EvalRun>,
}

/// The configuration and models behind a results file, so two files can be compared knowing what changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSetup {
    /// When the first question was sent.
    pub started_at: DateTime<Utc>,
    /// When the last run finished; `None` until then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    /// Path of the golden set, as given.
    pub golden: String,
    /// `git describe --always --dirty` of the working tree, when it is a git checkout.
    pub git: Option<String>,
    /// Every tunable the pipeline ran with.
    #[serde(deserialize_with = "recorded_config")]
    pub config: PipelineConfig,
    /// The models it called.
    pub models: Models,
}

/// Files written before `rerank.judge` existed record a flat `rerank.candidates`, which System One judged.
fn recorded_config<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<PipelineConfig, D::Error> {
    let mut config = serde_json::Value::deserialize(deserializer)?;
    if let Some(rerank) = config
        .get_mut("rerank")
        .and_then(serde_json::Value::as_object_mut)
        && !rerank.contains_key("judge")
        && let Some(candidates) = rerank.remove("candidates")
    {
        rerank.insert(
            "judge".into(),
            serde_json::json!({ "system_one": { "candidates": candidates } }),
        );
    }
    serde_json::from_value(config).map_err(serde::de::Error::custom)
}

/// One golden question, its run, and how the run scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalRun {
    /// The question and its expected answer.
    pub golden: GoldenQuestion,
    /// Derived from `golden` and `report`; kept for reading the file, never read back.
    pub scores: RunScores,
    /// The full run, trace included; tuning re-reads its recorded scores.
    pub report: RunReport,
}

/// One run's contribution to each metric. `None` means the run does not count toward that metric.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunScores {
    /// The run's category equals the golden one; `None` when the run halted before `Classify` recorded one.
    pub classified_correctly: Option<bool>,
    /// Fraction of the golden relevant chunks among the first [`RECALL_K`] retrieved matches (context chunks do not
    /// count), in `0..=1`; `None` when the
    /// golden question lists none or retrieval left no record.
    pub recall_at_50: Option<f64>,
    /// A golden relevant chunk is among the first [`HIT_K`] reranked matches; `None` as for `recall_at_50`.
    pub hit_at_5: Option<bool>,
    /// The verdict matches the expected answer; `None` unless answered with something to grade.
    pub answer_correct: Option<bool>,
    /// The run declined.
    pub declined: bool,
    /// The run failed.
    pub failed: bool,
    /// Every citation is a chunk this run retrieved or reranked; `None` unless answered.
    pub citations_valid: Option<bool>,
}

/// Told about each golden question as its run starts and finishes, in completion order.
pub trait Progress: Sync {
    /// The question was sent to the answerer.
    fn started(&self, golden: &GoldenQuestion);
    /// The run finished and was scored; `elapsed` is its wall-clock time.
    fn finished(&self, run: &EvalRun, elapsed: Duration);
}

/// Reports nothing.
impl Progress for () {
    fn started(&self, _: &GoldenQuestion) {}
    fn finished(&self, _: &EvalRun, _: Duration) {}
}

/// Answers every golden question, at most `concurrency` at a time, and scores the runs.
/// Runs finish in any order; `Results` keeps the golden set's order.
pub async fn evaluate(
    answerer: &dyn Answerer,
    golden: Vec<GoldenQuestion>,
    concurrency: usize,
    progress: &dyn Progress,
) -> Results {
    let mut runs: Vec<(usize, EvalRun)> = stream::iter(golden.into_iter().enumerate())
        .map(|(index, golden)| async move {
            progress.started(&golden);
            let start = Instant::now();
            let report = answerer.answer(golden.question.clone()).await;
            let run = EvalRun::new(golden, report);
            progress.finished(&run, start.elapsed());
            (index, run)
        })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    runs.sort_by_key(|(index, _)| *index);
    Results::new(runs.into_iter().map(|(_, run)| run).collect())
}

impl Results {
    /// Computes the metrics; no setup.
    pub fn new(runs: Vec<EvalRun>) -> Self {
        Self {
            setup: None,
            metrics: MetricsReport::of(&runs),
            runs,
        }
    }
}

impl EvalRun {
    /// Scores `report` against `golden`.
    pub fn new(golden: GoldenQuestion, report: RunReport) -> Self {
        Self {
            scores: RunScores::of(&golden, &report),
            golden,
            report,
        }
    }
}

impl RunScores {
    /// Scores one run from its outcome and the last record of each step in its trace.
    pub fn of(golden: &GoldenQuestion, report: &RunReport) -> Self {
        let expected = golden.expected.category();
        let classify = last_record(report, |record| match record {
            TraceRecord::Classify(record) => Some(record),
            _ => None,
        });
        let retrieve = retrieve_record(report);
        let rerank = last_record(report, |record| match record {
            TraceRecord::Rerank(record) => Some(record),
            _ => None,
        });
        let relevant: HashSet<&ChunkId> = golden.relevant_chunks.iter().collect();

        Self {
            classified_correctly: classified_correctly(report, classify, expected),
            recall_at_50: retrieve
                .filter(|_| !relevant.is_empty())
                .map(|record| recall(RECALL_K, &relevant, record)),
            hit_at_5: rerank
                .filter(|_| !relevant.is_empty())
                .map(|record| hit(&relevant, record)),
            answer_correct: match &report.outcome {
                Outcome::Answered(answer) => golden.expected.matches(answer.verdict.as_ref()),
                _ => None,
            },
            declined: matches!(report.outcome, Outcome::Declined { .. }),
            failed: matches!(report.outcome, Outcome::Failed { .. }),
            citations_valid: match &report.outcome {
                Outcome::Answered(answer) => Some(citations_valid(answer, retrieve, rerank)),
                _ => None,
            },
        }
    }
}

fn last_record<'a, T>(
    report: &'a RunReport,
    pick: impl Fn(&'a TraceRecord) -> Option<&'a T>,
) -> Option<&'a T> {
    report
        .trace
        .entries()
        .iter()
        .rev()
        .find_map(|entry| entry.record.as_ref().and_then(&pick))
}

/// The final category wins (a `choice` can be reclassified as `open`); a run that halted at
/// `Classify` falls back to what the step recorded.
fn classified_correctly(
    report: &RunReport,
    classify: Option<&ClassifyRecord>,
    expected: CategoryKind,
) -> Option<bool> {
    if report.category.is_none() && classify.is_none() {
        return None;
    }
    let predicted = report
        .category
        .or_else(|| classify.and_then(|record| record.category));
    Some(predicted == Some(expected))
}

/// Fraction of `relevant` among the first `k` matches `record` lists; context chunks neither count nor take a
/// place. `None` when `relevant` is empty.
pub fn recall_at(k: usize, relevant: &[ChunkId], record: &RetrieveRecord) -> Option<f64> {
    let relevant: HashSet<&ChunkId> = relevant.iter().collect();
    (!relevant.is_empty()).then(|| recall(k, &relevant, record))
}

fn recall(k: usize, relevant: &HashSet<&ChunkId>, record: &RetrieveRecord) -> f64 {
    let found: HashSet<&ChunkId> = record
        .chunks
        .iter()
        .filter(|ranked| ranked.context_for.is_none())
        .take(k)
        .map(|ranked| &ranked.chunk.id)
        .filter(|id| relevant.contains(id))
        .collect();
    found.len() as f64 / relevant.len() as f64
}

/// The last `Retrieve` record in the run's trace.
pub(super) fn retrieve_record(report: &RunReport) -> Option<&RetrieveRecord> {
    last_record(report, |record| match record {
        TraceRecord::Retrieve(record) => Some(record),
        _ => None,
    })
}

fn hit(relevant: &HashSet<&ChunkId>, record: &RerankRecord) -> bool {
    record
        .chunks
        .iter()
        .take(HIT_K)
        .any(|ranked| relevant.contains(&ranked.chunk.id))
}

/// Retrieved means returned by `Retrieve` or by `Rerank`, which also sees parent expansions.
fn citations_valid(
    answer: &Answer,
    retrieve: Option<&RetrieveRecord>,
    rerank: Option<&RerankRecord>,
) -> bool {
    let retrieved: HashSet<&ChunkRef> = retrieve
        .map(|record| &record.chunks)
        .into_iter()
        .chain(rerank.map(|record| &record.chunks))
        .flatten()
        .map(|ranked| &ranked.chunk)
        .collect();
    let retrieved_ids: HashSet<&ChunkId> = retrieved.iter().map(|chunk| &chunk.id).collect();

    let citations_ok = answer
        .citations
        .iter()
        .all(|citation| retrieved.contains(&citation.chunk));
    let provenance_ok = cited_chunks(answer).all(|id| retrieved_ids.contains(id));
    citations_ok && provenance_ok
}

fn cited_chunks(answer: &Answer) -> impl Iterator<Item = &ChunkId> {
    let verdict_sources: Vec<&[Provenance]> = match &answer.verdict {
        None | Some(Verdict::Boolean { .. }) => vec![],
        Some(Verdict::Choice { selected, rejected }) => std::iter::once(selected)
            .chain(rejected.iter().map(|rejected| &rejected.option))
            .map(|cited| cited.sources.as_slice())
            .collect(),
        Some(Verdict::Ranking { items, .. }) => {
            items.iter().map(|item| item.sources.as_slice()).collect()
        }
    };
    answer
        .why
        .iter()
        .map(|claim| claim.sources.as_slice())
        .chain(verdict_sources)
        .flatten()
        .filter_map(|source| match source {
            Provenance::Chunk(id) => Some(id),
            Provenance::UserPrompt(_) => None,
        })
}

/// Every metric, overall and per golden category.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricsReport {
    /// Over every run.
    pub overall: Metrics,
    /// Grouped by the golden (expected) category, in `CATEGORIES` order.
    pub per_category: Vec<CategoryMetrics>,
}

/// The metrics of the runs whose golden category is `category`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CategoryMetrics {
    /// The golden (expected) category.
    pub category: CategoryKind,
    /// Its metrics.
    #[serde(flatten)]
    pub metrics: Metrics,
}

/// The per-step metrics over a set of runs; each is a mean of the matching [`RunScores`] field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Runs in the set.
    pub runs: usize,
    /// Mean of [`RunScores::classified_correctly`].
    pub classification_accuracy: Metric,
    /// Mean of [`RunScores::recall_at_50`].
    pub retrieval_recall_at_50: Metric,
    /// Mean of [`RunScores::hit_at_5`].
    pub rerank_hit_at_5: Metric,
    /// Mean of [`RunScores::answer_correct`]: correct answers among graded answered runs.
    pub answer_precision: Metric,
    /// Declined runs among all runs.
    pub decline_rate: Metric,
    /// Failed runs among all runs.
    pub failure_rate: Metric,
    /// Mean of [`RunScores::citations_valid`].
    pub citation_validity: Metric,
}

/// A mean over the `n` runs that count toward it; `value` is `None` when `n` is 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    /// The mean; in `0..=1` for every metric this module computes.
    pub value: Option<f64>,
    /// Runs that counted.
    pub n: usize,
}

impl Metric {
    /// The arithmetic mean of `values`.
    pub fn mean(values: impl IntoIterator<Item = f64>) -> Self {
        let (sum, n) = values
            .into_iter()
            .fold((0.0, 0), |(sum, n), value| (sum + value, n + 1));
        Self {
            value: (n > 0).then(|| sum / n as f64),
            n,
        }
    }

    /// The fraction of `true`s.
    pub fn rate(hits: impl IntoIterator<Item = bool>) -> Self {
        Self::mean(hits.into_iter().map(|hit| if hit { 1.0 } else { 0.0 }))
    }
}

impl MetricsReport {
    /// Groups runs by their golden category.
    pub fn of(runs: &[EvalRun]) -> Self {
        let scores = |category: Option<CategoryKind>| {
            runs.iter()
                .filter(move |run| category.is_none_or(|c| run.golden.expected.category() == c))
                .map(|run| &run.scores)
        };
        Self {
            overall: Metrics::of(scores(None)),
            per_category: CATEGORIES
                .into_iter()
                .map(|category| CategoryMetrics {
                    category,
                    metrics: Metrics::of(scores(Some(category))),
                })
                .collect(),
        }
    }

    /// The metrics of one golden category; `None` only for a report that lacks it.
    pub fn category(&self, category: CategoryKind) -> Option<&Metrics> {
        self.per_category
            .iter()
            .find(|entry| entry.category == category)
            .map(|entry| &entry.metrics)
    }
}

impl Metrics {
    /// Aggregates the scores of a set of runs.
    pub fn of<'a>(scores: impl Iterator<Item = &'a RunScores> + Clone) -> Self {
        Self {
            runs: scores.clone().count(),
            classification_accuracy: Metric::rate(
                scores.clone().filter_map(|s| s.classified_correctly),
            ),
            retrieval_recall_at_50: Metric::mean(scores.clone().filter_map(|s| s.recall_at_50)),
            rerank_hit_at_5: Metric::rate(scores.clone().filter_map(|s| s.hit_at_5)),
            answer_precision: Metric::rate(scores.clone().filter_map(|s| s.answer_correct)),
            decline_rate: Metric::rate(scores.clone().map(|s| s.declined)),
            failure_rate: Metric::rate(scores.clone().map(|s| s.failed)),
            citation_validity: Metric::rate(scores.filter_map(|s| s.citations_valid)),
        }
    }

    /// The field `name` refers to.
    pub fn get(&self, name: MetricName) -> Metric {
        match name {
            MetricName::ClassificationAccuracy => self.classification_accuracy,
            MetricName::RetrievalRecallAt50 => self.retrieval_recall_at_50,
            MetricName::RerankHitAt5 => self.rerank_hit_at_5,
            MetricName::AnswerPrecision => self.answer_precision,
            MetricName::DeclineRate => self.decline_rate,
            MetricName::FailureRate => self.failure_rate,
            MetricName::CitationValidity => self.citation_validity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Names a [`Metrics`] field; the serde name is the field name, as used in baseline `tolerances`.
pub enum MetricName {
    /// [`Metrics::classification_accuracy`].
    ClassificationAccuracy,
    /// [`Metrics::retrieval_recall_at_50`].
    #[serde(rename = "retrieval_recall_at_50")]
    RetrievalRecallAt50,
    /// [`Metrics::rerank_hit_at_5`].
    #[serde(rename = "rerank_hit_at_5")]
    RerankHitAt5,
    /// [`Metrics::answer_precision`].
    AnswerPrecision,
    /// [`Metrics::decline_rate`].
    DeclineRate,
    /// [`Metrics::failure_rate`].
    FailureRate,
    /// [`Metrics::citation_validity`].
    CitationValidity,
}

impl MetricName {
    /// Every metric, in report order.
    pub const ALL: [Self; 7] = [
        Self::ClassificationAccuracy,
        Self::RetrievalRecallAt50,
        Self::RerankHitAt5,
        Self::AnswerPrecision,
        Self::DeclineRate,
        Self::FailureRate,
        Self::CitationValidity,
    ];

    /// `false` for the decline and failure rates, where a rise is the regression.
    pub fn higher_is_better(self) -> bool {
        !matches!(self, Self::DeclineRate | Self::FailureRate)
    }

    /// Short column label for terminal tables.
    pub fn label(self) -> &'static str {
        match self {
            Self::ClassificationAccuracy => "classify",
            Self::RetrievalRecallAt50 => "recall@50",
            Self::RerankHitAt5 => "hit@5",
            Self::AnswerPrecision => "precision",
            Self::DeclineRate => "declined",
            Self::FailureRate => "failed",
            Self::CitationValidity => "citations",
        }
    }
}

/// A previous results file (or any JSON with its `metrics`), plus per-metric tolerances.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    /// The metrics to compare against.
    pub metrics: MetricsReport,
    /// Allowed drop (or rise, for decline and failure rates) before a metric counts as regressed. Missing means 0.
    #[serde(default)]
    pub tolerances: BTreeMap<MetricName, f64>,
    /// The runs behind `metrics`, when the file is a results file; needed to compare a subset of questions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<EvalRun>,
}

impl Baseline {
    /// The baseline as if it had run only the questions in `ids`.
    ///
    /// # Errors
    ///
    /// When the baseline has no recorded runs, or did not run every id.
    #[eros::context("restricting the baseline to {:?}", ids)]
    pub fn restricted_to(self, ids: &[String]) -> eros::Result<Self> {
        eros::ensure!(
            !self.runs.is_empty(),
            "the baseline has no recorded runs, so it cannot be compared on a subset of questions"
        );
        let runs: Vec<EvalRun> = self
            .runs
            .into_iter()
            .filter(|run| ids.contains(&run.golden.id))
            .collect();
        let missing: Vec<&str> = ids
            .iter()
            .filter(|id| !runs.iter().any(|run| run.golden.id == **id))
            .map(String::as_str)
            .collect();
        eros::ensure!(missing.is_empty(), "the baseline did not run {missing:?}");
        Ok(Self {
            metrics: MetricsReport::of(&runs),
            tolerances: self.tolerances,
            runs,
        })
    }
}

/// A metric worse than the baseline by more than its tolerance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Regression {
    /// `None` for the overall metrics.
    pub category: Option<CategoryKind>,
    /// Which metric.
    pub metric: MetricName,
    /// The baseline value.
    pub baseline: f64,
    /// `None` when no run counted toward the metric this time.
    pub current: Option<f64>,
    /// The allowed change that was exceeded.
    pub tolerance: f64,
}

/// Every metric, overall and per category, that is worse than the baseline by more than its tolerance.
pub fn regressions(baseline: &Baseline, current: &MetricsReport) -> Vec<Regression> {
    let scopes = std::iter::once((
        None,
        Some(&baseline.metrics.overall),
        Some(&current.overall),
    ))
    .chain(CATEGORIES.into_iter().map(|category| {
        (
            Some(category),
            baseline.metrics.category(category),
            current.category(category),
        )
    }));
    let mut found = Vec::new();
    for (category, base, now) in scopes {
        let Some(base) = base else { continue };
        for metric in MetricName::ALL {
            let Some(baseline_value) = base.get(metric).value else {
                continue;
            };
            let current_value = now.and_then(|now| now.get(metric).value);
            let tolerance = baseline.tolerances.get(&metric).copied().unwrap_or(0.0);
            let (base_steps, tolerance_steps) = (steps(baseline_value), steps(tolerance));
            let regressed = match current_value.map(steps) {
                None => true,
                Some(value) if metric.higher_is_better() => value < base_steps - tolerance_steps,
                Some(value) => value > base_steps + tolerance_steps,
            };
            if regressed {
                found.push(Regression {
                    category,
                    metric,
                    baseline: baseline_value,
                    current: current_value,
                    tolerance,
                });
            }
        }
    }
    found
}

/// The resolution [`regressions`] compares metric values and tolerances at. A mean parsed back from a results file
/// can differ in its last bit from the same mean computed from runs, so the raw `f64`s are not compared.
pub const METRIC_RESOLUTION: f64 = 1e-9;

/// `value` as a whole number of [`METRIC_RESOLUTION`] steps.
fn steps(value: f64) -> i64 {
    (value / METRIC_RESOLUTION).round() as i64
}

pub(super) fn scope_name(category: Option<CategoryKind>) -> &'static str {
    category.as_ref().map_or("overall", CategoryKind::as_str)
}

pub(super) fn percent(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{:.1}%", value * 100.0))
}

pub(super) fn cell(metric: Metric) -> String {
    format!("{} ({})", percent(metric.value), metric.n)
}

impl fmt::Display for MetricsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<8} {:>5}", "scope", "runs")?;
        for metric in MetricName::ALL {
            write!(f, " {:>14}", metric.label())?;
        }
        writeln!(f)?;
        let rows = std::iter::once((None, &self.overall)).chain(
            self.per_category
                .iter()
                .map(|entry| (Some(entry.category), &entry.metrics)),
        );
        for (category, metrics) in rows {
            write!(f, "{:<8} {:>5}", scope_name(category), metrics.runs)?;
            for metric in MetricName::ALL {
                write!(f, " {:>14}", cell(metrics.get(metric)))?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Each cell is value (runs counted). Recall@{RECALL_K} and hit@{HIT_K} count runs with \
             relevant chunks whose step ran; precision counts answered runs with a gradable answer."
        )
    }
}

impl fmt::Display for Regression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}: {} -> {} (tolerance {:.1} pts)",
            scope_name(self.category),
            self.metric.label(),
            percent(Some(self.baseline)),
            percent(self.current),
            self.tolerance * 100.0,
        )
    }
}
