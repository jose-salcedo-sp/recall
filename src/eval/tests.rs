use std::collections::HashMap;

use async_trait::async_trait;
use chrono::Utc;

use super::{
    golden::{Expected, GoldenQuestion, YesNo},
    metrics::{
        Baseline, EvalRun, Metric, MetricName, MetricsReport, Progress, Results, evaluate,
        regressions,
    },
    tune::{Candidate, tune},
    *,
};
use crate::{
    config::{PerCategory, StepName},
    domain::{
        AtLeastOne, CategoryKind, ChunkId, ChunkRef, ChunkVersion, Cited, Probability, Provenance,
        Question, RunId,
    },
    pipeline::{
        DeclineReason,
        report::{Answer, CannedAnswerer, Citation, Outcome, RunReport, Verdict},
        trace::{
            ClassifyRecord, RankedChunk, RerankRecord, RetrieveRecord, StepOutcome, Trace,
            TraceEntry, TraceRecord,
        },
    },
    retrieval::Relevance,
};

// ---- fixtures ----

fn probability(value: f32) -> Probability {
    Probability::new(value).unwrap()
}

fn chunk(id: &str) -> ChunkRef {
    ChunkRef {
        id: ChunkId(id.into()),
        version: ChunkVersion(format!("v-{id}")),
    }
}

fn ranked(ids: &[&str]) -> Vec<RankedChunk> {
    ids.iter()
        .enumerate()
        .map(|(rank, id)| RankedChunk {
            chunk: chunk(id),
            score: Relevance(1.0 - rank as f32 / 100.0),
            probability: None,
            context_for: None,
        })
        .collect()
}

fn entry(step: &str, outcome: StepOutcome, record: Option<TraceRecord>) -> TraceEntry {
    TraceEntry {
        step: step.into(),
        group: None,
        started_at: Utc::now(),
        duration_ms: 1,
        model: None,
        prompt_version: None,
        outcome,
        record,
    }
}

fn passed(score: Option<f32>) -> StepOutcome {
    StepOutcome::Passed {
        score: score.map(probability),
        threshold: None,
    }
}

fn low_confidence(score: f32, threshold: f32) -> StepOutcome {
    StepOutcome::Declined {
        reason: DeclineReason::LowConfidence,
        score: Some(probability(score)),
        threshold: Some(probability(threshold)),
    }
}

fn classify(category: CategoryKind, score: f32) -> TraceEntry {
    entry(
        "classify",
        passed(Some(score)),
        Some(TraceRecord::Classify(ClassifyRecord {
            raw: None,
            category: Some(category),
            fallback_from: None,
        })),
    )
}

fn retrieve(ids: &[&str]) -> TraceEntry {
    entry(
        "retrieve",
        passed(None),
        Some(TraceRecord::Retrieve(RetrieveRecord {
            chunks: ranked(ids),
            expansion: Default::default(),
        })),
    )
}

fn rerank_with(outcome: StepOutcome, ids: &[&str]) -> TraceEntry {
    entry(
        "rerank",
        outcome,
        Some(TraceRecord::Rerank(RerankRecord {
            raw: None,
            chunks: ranked(ids),
        })),
    )
}

fn rerank(ids: &[&str]) -> TraceEntry {
    rerank_with(passed(Some(0.9)), ids)
}

fn scored(step: &str, outcome: StepOutcome) -> TraceEntry {
    entry(step, outcome, None)
}

/// `Trace::push` is private to the pipeline; the trace's serialized form is its public shape.
fn trace(entries: Vec<TraceEntry>) -> Trace {
    serde_json::from_value(serde_json::to_value(entries).unwrap()).unwrap()
}

fn golden(id: &str, expected: Expected, relevant: &[&str]) -> GoldenQuestion {
    GoldenQuestion {
        id: id.into(),
        synthetic: true,
        question: Question::new(format!("question {id}")),
        expected,
        relevant_chunks: relevant.iter().map(|id| ChunkId((*id).into())).collect(),
        source_thread: format!("https://example.invalid/{id}"),
    }
}

fn cited(value: &str, source: &str) -> Cited<String> {
    Cited {
        value: value.into(),
        sources: AtLeastOne::one(Provenance::Chunk(ChunkId(source.into()))),
    }
}

fn citation(id: &str) -> Citation {
    Citation {
        chunk: chunk(id),
        title: id.into(),
        url: format!("https://example.invalid/{id}"),
        excerpt: String::new(),
    }
}

fn answered(verdict: Option<Verdict>, cites: &[&str]) -> Outcome {
    Outcome::Answered(Answer {
        verdict,
        why: cites.iter().map(|id| cited("because", id)).collect(),
        citations: cites.iter().map(|id| citation(id)).collect(),
    })
}

fn yes(value: bool) -> Option<Verdict> {
    Some(Verdict::Boolean {
        yes: value,
        confidence: probability(0.9),
    })
}

fn selected(option: &str, source: &str) -> Option<Verdict> {
    Some(Verdict::Choice {
        selected: cited(option, source),
        rejected: vec![],
    })
}

fn declined(at: &str) -> Outcome {
    Outcome::Declined {
        at: at.into(),
        reason: DeclineReason::LowConfidence,
        score: None,
        threshold: None,
        sources: vec![],
    }
}

fn failed(at: &str) -> Outcome {
    Outcome::Failed {
        at: at.into(),
        error: "boom".into(),
    }
}

fn report(
    golden: &GoldenQuestion,
    category: Option<CategoryKind>,
    outcome: Outcome,
    entries: Vec<TraceEntry>,
) -> RunReport {
    RunReport {
        id: RunId::new(),
        question: golden.question.clone(),
        category,
        outcome,
        trace: trace(entries),
    }
}

/// Returns a crafted report per question text.
struct Scripted(HashMap<String, RunReport>);

impl Scripted {
    fn new(runs: &[(GoldenQuestion, RunReport)]) -> Self {
        Self(
            runs.iter()
                .map(|(golden, report)| (golden.question.as_str().to_owned(), report.clone()))
                .collect(),
        )
    }
}

#[async_trait]
impl Answerer for Scripted {
    async fn answer(&self, question: Question) -> RunReport {
        self.0[question.as_str()].clone()
    }
}

/// Five runs whose metrics are computed by hand in `metrics_match_hand_computed_values`.
fn fixture() -> Vec<(GoldenQuestion, RunReport)> {
    use CategoryKind::*;

    // Correct boolean: recall 1/2, hit at rank 2, valid citations.
    let b1 = golden("b1", Expected::Boolean(YesNo::Yes), &["a", "b"]);
    let b1_report = report(
        &b1,
        Some(Boolean),
        answered(yes(true), &["a"]),
        vec![
            classify(Boolean, 0.9),
            retrieve(&["a", "x", "y"]),
            rerank(&["x", "a"]),
            scored("answer", passed(Some(0.8))),
        ],
    );

    // Wrong choice: recall 1, relevant chunk only at rerank rank 6, cites an unretrieved chunk.
    let c1 = golden("c1", Expected::Choice("S_TCODE".into()), &["c"]);
    let c1_report = report(
        &c1,
        Some(Choice),
        answered(selected("S_TABU_DIS", "z"), &["z"]),
        vec![
            classify(Choice, 0.8),
            retrieve(&["c"]),
            rerank(&["x1", "x2", "x3", "x4", "x5", "c"]),
            scored("answer", passed(Some(0.6))),
        ],
    );

    // Ranking misclassified as open, nothing relevant retrieved, declined at rerank.
    let r1 = golden(
        "r1",
        Expected::Ranking(vec!["p".into(), "q".into()]),
        &["r"],
    );
    let r1_report = report(
        &r1,
        Some(Open),
        declined("rerank"),
        vec![
            classify(Open, 0.7),
            retrieve(&["s"]),
            rerank_with(low_confidence(0.2, 0.3), &["s"]),
        ],
    );

    // Failed before anything was recorded.
    let o1 = golden("o1", Expected::Open, &[]);
    let o1_report = report(&o1, None, failed("classify"), vec![]);

    // Answered open: nothing to grade, valid citation.
    let o2 = golden("o2", Expected::Open, &["o"]);
    let o2_report = report(
        &o2,
        Some(Open),
        answered(None, &["o"]),
        vec![classify(Open, 0.9), retrieve(&["o"]), rerank(&["o"])],
    );

    vec![
        (b1, b1_report),
        (c1, c1_report),
        (r1, r1_report),
        (o1, o1_report),
        (o2, o2_report),
    ]
}

async fn evaluate_fixture(runs: &[(GoldenQuestion, RunReport)]) -> Results {
    let golden = runs.iter().map(|(golden, _)| golden.clone()).collect();
    evaluate(&Scripted::new(runs), golden, 2, &()).await
}

fn metric(value: f64, n: usize) -> Metric {
    Metric {
        value: Some(value),
        n,
    }
}

fn assert_metric(actual: Metric, expected: Metric) {
    assert_eq!(actual.n, expected.n, "n of {actual:?}");
    match (actual.value, expected.value) {
        (Some(a), Some(e)) => assert!((a - e).abs() < 1e-9, "{a} != {e}"),
        (a, e) => assert_eq!(a, e),
    }
}

const NONE: Metric = Metric { value: None, n: 0 };

// ---- golden set ----

#[test]
fn example_golden_set_parses_and_is_marked_synthetic() {
    let questions = golden::parse(include_str!("../../eval/golden/example.jsonl")).unwrap();
    assert!(questions.len() >= 2);
    assert!(questions.iter().all(|q| q.synthetic));
    assert!(
        questions
            .iter()
            .all(|q| q.question.as_str().starts_with("SYNTHETIC"))
    );
    let categories: Vec<CategoryKind> = questions.iter().map(|q| q.expected.category()).collect();
    assert_eq!(categories, metrics::CATEGORIES);
}

#[test]
fn daas_golden_set_parses_and_points_at_markdown_chunks() {
    let questions = golden::parse(include_str!("../../eval/golden/daas.jsonl")).unwrap();
    assert!(questions.len() >= 50);
    assert!(questions.iter().all(|q| {
        !q.relevant_chunks.is_empty()
            && q.relevant_chunks
                .iter()
                .all(|c| c.0.starts_with("markdown:") && c.0.contains('#'))
    }));
}

#[test]
fn golden_line_round_trips_through_serde() {
    let line = r#"{"id":"q1","question":"Which?","category":"choice","expected":"B","relevant_chunks":["markdown:arm-1.md#c1"],"source_thread":"https://x"}"#;
    let parsed = golden::parse(line).unwrap();
    assert_eq!(parsed[0].expected, Expected::Choice("B".into()));
    assert!(!parsed[0].synthetic);
    let reparsed = golden::parse(&serde_json::to_string(&parsed[0]).unwrap()).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn golden_errors_report_line_numbers() {
    let good = r#"{"id":"q1","question":"Is it?","category":"boolean","expected":"yes","relevant_chunks":[],"source_thread":"https://x"}"#;
    let cases = [
        (format!("{good}\n\n{{not json"), 3, "key must be a string"),
        (
            format!(
                "{good}\n{}",
                r#"{"id":"q2","question":"Is it?","category":"boolean","expected":"maybe","relevant_chunks":[],"source_thread":"https://x"}"#
            ),
            2,
            "unknown variant `maybe`",
        ),
        (
            r#"{"id":"q2","question":"Is it?","category":"boolean","relevant_chunks":[],"source_thread":"https://x"}"#.to_owned(),
            1,
            "expected",
        ),
        (
            r#"{"id":"q2","question":"Rank","category":"ranking","expected":["only"],"relevant_chunks":[],"source_thread":"https://x"}"#.to_owned(),
            1,
            "at least 2 items",
        ),
        (
            r#"{"id":"q2","question":"Why?","category":"open","source_thread":"https://x"}"#
                .to_owned(),
            1,
            "relevant_chunks",
        ),
        (format!("{good}\n{good}"), 2, "duplicate id `q1`"),
    ];
    for (text, line, message) in cases {
        let error = golden::parse(&text).unwrap_err();
        assert_eq!(error.line, line, "{error}");
        assert!(error.message.contains(message), "{error}");
        assert!(error.to_string().starts_with(&format!("line {line}: ")));
    }
}

#[test]
fn expected_answers_grade_verdicts() {
    let choice = Expected::Choice("S_TCODE".into());
    assert_eq!(
        choice.matches(selected(" s_tcode ", "a").as_ref()),
        Some(true)
    );
    assert_eq!(choice.matches(yes(true).as_ref()), Some(false));
    assert_eq!(choice.matches(None), Some(false));

    let ranking = Expected::Ranking(vec!["a".into(), "b".into()]);
    let items = |order: [&str; 2]| Verdict::Ranking {
        criterion: "order".into(),
        items: order.iter().map(|item| cited(item, "x")).collect(),
    };
    assert_eq!(ranking.matches(Some(&items(["a", "b"]))), Some(true));
    assert_eq!(ranking.matches(Some(&items(["b", "a"]))), Some(false));

    assert_eq!(
        Expected::Boolean(YesNo::No).matches(yes(false).as_ref()),
        Some(true)
    );
    assert_eq!(Expected::Open.matches(yes(true).as_ref()), None);
}

// ---- metrics ----

#[tokio::test]
async fn metrics_match_hand_computed_values() {
    let results = evaluate_fixture(&fixture()).await;
    let overall = &results.metrics.overall;
    assert_eq!(overall.runs, 5);
    // b1, c1, o2 right; r1 wrong; o1 never classified.
    assert_metric(overall.classification_accuracy, metric(0.75, 4));
    // (0.5 + 1 + 0 + 1) / 4; o1 has no relevant chunks.
    assert_metric(overall.retrieval_recall_at_50, metric(0.625, 4));
    // b1 and o2 hit; c1's relevant chunk is at rank 6.
    assert_metric(overall.rerank_hit_at_5, metric(0.5, 4));
    // b1 right, c1 wrong; o2 is open and ungraded.
    assert_metric(overall.answer_precision, metric(0.5, 2));
    assert_metric(overall.decline_rate, metric(0.2, 5));
    assert_metric(overall.failure_rate, metric(0.2, 5));
    // c1 cites an unretrieved chunk.
    assert_metric(overall.citation_validity, metric(2.0 / 3.0, 3));

    let ids: Vec<&str> = results.runs.iter().map(|r| r.golden.id.as_str()).collect();
    assert_eq!(ids, ["b1", "c1", "r1", "o1", "o2"]);
}

#[tokio::test]
async fn metrics_break_down_per_expected_category() {
    let results = evaluate_fixture(&fixture()).await;
    let category = |kind| results.metrics.category(kind).unwrap();

    let boolean = category(CategoryKind::Boolean);
    assert_eq!(boolean.runs, 1);
    assert_metric(boolean.answer_precision, metric(1.0, 1));
    assert_metric(boolean.retrieval_recall_at_50, metric(0.5, 1));

    let choice = category(CategoryKind::Choice);
    assert_metric(choice.answer_precision, metric(0.0, 1));
    assert_metric(choice.rerank_hit_at_5, metric(0.0, 1));
    assert_metric(choice.citation_validity, metric(0.0, 1));

    let ranking = category(CategoryKind::Ranking);
    assert_metric(ranking.classification_accuracy, metric(0.0, 1));
    assert_metric(ranking.decline_rate, metric(1.0, 1));
    assert_metric(ranking.answer_precision, NONE);

    let open = category(CategoryKind::Open);
    assert_eq!(open.runs, 2);
    assert_metric(open.classification_accuracy, metric(1.0, 1));
    assert_metric(open.failure_rate, metric(0.5, 2));
    assert_metric(open.answer_precision, NONE);
    assert_metric(open.citation_validity, metric(1.0, 1));
}

#[tokio::test]
async fn recall_and_hit_only_count_the_top_k() {
    let relevant = golden("q", Expected::Open, &["target"]);
    let mut retrieved: Vec<String> = (0..metrics::RECALL_K).map(|i| format!("n{i}")).collect();
    retrieved.push("target".into());
    let retrieved: Vec<&str> = retrieved.iter().map(String::as_str).collect();
    let run = (
        relevant.clone(),
        report(
            &relevant,
            Some(CategoryKind::Open),
            answered(None, &[]),
            vec![
                retrieve(&retrieved),
                rerank(&["n0", "n1", "n2", "n3", "n4", "target"]),
            ],
        ),
    );
    let results = evaluate_fixture(&[run]).await;
    assert_metric(
        results.metrics.overall.retrieval_recall_at_50,
        metric(0.0, 1),
    );
    assert_metric(results.metrics.overall.rerank_hit_at_5, metric(0.0, 1));
}

#[test]
fn recall_counts_matches_only() {
    let q = golden("q", Expected::Open, &["target", "neighbour"]);
    // 50 matches, each followed by a context chunk; `target` is the 50th match at position 99, and the relevant
    // `neighbour` is only ever context.
    let mut chunks = Vec::new();
    for i in 0..metrics::RECALL_K {
        let id = if i == metrics::RECALL_K - 1 {
            "target".to_owned()
        } else {
            format!("m{i}")
        };
        let mut pair = ranked(&[&id, if i == 0 { "neighbour" } else { "ctx" }]);
        pair[1].context_for = Some(ChunkId(id.clone()));
        chunks.extend(pair);
    }
    let record = RetrieveRecord {
        chunks,
        expansion: Default::default(),
    };

    assert_eq!(
        metrics::recall_at(metrics::RECALL_K, &q.relevant_chunks, &record),
        Some(0.5)
    );
    assert_eq!(
        metrics::recall_at(retrieval::SHORT_RECALL_K, &q.relevant_chunks, &record),
        Some(0.0)
    );
    let run = EvalRun::new(
        q.clone(),
        report(
            &q,
            Some(CategoryKind::Open),
            answered(None, &[]),
            vec![entry(
                "retrieve",
                passed(None),
                Some(TraceRecord::Retrieve(record)),
            )],
        ),
    );
    assert_eq!(run.scores.recall_at_50, Some(0.5));
}

#[test]
fn results_written_before_context_for_still_parse() {
    let old = r#"{"chunk": {"id": "markdown:a.md#x", "version": "v1"}, "score": 0.5}"#;
    let chunk: RankedChunk = serde_json::from_str(old).unwrap();
    assert_eq!(chunk.context_for, None);
    assert_eq!(serde_json::to_string(&chunk).unwrap(), old.replace(' ', ""));

    let v4 = include_str!("../../eval/results/daas-2026-10-21-v4.json");
    let results: Results = serde_json::from_str(v4).unwrap();
    assert_eq!(results.runs.len(), 58);
    assert_eq!(
        results.setup.as_ref().unwrap().config.rerank.judge(),
        crate::config::RerankJudge::SystemOne(crate::config::SystemOneJudge { candidates: 25 })
    );
    let baseline = retrieval::BaselineRecords::parse(v4).unwrap();
    for run in &results.runs {
        assert!(baseline.contains(&run.golden.id));
        assert_eq!(
            baseline.record(&run.golden.id),
            metrics::retrieve_record(&run.report)
        );
    }
}

#[test]
fn citation_validity_checks_versions_and_claim_provenance() {
    let q = golden("q", Expected::Open, &[]);
    let base = vec![retrieve(&["a"]), rerank(&["a", "expanded"])];

    let mut stale = answered(None, &["a"]);
    if let Outcome::Answered(answer) = &mut stale {
        answer.citations[0].chunk.version = ChunkVersion("old".into());
    }
    let mut uncited_claim = answered(None, &["expanded"]);
    if let Outcome::Answered(answer) = &mut uncited_claim {
        answer.why.push(cited("made up", "elsewhere"));
    }

    for (outcome, valid) in [
        (answered(None, &["a", "expanded"]), true),
        (stale, false),
        (uncited_claim, false),
    ] {
        let run = EvalRun::new(q.clone(), report(&q, None, outcome, base.clone()));
        assert_eq!(run.scores.citations_valid, Some(valid));
    }
}

#[derive(Default)]
struct RecordingProgress {
    started: std::sync::Mutex<Vec<String>>,
    finished: std::sync::Mutex<Vec<String>>,
}

impl Progress for RecordingProgress {
    fn started(&self, golden: &GoldenQuestion) {
        self.started.lock().unwrap().push(golden.id.clone());
    }

    fn finished(&self, run: &EvalRun, _: std::time::Duration) {
        self.finished.lock().unwrap().push(run.golden.id.clone());
    }
}

#[tokio::test]
async fn progress_sees_every_run_and_results_keep_golden_order() {
    let runs = fixture();
    let golden: Vec<GoldenQuestion> = runs.iter().map(|(golden, _)| golden.clone()).collect();
    let ids: Vec<String> = golden.iter().map(|g| g.id.clone()).collect();
    let progress = RecordingProgress::default();

    let results = evaluate(&Scripted::new(&runs), golden, 3, &progress).await;

    let mut started = progress.started.into_inner().unwrap();
    let mut finished = progress.finished.into_inner().unwrap();
    started.sort();
    finished.sort();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(started, sorted);
    assert_eq!(finished, sorted);
    let order: Vec<String> = results.runs.iter().map(|r| r.golden.id.clone()).collect();
    assert_eq!(order, ids);
}

#[tokio::test]
async fn progress_line_shows_status_scores_and_the_halt() {
    let results = evaluate_fixture(&fixture()).await;
    let elapsed = std::time::Duration::from_millis(41_200);
    let by_id = |id: &str| results.runs.iter().find(|r| r.golden.id == id).unwrap();

    let answered = progress::line(by_id("o2"), elapsed, 3, 58);
    assert!(
        answered.contains("ANSWERED [   41.2s] ( 3/58) o2 open"),
        "{answered}"
    );
    assert!(
        answered.contains("classify ✓  hit@5 ✓  recall 100%  answer -"),
        "{answered}"
    );

    let failed = progress::line(by_id("o1"), elapsed, 4, 58);
    assert!(failed.contains("FAILED"), "{failed}");
    assert!(failed.contains("at classify:"), "{failed}");
}

#[tokio::test]
async fn canned_failures_count_only_toward_failure_rate() {
    let answerer = CannedAnswerer {
        category: None,
        outcome: failed("pipeline"),
    };
    let golden = fixture().into_iter().map(|(golden, _)| golden).collect();
    let results = evaluate(&answerer, golden, 3, &()).await;
    let overall = &results.metrics.overall;
    assert_metric(overall.failure_rate, metric(1.0, 5));
    assert_metric(overall.decline_rate, metric(0.0, 5));
    assert_metric(overall.classification_accuracy, NONE);
    assert_metric(overall.retrieval_recall_at_50, NONE);
    assert_metric(overall.answer_precision, NONE);
    assert_metric(overall.citation_validity, NONE);
}

#[tokio::test]
async fn metrics_table_lists_every_scope_and_metric() {
    let table = evaluate_fixture(&fixture()).await.metrics.to_string();
    for label in MetricName::ALL.map(MetricName::label) {
        assert!(table.contains(label), "{table}");
    }
    for scope in ["overall", "boolean", "choice", "ranking", "open"] {
        assert!(table.lines().any(|line| line.starts_with(scope)), "{table}");
    }
    assert!(table.contains("62.5% (4)"), "{table}");
}

// ---- baseline gate ----

#[tokio::test]
async fn baseline_gate_passes_on_equal_metrics_and_fails_on_regression() {
    let runs = fixture();
    let baseline_results = evaluate_fixture(&runs).await;
    let baseline = Baseline {
        metrics: baseline_results.metrics.clone(),
        tolerances: Default::default(),
        runs: vec![],
    };
    assert_eq!(regressions(&baseline, &baseline_results.metrics), vec![]);

    // b1 now answers "no": precision drops overall and for boolean.
    let mut worse = runs.clone();
    worse[0].1.outcome = answered(yes(false), &["a"]);
    let current = evaluate_fixture(&worse).await.metrics;
    let found = regressions(&baseline, &current);
    let found: Vec<(Option<CategoryKind>, MetricName)> =
        found.iter().map(|r| (r.category, r.metric)).collect();
    assert_eq!(
        found,
        [
            (None, MetricName::AnswerPrecision),
            (Some(CategoryKind::Boolean), MetricName::AnswerPrecision),
        ]
    );

    // A tolerance of 0.5 absorbs the overall drop (0.5 -> 0.0 is exactly 0.5), not boolean's (1.0 -> 0.0).
    let tolerant = Baseline {
        tolerances: [(MetricName::AnswerPrecision, 0.5)].into(),
        ..baseline.clone()
    };
    let found = regressions(&tolerant, &current);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].category, Some(CategoryKind::Boolean));
}

fn results_file<T: serde::de::DeserializeOwned>(name: &str) -> T {
    read_json(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("eval/results")
            .join(name),
    )
    .unwrap()
}

/// The gate as `drew eval run` applies it: the baseline parsed from its file, the current metrics computed from runs.
fn gate(baseline: &str, current: &str) -> Vec<(Option<CategoryKind>, MetricName)> {
    let baseline: Baseline = results_file(baseline);
    let current = MetricsReport::of(&results_file::<Results>(current).runs);
    regressions(&baseline, &current)
        .iter()
        .map(|r| (r.category, r.metric))
        .collect()
}

#[test]
fn baseline_gate_passes_equal_counts_read_back_from_a_results_file() {
    // Both classify 54 of 58; v5 improves or keeps every other metric.
    assert_eq!(
        gate("daas-2026-10-21-v4.json", "daas-2026-10-22-v5.json"),
        vec![]
    );
    assert_eq!(
        gate("daas-2026-10-22-v5.json", "daas-2026-10-22-v5.json"),
        vec![]
    );
}

#[tokio::test]
async fn baseline_gate_treats_rising_decline_rate_and_lost_metrics_as_regressions() {
    let runs = fixture();
    let baseline = Baseline {
        metrics: evaluate_fixture(&runs).await.metrics,
        tolerances: Default::default(),
        runs: vec![],
    };

    // b1 declines: decline rate rises, and boolean precision and citation validity lose their only run.
    let mut worse = runs.clone();
    worse[0].1.outcome = declined("answer");
    let current = evaluate_fixture(&worse).await.metrics;
    let found = regressions(&baseline, &current);
    let decline = found
        .iter()
        .find(|r| r.category.is_none() && r.metric == MetricName::DeclineRate)
        .unwrap();
    assert_eq!((decline.baseline, decline.current), (0.2, Some(0.4)));
    let lost = found
        .iter()
        .find(|r| {
            r.category == Some(CategoryKind::Boolean) && r.metric == MetricName::AnswerPrecision
        })
        .unwrap();
    assert_eq!(lost.current, None);
    assert!(lost.to_string().contains("boolean precision: 100.0% -> -"));

    // An improving decline rate is not a regression.
    let mut better = runs.clone();
    better[2].1.outcome = answered(None, &["s"]);
    let current = evaluate_fixture(&better).await.metrics;
    assert!(
        regressions(&baseline, &current)
            .iter()
            .all(|r| r.metric != MetricName::DeclineRate)
    );
}

#[tokio::test]
async fn run_command_writes_results_and_gates_on_the_baseline_file() {
    let runs = fixture();
    let dir = std::env::temp_dir().join(format!("drew-eval-{}", RunId::new()));
    fs::create_dir_all(&dir).unwrap();
    let golden_path = dir.join("golden.jsonl");
    let golden_text: Vec<String> = runs
        .iter()
        .map(|(golden, _)| serde_json::to_string(golden).unwrap())
        .collect();
    fs::write(&golden_path, golden_text.join("\n")).unwrap();
    let args = |out: &str, baseline: Option<PathBuf>| RunArgs {
        golden: golden_path.clone(),
        out: dir.join(out),
        baseline,
        concurrency: 2,
        questions: vec![],
    };

    run_golden(
        Arc::new(Scripted::new(&runs)),
        args("first.json", None),
        None,
    )
    .await
    .unwrap();
    let first: Results = read_json(&dir.join("first.json")).unwrap();
    assert_eq!(first.runs.len(), 5);
    assert_eq!(first.runs[0].report, runs[0].1);

    // A results file is a valid baseline as is.
    run_golden(
        Arc::new(Scripted::new(&runs)),
        args("second.json", Some(dir.join("first.json"))),
        None,
    )
    .await
    .unwrap();

    let mut worse = runs.clone();
    worse[0].1.outcome = answered(yes(false), &["a"]);
    let error = run_golden(
        Arc::new(Scripted::new(&worse)),
        args("third.json", Some(dir.join("first.json"))),
        None,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("regressed"), "{error}");
    assert!(dir.join("third.json").exists());

    // Tolerances stored in the baseline file loosen the gate.
    let mut tolerant: serde_json::Value = read_json(&dir.join("first.json")).unwrap();
    tolerant["tolerances"] = MetricName::ALL
        .iter()
        .map(|metric| {
            (
                serde_json::to_value(metric)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned(),
                1.0.into(),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    write_json(&dir.join("tolerant.json"), &tolerant).unwrap();
    run_golden(
        Arc::new(Scripted::new(&worse)),
        args("fourth.json", Some(dir.join("tolerant.json"))),
        None,
    )
    .await
    .unwrap();

    fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn one_question_runs_alone_and_is_gated_against_the_same_question() {
    let runs = fixture();
    let dir = std::env::temp_dir().join(format!("drew-eval-{}", RunId::new()));
    fs::create_dir_all(&dir).unwrap();
    let golden_path = dir.join("golden.jsonl");
    let golden_text: Vec<String> = runs
        .iter()
        .map(|(golden, _)| serde_json::to_string(golden).unwrap())
        .collect();
    fs::write(&golden_path, golden_text.join("\n")).unwrap();
    let args = |out: &str, baseline: Option<PathBuf>, questions: &[&str]| RunArgs {
        golden: golden_path.clone(),
        out: dir.join(out),
        baseline,
        concurrency: 2,
        questions: questions.iter().map(|id| (*id).to_owned()).collect(),
    };
    run_golden(
        Arc::new(Scripted::new(&runs)),
        args("all.json", None, &[]),
        None,
    )
    .await
    .unwrap();

    // Against the whole baseline, o2 alone would "lose" precision; restricted to o2 it matches.
    let baseline = Some(dir.join("all.json"));
    run_golden(
        Arc::new(Scripted::new(&runs)),
        args("o2.json", baseline.clone(), &["o2"]),
        None,
    )
    .await
    .unwrap();
    let only: Results = read_json(&dir.join("o2.json")).unwrap();
    let ids: Vec<&str> = only.runs.iter().map(|run| run.golden.id.as_str()).collect();
    assert_eq!(ids, ["o2"]);

    let error = run_golden(
        Arc::new(Scripted::new(&runs)),
        args("nope.json", None, &["o2", "missing"]),
        None,
    )
    .await
    .unwrap_err();
    assert!(format!("{error:#?}").contains("\"missing\""), "{error:#?}");

    let metrics_only = serde_json::json!({ "metrics": only.metrics });
    write_json(&dir.join("metrics-only.json"), &metrics_only).unwrap();
    let error = run_golden(
        Arc::new(Scripted::new(&runs)),
        args("subset.json", Some(dir.join("metrics-only.json")), &["o2"]),
        None,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("no recorded runs"), "{error}");

    fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn golden_parse_errors_surface_through_the_run_command() {
    let dir = std::env::temp_dir().join(format!("drew-eval-{}", RunId::new()));
    fs::create_dir_all(&dir).unwrap();
    let golden_path = dir.join("golden.jsonl");
    fs::write(&golden_path, "\n{}\n").unwrap();
    let error = run_golden(
        Arc::new(CannedAnswerer {
            category: None,
            outcome: Outcome::Failed {
                at: "pipeline".into(),
                error: "unused".into(),
            },
        }),
        RunArgs {
            golden: golden_path,
            out: dir.join("out.json"),
            baseline: None,
            concurrency: 1,
            questions: vec![],
        },
        None,
    )
    .await
    .unwrap_err();
    assert!(format!("{error:#?}").contains("line 2: "), "{error:#?}");
    fs::remove_dir_all(&dir).unwrap();
}

// ---- tuning ----

/// Three correct answers, one wrong `choice` answer, and one run declined at `answer` on low confidence.
fn tuning_results() -> Results {
    use CategoryKind::*;
    let boolean_run = |id: &str, answer_score: f32| {
        let g = golden(id, Expected::Boolean(YesNo::Yes), &["a"]);
        let r = report(
            &g,
            Some(Boolean),
            answered(yes(true), &["a"]),
            vec![
                classify(Boolean, 0.9),
                retrieve(&["a"]),
                rerank(&["a"]),
                scored("answer", passed(Some(answer_score))),
            ],
        );
        EvalRun::new(g, r)
    };
    let wrong = {
        let g = golden("wrong", Expected::Choice("A".into()), &["a"]);
        let r = report(
            &g,
            Some(Choice),
            answered(selected("B", "a"), &["a"]),
            vec![
                classify(Choice, 0.8),
                retrieve(&["a"]),
                rerank_with(passed(Some(0.8)), &["a"]),
                scored("answer", passed(Some(0.4))),
            ],
        );
        EvalRun::new(g, r)
    };
    let low = {
        let g = golden("low", Expected::Boolean(YesNo::No), &["a"]);
        let r = report(
            &g,
            Some(Boolean),
            declined("answer"),
            vec![
                classify(Boolean, 0.9),
                retrieve(&["a"]),
                rerank(&["a"]),
                scored("answer", low_confidence(0.3, 0.35)),
            ],
        );
        EvalRun::new(g, r)
    };
    Results::new(vec![
        boolean_run("a1", 0.9),
        boolean_run("a2", 0.8),
        boolean_run("a3", 0.7),
        wrong,
        low,
    ])
}

fn candidate(name: &str, thresholds: &[(StepName, f32)]) -> Candidate {
    Candidate {
        name: name.into(),
        thresholds: thresholds
            .iter()
            .map(|(step, t)| (*step, probability(*t)))
            .collect(),
        per_category: PerCategory::default(),
    }
}

#[test]
fn tuning_reapplies_thresholds_to_recorded_scores() {
    let results = tuning_results();
    let mut lenient_choice = candidate("lenient-choice", &[(StepName::Answer, 0.5)]);
    lenient_choice.per_category.choice =
        [(StepName::Answer, probability(0.3))].into_iter().collect();
    let candidates = [
        candidate("none", &[]),
        candidate("answer-0.5", &[(StepName::Answer, 0.5)]),
        candidate("answer-0.75", &[(StepName::Answer, 0.75)]),
        lenient_choice,
        candidate(
            "rerank-0.85",
            &[(StepName::Rerank, 0.85), (StepName::Answer, 0.35)],
        ),
    ];
    let report = tune(&results, &candidates);
    let summary: Vec<_> = report
        .candidates
        .iter()
        .map(|c| {
            let s = &c.overall;
            (
                c.name.as_str(),
                s.answered,
                s.declined,
                s.unresolved,
                s.answered_precision.value,
                s.decline_rate.value,
                c.meets_launch_gate,
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            // `low` would pass every step, but nothing after its original halt was recorded.
            ("none", 4, 1, 1, Some(0.75), Some(0.2), false),
            // Exactly at both gate limits.
            ("answer-0.5", 3, 2, 0, Some(1.0), Some(0.4), true),
            ("answer-0.75", 2, 3, 0, Some(1.0), Some(0.6), false),
            ("lenient-choice", 4, 1, 0, Some(0.75), Some(0.2), false),
            // `wrong` falls below the rerank threshold; `low` at 0.3 below answer 0.35.
            ("rerank-0.85", 3, 2, 0, Some(1.0), Some(0.4), true),
        ]
    );

    let per_choice = |name: &str| {
        let c = report.candidates.iter().find(|c| c.name == name).unwrap();
        let entry = c
            .per_category
            .iter()
            .find(|e| e.category == CategoryKind::Choice)
            .unwrap();
        (entry.stats.answered, entry.stats.declined)
    };
    assert_eq!(per_choice("answer-0.5"), (0, 1));
    assert_eq!(per_choice("lenient-choice"), (1, 0));
}

#[test]
fn tuning_table_highlights_launch_gate_candidates() {
    let results = tuning_results();
    let report = tune(
        &results,
        &[
            candidate("none", &[]),
            candidate("answer-0.5", &[(StepName::Answer, 0.5)]),
        ],
    );
    let table = report.to_string();
    let row = |name: &str| table.lines().find(|line| line.starts_with(name)).unwrap();
    assert!(row("answer-0.5").ends_with("PASS"), "{table}");
    assert!(!row("none").contains("PASS"), "{table}");
    assert!(table.ends_with("met by answer-0.5"), "{table}");

    let none_only = tune(&results, &[candidate("none", &[])]).to_string();
    assert!(none_only.ends_with("no candidate meets it"), "{none_only}");
}

#[test]
fn a_candidate_threshold_outside_zero_to_one_is_rejected() {
    for bad in [
        r#"[{"name": "x", "thresholds": {"answer": 1.5}}]"#,
        r#"[{"name": "x", "per_category": {"boolean": {"answer": -0.1}}}]"#,
    ] {
        assert!(
            serde_json::from_str::<Vec<Candidate>>(bad).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn a_misspelled_candidate_step_or_field_is_rejected_by_name() {
    for (bad, key) in [
        (r#"[{"name": "x", "thresholds": {"anser": 0.7}}]"#, "anser"),
        (
            r#"[{"name": "x", "thresholds": {"classify": 0.7}}]"#,
            "classify",
        ),
        (
            r#"[{"name": "x", "per_category": {"boolean": {"rerankk": 0.7}}}]"#,
            "rerankk",
        ),
        (
            r#"[{"name": "x", "per_category": {"boolen": {}}}]"#,
            "boolen",
        ),
        (
            r#"[{"name": "x", "threshold": {"answer": 0.7}}]"#,
            "threshold",
        ),
    ] {
        let error = serde_json::from_str::<Vec<Candidate>>(bad)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("`{key}`")), "{error}");
    }
}

#[test]
fn a_candidate_round_trips_and_pastes_into_the_config() {
    let text = r#"[{"name": "strict", "thresholds": {"answer": 0.5}, "per_category": {"choice": {"rerank": 0.25}}}]"#;
    let candidates: Vec<Candidate> = serde_json::from_str(text).unwrap();
    assert_eq!(
        serde_json::to_value(&candidates).unwrap(),
        serde_json::from_str::<serde_json::Value>(text).unwrap()
    );
    let config = crate::config::PipelineConfig::parse(
        r#"{"thresholds": {"answer": 0.5}, "per_category": {"choice": {"rerank": 0.25}}}"#,
    )
    .unwrap();
    assert_eq!(config.thresholds, candidates[0].thresholds);
    assert_eq!(config.per_category, candidates[0].per_category);
}

// ---- retrieval only ----

fn corpus() -> Vec<crate::domain::Chunk> {
    use crate::pipeline::steps::fixtures::chunk;
    vec![
        chunk("roles", "sap", "which SAP role grants AGR_USERS access"),
        chunk(
            "sheet",
            "mitigations",
            "the mitigation detail sheet stays empty",
        ),
        chunk("noise", "other", "unrelated text about lunch"),
    ]
}

fn retrieval_golden(id: &str, question: &str, relevant: &crate::domain::Chunk) -> GoldenQuestion {
    GoldenQuestion {
        id: id.into(),
        synthetic: true,
        question: Question::new(question),
        expected: Expected::Open,
        relevant_chunks: vec![relevant.id.clone()],
        source_thread: format!("https://example.invalid/{id}"),
    }
}

#[tokio::test]
async fn retrieval_command_scores_matches_and_diffs_each_question_against_a_baseline() {
    use crate::retrieval::fakes::KeywordRetriever;
    use retrieval::{Change, RetrievalArgs, RetrievalResults, run_retrieval};

    let chunks = corpus();
    let golden = [
        retrieval_golden("roles", "which SAP role grants AGR_USERS", &chunks[0]),
        retrieval_golden("sheet", "mitigation detail sheet empty", &chunks[1]),
    ];
    let dir = std::env::temp_dir().join(format!("drew-eval-{}", RunId::new()));
    fs::create_dir_all(&dir).unwrap();
    let golden_path = dir.join("golden.jsonl");
    let lines: Vec<String> = golden
        .iter()
        .map(|g| serde_json::to_string(g).unwrap())
        .collect();
    fs::write(&golden_path, lines.join("\n")).unwrap();
    let args = |out: &str, baseline: Option<PathBuf>| RetrievalArgs {
        golden: golden_path.clone(),
        out: Some(dir.join(out)),
        baseline,
        concurrency: 2,
        questions: vec![],
    };
    let retriever = |chunks: Vec<crate::domain::Chunk>| Arc::new(KeywordRetriever { chunks });

    // A full `eval run` baseline whose retrieval missed `roles` and found `sheet`.
    let full = Results::new(
        golden
            .iter()
            .map(|g| {
                let ids: Vec<&str> = match g.id.as_str() {
                    "roles" => vec!["elsewhere"],
                    _ => vec![g.relevant_chunks[0].0.as_str()],
                };
                let report = report(g, None, answered(None, &[]), vec![retrieve(&ids)]);
                EvalRun::new(g.clone(), report)
            })
            .collect(),
    );
    write_json(&dir.join("full.json"), &full).unwrap();

    run_retrieval(
        retriever(chunks.clone()),
        args("first.json", Some(dir.join("full.json"))),
        None,
    )
    .await
    .unwrap();
    let first: RetrievalResults = read_json(&dir.join("first.json")).unwrap();
    assert_eq!(first.runs.len(), 2);
    assert_eq!(first.runs[0].scores.recall_at_10, Some(1.0));
    assert_eq!(first.runs[1].scores.recall_at_50, Some(1.0));
    assert_eq!(first.metrics.overall.recall_at_50.value, Some(1.0));
    let comparison = first.baseline.as_ref().unwrap();
    let changes: Vec<Change> = comparison.questions.iter().map(|q| q.change).collect();
    assert_eq!(changes, [Change::Gained, Change::Unchanged]);
    assert_eq!(comparison.metrics.overall.recall_at_50.value, Some(0.5));
    assert!(
        first
            .to_string()
            .contains("gained 1 [\"roles\"], lost 0 [], unchanged 1")
    );

    // A retrieval results file is a baseline too; losing `sheet` from the corpus loses its recall and fails.
    let error = run_retrieval(
        retriever(vec![chunks[0].clone(), chunks[2].clone()]),
        args("second.json", Some(dir.join("first.json"))),
        None,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("lost recall"), "{error}");
    let second: RetrievalResults = read_json(&dir.join("second.json")).unwrap();
    let changes: Vec<Change> = second
        .baseline
        .unwrap()
        .questions
        .iter()
        .map(|q| q.change)
        .collect();
    assert_eq!(changes, [Change::Unchanged, Change::Lost]);

    // A selection must be in the baseline; this one is.
    run_retrieval(
        retriever(chunks),
        RetrievalArgs {
            questions: vec!["sheet".into()],
            ..args("third.json", Some(dir.join("first.json")))
        },
        None,
    )
    .await
    .unwrap();
    let third: RetrievalResults = read_json(&dir.join("third.json")).unwrap();
    assert_eq!(third.runs.len(), 1);

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn tune_command_reads_results_and_candidate_files() {
    let dir = std::env::temp_dir().join(format!("drew-eval-{}", RunId::new()));
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("results.json"), &tuning_results()).unwrap();
    fs::write(
        dir.join("candidates.json"),
        r#"[{"name": "strict", "thresholds": {"answer": 0.5}, "per_category": {"choice": {"answer": 0.3}}}]"#,
    )
    .unwrap();
    run_tune(&TuneArgs {
        results: dir.join("results.json"),
        candidates: dir.join("candidates.json"),
        out: Some(dir.join("tune.json")),
    })
    .unwrap();
    let report: tune::TuneReport = read_json(&dir.join("tune.json")).unwrap();
    assert_eq!(report.candidates[0].overall.answered, 4);
    fs::remove_dir_all(&dir).unwrap();
}
