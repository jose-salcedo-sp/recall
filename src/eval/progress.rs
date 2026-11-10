//! Live feedback for `drew eval run`: one line per finished question, above a progress bar.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::pipeline::{DeclineReason, report::Outcome};

use super::{
    golden::GoldenQuestion,
    metrics::{EvalRun, Progress, scope_name},
};

const DETAIL_CHARS: usize = 100;

/// The bar draws to stderr and hides itself when stderr is not a terminal; the lines always print to stdout.
pub struct Bar {
    bar: ProgressBar,
    total: usize,
    done: AtomicUsize,
    running: Mutex<Vec<String>>,
}

impl Bar {
    /// Prints the starting line and starts the bar's spinner.
    pub fn new(total: usize, concurrency: usize) -> Self {
        let bar = ProgressBar::new(total as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.cyan} [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} {msg}",
            )
            .expect("the progress template is a valid constant")
            .progress_chars("=> "),
        );
        bar.enable_steady_tick(Duration::from_millis(120));
        bar.suspend(|| {
            println!(
                "{} {total} golden questions, {concurrency} at a time",
                style("    Starting").green().bold()
            )
        });
        Self {
            bar,
            total,
            done: AtomicUsize::new(0),
            running: Mutex::new(Vec::new()),
        }
    }

    /// Clears the bar; the per-question lines stay.
    pub fn finish(&self) {
        self.bar.finish_and_clear();
    }

    fn show_running(&self, running: &[String]) {
        self.bar.set_message(if running.is_empty() {
            String::new()
        } else {
            format!("running {}", running.join(", "))
        });
    }
}

impl Progress for Bar {
    fn started(&self, golden: &GoldenQuestion) {
        let mut running = self
            .running
            .lock()
            .expect("progress lock poisoned: a reporter panicked while holding it");
        running.push(golden.id.clone());
        self.show_running(&running);
    }

    fn finished(&self, run: &EvalRun, elapsed: Duration) {
        let position = self.done.fetch_add(1, Ordering::SeqCst) + 1;
        let line = line(run, elapsed, position, self.total);
        self.bar.suspend(|| println!("{line}"));
        self.bar.inc(1);
        let mut running = self
            .running
            .lock()
            .expect("progress lock poisoned: a reporter panicked while holding it");
        running.retain(|id| *id != run.golden.id);
        self.show_running(&running);
    }
}

/// `    ANSWERED [  41.2s] ( 3/58) daas-q13 boolean  classify ✓  hit@5 ✗  recall 50%  answer ✗`
pub fn line(run: &EvalRun, elapsed: Duration, position: usize, total: usize) -> String {
    let status = match &run.report.outcome {
        Outcome::Answered(_) => style(format!("{:>12}", "ANSWERED")).green().bold(),
        Outcome::Declined { .. } => style(format!("{:>12}", "DECLINED")).yellow().bold(),
        Outcome::Failed { .. } => style(format!("{:>12}", "FAILED")).red().bold(),
    };
    let width = total.to_string().len();
    let scores = &run.scores;
    let mut line = format!(
        "{status} [{:>7.1}s] ({position:>width$}/{total}) {} {:<8} classify {}  hit@5 {}  recall {}  answer {}",
        elapsed.as_secs_f64(),
        style(&run.golden.id).bold(),
        scope_name(Some(run.golden.expected.category())),
        mark(scores.classified_correctly),
        mark(scores.hit_at_5),
        scores
            .recall_at_50
            .map_or_else(|| "-".to_owned(), |r| format!("{:.0}%", r * 100.0)),
        mark(scores.answer_correct),
    );
    let detail = match &run.report.outcome {
        Outcome::Answered(_) => None,
        Outcome::Declined { at, reason, .. } => Some(format!("at {at}: {}", decline(reason))),
        Outcome::Failed { at, error } => Some(format!("at {at}: {error}")),
    };
    if let Some(detail) = detail {
        line.push_str(&format!("\n{:>14}{}", "", style(truncate(&detail)).dim()));
    }
    line
}

fn mark(value: Option<bool>) -> console::StyledObject<&'static str> {
    match value {
        Some(true) => style("✓").green(),
        Some(false) => style("✗").red(),
        None => style("-").dim(),
    }
}

fn decline(reason: &DeclineReason) -> String {
    match reason {
        DeclineReason::LowConfidence => "low confidence".into(),
        DeclineReason::InvalidModelOutput(detail) => format!("invalid model output: {detail}"),
        DeclineReason::NoSources => "no sources".into(),
        DeclineReason::Unverified(detail) => format!("unverified: {detail}"),
        DeclineReason::ContradictsVerdict => "the sources contradict the verdict".into(),
    }
}

fn truncate(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= DETAIL_CHARS {
        return flat;
    }
    let mut cut: String = flat.chars().take(DETAIL_CHARS).collect();
    cut.push('…');
    cut
}
