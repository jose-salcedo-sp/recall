//! Golden set: JSONL, one verified question per line.

use std::collections::HashSet;

use eros::IntoUnion;
use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, ChunkId, Question},
    pipeline::report::Verdict,
};

/// One verified question with its expected category, answer and relevant chunks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoldenQuestion {
    /// Unique within the set; selects the question with `--question`.
    pub id: String,
    /// Hand-written placeholder, not mined from a real thread.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub synthetic: bool,
    /// Asked verbatim.
    pub question: Question,
    /// Serialized as `category` and `expected` keys.
    #[serde(flatten)]
    pub expected: Expected,
    /// Chunks a correct retrieval should surface; recall@50 and hit@5 are measured against these.
    pub relevant_chunks: Vec<ChunkId>,
    /// The Slack thread (or page) the question was mined from.
    pub source_thread: String,
}

/// The expected category and, for decisive categories, the expected answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "category", content = "expected", rename_all = "snake_case")]
pub enum Expected {
    /// The expected yes or no.
    Boolean(YesNo),
    /// The correct option text.
    Choice(String),
    /// Item texts in the expected order.
    Ranking(Vec<String>),
    /// No verdict to grade; only classification, retrieval and citations are scored.
    Open,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// A boolean golden answer, spelled `yes` or `no` in the file.
pub enum YesNo {
    /// Serialized `yes`.
    Yes,
    /// Serialized `no`.
    No,
}

impl Expected {
    /// The category the question should be classified as.
    pub fn category(&self) -> CategoryKind {
        match self {
            Self::Boolean(_) => CategoryKind::Boolean,
            Self::Choice(_) => CategoryKind::Choice,
            Self::Ranking(_) => CategoryKind::Ranking,
            Self::Open => CategoryKind::Open,
        }
    }

    /// Whether `verdict` is the expected answer, comparing option and item texts case-insensitively after
    /// trimming. `None` when there is nothing to grade automatically (`open`).
    pub fn matches(&self, verdict: Option<&Verdict>) -> Option<bool> {
        let matched = match (self, verdict) {
            (Self::Open, _) => return None,
            (Self::Boolean(expected), Some(Verdict::Boolean { yes, .. })) => {
                *yes == (*expected == YesNo::Yes)
            }
            (Self::Choice(expected), Some(Verdict::Choice { selected, .. })) => {
                same_text(expected, &selected.value)
            }
            (Self::Ranking(expected), Some(Verdict::Ranking { items, .. })) => {
                expected.len() == items.len()
                    && expected
                        .iter()
                        .zip(items)
                        .all(|(expected, item)| same_text(expected, &item.value))
            }
            _ => false,
        };
        Some(matched)
    }
}

fn same_text(a: &str, b: &str) -> bool {
    a.trim().to_lowercase() == b.trim().to_lowercase()
}

/// A golden-set line that does not parse or fails validation.
#[derive(Debug, thiserror::Error)]
#[error("line {line}: {message}")]
pub struct GoldenError {
    /// 1-based line number.
    pub line: usize,
    /// What is wrong with it.
    pub message: String,
}

/// Parses a JSONL golden set. Blank lines are skipped; line numbers in errors are 1-based.
///
/// # Errors
///
/// [`GoldenError`] for the first line that is not valid JSON, has a blank id or question, a blank choice option,
/// a ranking of fewer than 2 items, or an id already used.
pub fn parse(text: &str) -> eros::Result<Vec<GoldenQuestion>, (GoldenError,)> {
    let mut questions = Vec::new();
    let mut ids = HashSet::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        if raw.trim().is_empty() {
            continue;
        }
        let error = |message: String| GoldenError { line, message };
        let question: GoldenQuestion = serde_json::from_str(raw)
            .map_err(|e| error(e.to_string()))
            .union()?;
        validate(&question).map_err(error).union()?;
        if !ids.insert(question.id.clone()) {
            return Err(error(format!("duplicate id `{}`", question.id))).union();
        }
        questions.push(question);
    }
    Ok(questions)
}

fn validate(question: &GoldenQuestion) -> Result<(), String> {
    if question.id.trim().is_empty() {
        return Err("empty id".into());
    }
    if question.question.as_str().trim().is_empty() {
        return Err("empty question".into());
    }
    match &question.expected {
        Expected::Choice(option) if option.trim().is_empty() => {
            Err("choice expects a non-empty option".into())
        }
        Expected::Ranking(items) if items.len() < 2 => {
            Err("ranking expects at least 2 items".into())
        }
        _ => Ok(()),
    }
}
