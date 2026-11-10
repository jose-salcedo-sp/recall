//! OpenAI chat-completions wire types, plus the `drew` extension object.

use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Question, RunId},
    pipeline::{
        DeclineReason,
        report::{Citation, Outcome, RunReport, Verdict},
    },
    store::runs::status,
};

use super::error::ApiError;

/// The one model id `/v1/models` lists and every completion reports.
pub const MODEL: &str = "drew-ai";

/// `POST /v1/chat/completions` body. Fields Drew ignores (temperature, tools, and so on) are accepted and dropped.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// The conversation; only the last `user` message is used.
    pub messages: Vec<Message>,
    /// `true` returns the verified answer as server-sent events, in one content chunk.
    #[serde(default)]
    pub stream: Option<bool>,
}

/// One request message.
#[derive(Debug, Deserialize)]
pub struct Message {
    /// `user`, `system`, `assistant`, and so on.
    pub role: String,
    /// Missing or `null` counts as empty.
    #[serde(default)]
    pub content: Option<Content>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
/// Message content: a string or a list of parts.
pub enum Content {
    /// Plain text.
    Text(String),
    /// Content parts; only `text` parts are accepted, joined with newlines.
    Parts(Vec<Part>),
}

/// One content part.
#[derive(Debug, Deserialize)]
pub struct Part {
    /// The part's `type`; anything but `text` is rejected.
    #[serde(rename = "type")]
    pub kind: String,
    /// The text of a `text` part.
    #[serde(default)]
    pub text: Option<String>,
}

impl ChatRequest {
    /// The text of the last `user` message. System messages and earlier history are ignored.
    ///
    /// # Errors
    ///
    /// A `400` when there is no user message, it is blank, or it has a non-text part.
    pub fn question(&self) -> Result<Question, ApiError> {
        let message = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .ok_or_else(|| ApiError::bad_request("No user message in `messages`."))?;
        let text = match &message.content {
            None => String::new(),
            Some(Content::Text(text)) => text.clone(),
            Some(Content::Parts(parts)) => parts
                .iter()
                .map(|part| match (part.kind.as_str(), &part.text) {
                    ("text", Some(text)) => Ok(text.as_str()),
                    _ => Err(ApiError::bad_request(format!(
                        "Unsupported content part `{}`; only text parts are accepted.",
                        part.kind
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("\n"),
        };
        if text.trim().is_empty() {
            return Err(ApiError::bad_request("The last user message is empty."));
        }
        Ok(Question::new(text))
    }
}

/// Category, outcome, and citations for one run. The full trace is on `GET /v1/runs/{id}`.
#[derive(Debug, Serialize)]
pub struct Drew {
    /// For `GET /v1/runs/{id}`.
    pub run_id: RunId,
    /// What `Classify` chose; `None` when the run halted before classifying.
    pub category: Option<CategoryKind>,
    /// `answered`, `declined`, or `failed`.
    pub outcome: &'static str,
    /// The step that declined or failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub halted_at: Option<String>,
    /// Set for a decline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decline_reason: Option<DeclineReason>,
    /// Set for an answered decisive category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    /// Numbered as in the content: the first is `[1]`. For a decline, the ranked sources.
    pub citations: Vec<Citation>,
}

impl Drew {
    /// The public part of a report: no trace, scores or error detail.
    pub fn new(report: &RunReport) -> Self {
        let (halted_at, decline_reason, verdict, citations) = match &report.outcome {
            Outcome::Answered(answer) => {
                (None, None, answer.verdict.clone(), answer.citations.clone())
            }
            Outcome::Declined {
                at,
                reason,
                sources,
                ..
            } => (
                Some(at.clone()),
                Some(reason.clone()),
                None,
                sources.clone(),
            ),
            Outcome::Failed { at, .. } => (Some(at.clone()), None, None, vec![]),
        };
        Self {
            run_id: report.id,
            category: report.category,
            outcome: status(&report.outcome),
            halted_at,
            decline_reason,
            verdict,
            citations,
        }
    }
}

/// A non-streaming chat completion.
#[derive(Debug, Serialize)]
pub struct Completion {
    /// `chatcmpl-{run_id}`.
    pub id: String,
    /// Always `chat.completion`.
    pub object: &'static str,
    /// Unix seconds.
    pub created: i64,
    /// Always [`MODEL`].
    pub model: &'static str,
    /// The single answer.
    pub choices: [CompletionChoice; 1],
    /// Drew's extension object.
    pub drew: Drew,
}

/// The one choice of a [`Completion`].
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    /// Always `0`.
    pub index: u32,
    /// The rendered answer.
    pub message: AssistantMessage,
    /// Always `stop`.
    pub finish_reason: &'static str,
}

/// The answer message.
#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    /// Always `assistant`.
    pub role: &'static str,
    /// Markdown in the per-category format, with numbered citations.
    pub content: String,
}

impl Completion {
    /// `created` is Unix seconds.
    pub fn new(drew: Drew, content: String, created: i64) -> Self {
        Self {
            id: completion_id(drew.run_id),
            object: "chat.completion",
            created,
            model: MODEL,
            choices: [CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content,
                },
                finish_reason: "stop",
            }],
            drew,
        }
    }
}

/// One server-sent event of a streamed completion.
#[derive(Debug, Serialize)]
pub struct Chunk {
    /// `chatcmpl-{run_id}`, the same for every chunk of a completion.
    pub id: String,
    /// Always `chat.completion.chunk`.
    pub object: &'static str,
    /// Unix seconds.
    pub created: i64,
    /// Always [`MODEL`].
    pub model: &'static str,
    /// The single choice's delta.
    pub choices: [ChunkChoice; 1],
    /// Only on the final chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drew: Option<Drew>,
}

/// The one choice of a [`Chunk`].
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    /// Always `0`.
    pub index: u32,
    /// What this chunk adds.
    pub delta: Delta,
    /// `stop` on the final chunk, else `None`.
    pub finish_reason: Option<&'static str>,
}

/// The increment a [`Chunk`] carries; empty on the final chunk.
#[derive(Debug, Default, Serialize)]
pub struct Delta {
    /// `assistant` on the content chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// The whole rendered answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl Chunk {
    /// The whole verified answer in one content chunk, then a final chunk carrying `finish_reason` and `drew`.
    pub fn pair(drew: Drew, content: String, created: i64) -> [Self; 2] {
        let id = completion_id(drew.run_id);
        let make = |delta, finish_reason, drew| Self {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: MODEL,
            choices: [ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            drew,
        };
        [
            make(
                Delta {
                    role: Some("assistant"),
                    content: Some(content),
                },
                None,
                None,
            ),
            make(Delta::default(), Some("stop"), Some(drew)),
        ]
    }
}

fn completion_id(run_id: RunId) -> String {
    format!("chatcmpl-{run_id}")
}

/// `GET /v1/models` body.
#[derive(Debug, Serialize)]
pub struct ModelList {
    /// Always `list`.
    pub object: &'static str,
    /// The single model, [`MODEL`].
    pub data: [ModelCard; 1],
}

/// One listed model.
#[derive(Debug, Serialize)]
pub struct ModelCard {
    /// [`MODEL`].
    pub id: &'static str,
    /// Always `model`.
    pub object: &'static str,
    /// Always `0`; the model has no creation date.
    pub created: i64,
    /// Always `drew`.
    pub owned_by: &'static str,
}

impl ModelList {
    /// The list with Drew's one model.
    pub fn drew() -> Self {
        Self {
            object: "list",
            data: [ModelCard {
                id: MODEL,
                object: "model",
                created: 0,
                owned_by: "drew",
            }],
        }
    }
}
