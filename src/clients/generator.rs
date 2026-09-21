use eventsource_stream::Eventsource;
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::clients::json_or_status;
use crate::error::{RecallError, Result};
use crate::types::{AdmittedCitations, Stage};

const SYSTEM_PROMPT: &str = "You are the user's own memory, answering in their voice.\n\
     Answer ONLY from the numbered memories provided below. They are the sole permitted \
     source of fact.\n\
     Cite the memory you used inline as [memory_N], matching its number.\n\
     If the memories do not contain the answer, say you do not have it. Never guess, \
     never draw on outside knowledge, and never invent a citation.";

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

/// Fence the admitted memories so the model cannot confuse them with instructions,
/// and number them so `[memory_N]` citations are checkable against the admitted set.
fn build_prompt(question: &str, admitted: &AdmittedCitations) -> Vec<Message> {
    let mut fenced = String::new();
    for (i, c) in admitted.as_slice().iter().enumerate() {
        let attribution = match &c.grantor_name {
            Some(g) => format!(" (shared with you by {g})"),
            None => String::new(),
        };
        fenced.push_str(&format!(
            "[memory_{i}]{attribution}\n```\n{}\n```\n\n",
            c.text.trim()
        ));
    }

    vec![
        Message {
            role: "system",
            content: format!("{SYSTEM_PROMPT}\n\nMemories:\n\n{fenced}"),
        },
        Message {
            role: "user",
            content: question.to_string(),
        },
    ]
}

#[derive(Clone)]
pub struct GeneratorClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl GeneratorClient {
    pub fn new(http: reqwest::Client, base_url: String, model: String) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
        }
    }

    fn request(&self, question: &str, admitted: &AdmittedCitations, stream: bool) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            messages: build_prompt(question, admitted),
            stream,
            temperature: 0.2,
            max_tokens: 512,
        }
    }

    /// Non-streaming, for `/v1/ask/sync`.
    pub async fn complete(
        &self,
        question: &str,
        admitted: &AdmittedCitations,
    ) -> Result<(String, serde_json::Value)> {
        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&self.request(question, admitted, false))
            .send()
            .await
            .map_err(|source| RecallError::Upstream {
                stage: Stage::Generate,
                source,
            })?;

        let parsed: ChatResponse = json_or_status(Stage::Generate, resp).await?;
        let text = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok((text, parsed.usage.unwrap_or_else(|| json!({}))))
    }

    /// Streaming, for `/v1/ask`. Yields token text as it arrives.
    ///
    /// Note there is deliberately no retry here. Once a token has reached the client a
    /// retry would be visible as duplicated output, which is why architecture.md
    /// specifies an `error` event after any tokens already sent.
    pub async fn stream(
        &self,
        question: &str,
        admitted: &AdmittedCitations,
    ) -> Result<BoxStream<'static, Result<String>>> {
        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&self.request(question, admitted, true))
            .send()
            .await
            .map_err(|source| RecallError::Upstream {
                stage: Stage::Generate,
                source,
            })?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(RecallError::UpstreamStatus {
                stage: Stage::Generate,
                status: status.as_u16(),
                body: body.chars().take(500).collect(),
            });
        }

        let stream = resp
            .bytes_stream()
            .eventsource()
            .map(|event| match event {
                Ok(ev) => {
                    if ev.data == "[DONE]" {
                        return Ok(None);
                    }
                    match serde_json::from_str::<ChatChunk>(&ev.data) {
                        Ok(chunk) => Ok(chunk
                            .choices
                            .into_iter()
                            .next()
                            .and_then(|c| c.delta.content)
                            .filter(|s| !s.is_empty())),
                        // A chunk we cannot parse is not worth killing the stream over.
                        Err(e) => {
                            tracing::debug!(error = %e, "skipping unparseable generator chunk");
                            Ok(None)
                        }
                    }
                }
                Err(e) => Err(RecallError::UpstreamStatus {
                    stage: Stage::Generate,
                    status: 502,
                    body: format!("generator stream broke: {e}"),
                }),
            })
            .filter_map(|r| async move {
                match r {
                    Ok(Some(text)) => Some(Ok(text)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(stream.boxed())
    }

    pub async fn healthy(&self) -> bool {
        match self.http.get(format!("{}/health", self.base_url)).send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }
}
