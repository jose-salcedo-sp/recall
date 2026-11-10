//! `Embedder` over Bedrock (Titan Text Embeddings v2 or Cohere Embed v3, 1024 dimensions).

use std::time::Duration;

use async_trait::async_trait;
use aws_config::{BehaviorVersion, timeout::TimeoutConfig};
use aws_sdk_bedrockruntime::{Client, error::DisplayErrorContext, primitives::Blob};
use eros::{Context, bail};
use futures::{StreamExt, TryStreamExt, stream};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef};

/// Bedrock rejects longer `inputText`; Titan v2 also caps input at 8,192 tokens, which chunking must respect.
const TITAN_MAX_CHARS: usize = 50_000;
/// Bedrock rejects longer texts; token overflow below this is truncated from the end (`truncate: END`).
const COHERE_MAX_CHARS: usize = 2_048;
const COHERE_MAX_BATCH: usize = 96;
/// Titan embeds one text per call, so calls run concurrently.
const CONCURRENT_CALLS: usize = 8;
/// One call embeds at most a `COHERE_MAX_BATCH` batch; this bounds it including its retries.
const OPERATION_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The Bedrock embedding models whose request and response shapes this adapter knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Titan Text Embeddings v2: one text per call, 1024 dimensions, normalized.
    TitanV2,
    /// Cohere Embed v3: up to 96 texts per call; distinguishes documents from queries.
    CohereV3,
}

impl Family {
    /// Accepts plain model ids and cross-region inference profile ids (`us.cohere.embed-english-v3`).
    ///
    /// # Errors
    ///
    /// When the id names neither family.
    pub fn of(model_id: &str) -> eros::Result<Self> {
        if model_id.contains("amazon.titan-embed-text-v2") {
            Ok(Self::TitanV2)
        } else if model_id.contains("cohere.embed-") && model_id.contains("-v3") {
            Ok(Self::CohereV3)
        } else {
            bail!(
                "unsupported embedding model {model_id:?}: expected Titan Text Embeddings v2 or Cohere Embed v3"
            )
        }
    }

    /// Request parameters that shape the vectors, recorded as the model version.
    fn profile(self) -> &'static str {
        match self {
            Self::TitanV2 => "dimensions=1024;normalize=true",
            Self::CohereV3 => "truncate=END",
        }
    }
}

/// One `InvokeModel` call: its JSON body and how many embeddings it returns.
#[derive(Debug, PartialEq)]
pub struct Request {
    /// The `InvokeModel` body.
    pub body: Value,
    /// Texts in this call.
    pub count: usize,
}

/// Splits `texts` into calls in input order, truncating each text to the family's character limit.
pub fn requests(family: Family, texts: &[String], purpose: EmbedPurpose) -> Vec<Request> {
    match family {
        Family::TitanV2 => texts
            .iter()
            .map(|text| Request {
                body: json!({
                    "inputText": truncate(text, TITAN_MAX_CHARS),
                    "dimensions": EMBEDDING_DIM,
                    "normalize": true,
                }),
                count: 1,
            })
            .collect(),
        Family::CohereV3 => {
            let input_type = match purpose {
                EmbedPurpose::Document => "search_document",
                EmbedPurpose::Query => "search_query",
            };
            texts
                .chunks(COHERE_MAX_BATCH)
                .map(|batch| Request {
                    body: json!({
                        "texts": batch
                            .iter()
                            .map(|text| truncate(text, COHERE_MAX_CHARS))
                            .collect::<Vec<_>>(),
                        "input_type": input_type,
                        "truncate": "END",
                    }),
                    count: batch.len(),
                })
                .collect()
        }
    }
}

/// The vectors of one call's response body.
///
/// # Errors
///
/// When the body does not parse, or holds other than `count` vectors of [`EMBEDDING_DIM`].
pub fn parse_response(family: Family, body: &[u8], count: usize) -> eros::Result<Vec<Vec<f32>>> {
    #[derive(Deserialize)]
    struct Titan {
        embedding: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct Cohere {
        embeddings: Vec<Vec<f32>>,
    }

    let embeddings = match family {
        Family::TitanV2 => vec![
            serde_json::from_slice::<Titan>(body)
                .context("invalid Titan embedding response")?
                .embedding,
        ],
        Family::CohereV3 => {
            serde_json::from_slice::<Cohere>(body)
                .context("invalid Cohere embedding response")?
                .embeddings
        }
    };
    if embeddings.len() != count {
        bail!("expected {count} embeddings, got {}", embeddings.len());
    }
    if let Some(bad) = embeddings.iter().find(|e| e.len() != EMBEDDING_DIM) {
        bail!(
            "expected {EMBEDDING_DIM}-dimensional embeddings, got {}",
            bad.len()
        );
    }
    Ok(embeddings)
}

fn truncate(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

/// [`Embedder`] over `InvokeModel`, with up to 8 calls in flight. Its [`ModelRef::version`] records the
/// parameters that shape the vectors.
pub struct BedrockEmbedder {
    client: Client,
    model_id: String,
    family: Family,
}

impl BedrockEmbedder {
    /// # Errors
    ///
    /// When `model_id` names no supported [`Family`].
    pub fn new(client: Client, model_id: impl Into<String>) -> eros::Result<Self> {
        let model_id = model_id.into();
        Ok(Self {
            family: Family::of(&model_id)?,
            client,
            model_id,
        })
    }

    /// Region and credentials come from the standard AWS environment and profile chain.
    ///
    /// # Errors
    ///
    /// As [`BedrockEmbedder::new`].
    pub async fn load(model_id: &str) -> eros::Result<Self> {
        let timeouts = TimeoutConfig::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .operation_timeout(OPERATION_TIMEOUT)
            .build();
        let config = aws_config::defaults(BehaviorVersion::latest())
            .timeout_config(timeouts)
            .load()
            .await;
        Self::new(Client::new(&config), model_id)
    }

    async fn invoke(&self, request: Request) -> eros::Result<Vec<Vec<f32>>> {
        let output = self
            .client
            .invoke_model()
            .model_id(&self.model_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(serde_json::to_vec(&request.body)?))
            .send()
            .await
            .map_err(|e| {
                eros::error!(
                    "Bedrock InvokeModel {} failed: {}",
                    self.model_id,
                    DisplayErrorContext(e)
                )
            })?;
        parse_response(self.family, output.body.as_ref(), request.count)
    }
}

#[async_trait]
impl Embedder for BedrockEmbedder {
    fn model(&self) -> ModelRef {
        ModelRef::new(&self.model_id, self.family.profile())
    }

    async fn embed(&self, texts: &[String], purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        let batches: Vec<Vec<Vec<f32>>> = stream::iter(
            requests(self.family, texts, purpose)
                .into_iter()
                .map(|r| self.invoke(r)),
        )
        .buffered(CONCURRENT_CALLS)
        .try_collect()
        .await?;
        Ok(batches.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("text {i}")).collect()
    }

    fn vector_json(n: usize) -> Vec<Vec<f32>> {
        vec![vec![0.5; EMBEDDING_DIM]; n]
    }

    #[test]
    fn detects_supported_families() {
        assert_eq!(
            Family::of("amazon.titan-embed-text-v2:0").unwrap(),
            Family::TitanV2
        );
        assert_eq!(
            Family::of("cohere.embed-english-v3").unwrap(),
            Family::CohereV3
        );
        assert_eq!(
            Family::of("us.cohere.embed-multilingual-v3").unwrap(),
            Family::CohereV3
        );
        assert!(Family::of("amazon.titan-embed-text-v1").is_err());
        assert!(Family::of("cohere.embed-v4:0").is_err());
    }

    #[test]
    fn titan_sends_one_text_per_call_at_1024_dimensions_normalized() {
        let requests = requests(Family::TitanV2, &texts(2), EmbedPurpose::Document);
        assert_eq!(
            requests,
            vec![
                Request {
                    body: json!({"inputText": "text 0", "dimensions": 1024, "normalize": true}),
                    count: 1,
                },
                Request {
                    body: json!({"inputText": "text 1", "dimensions": 1024, "normalize": true}),
                    count: 1,
                },
            ]
        );
    }

    #[test]
    fn cohere_batches_at_most_96_texts_with_the_purpose_as_input_type() {
        let documents = requests(Family::CohereV3, &texts(200), EmbedPurpose::Document);
        assert_eq!(
            documents.iter().map(|r| r.count).collect::<Vec<_>>(),
            [96, 96, 8]
        );
        assert_eq!(documents[0].body["input_type"], "search_document");
        assert_eq!(documents[0].body["truncate"], "END");
        assert_eq!(documents[2].body["texts"][7], "text 199");

        let query = requests(Family::CohereV3, &texts(1), EmbedPurpose::Query);
        assert_eq!(
            query,
            vec![Request {
                body: json!({"texts": ["text 0"], "input_type": "search_query", "truncate": "END"}),
                count: 1,
            }]
        );
    }

    #[test]
    fn overlong_texts_are_cut_on_a_char_boundary() {
        let long = "é".repeat(COHERE_MAX_CHARS + 10);
        let requests = requests(Family::CohereV3, &[long], EmbedPurpose::Document);
        let sent = requests[0].body["texts"][0].as_str().unwrap();
        assert_eq!(sent.chars().count(), COHERE_MAX_CHARS);
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn parses_titan_response() {
        let body = json!({"embedding": vector_json(1)[0], "inputTextTokenCount": 3});
        let parsed =
            parse_response(Family::TitanV2, &serde_json::to_vec(&body).unwrap(), 1).unwrap();
        assert_eq!(parsed, vector_json(1));
    }

    #[test]
    fn parses_cohere_response() {
        let body = json!({
            "id": "abc",
            "embeddings": vector_json(2),
            "texts": ["a", "b"],
            "response_type": "embeddings_floats",
        });
        let parsed =
            parse_response(Family::CohereV3, &serde_json::to_vec(&body).unwrap(), 2).unwrap();
        assert_eq!(parsed, vector_json(2));
    }

    #[test]
    fn rejects_wrong_count_or_dimension() {
        let two = serde_json::to_vec(&json!({"embeddings": vector_json(2)})).unwrap();
        assert!(parse_response(Family::CohereV3, &two, 3).is_err());

        let short = serde_json::to_vec(&json!({"embedding": [0.1, 0.2]})).unwrap();
        assert!(parse_response(Family::TitanV2, &short, 1).is_err());

        assert!(parse_response(Family::TitanV2, b"not json", 1).is_err());
    }
}
