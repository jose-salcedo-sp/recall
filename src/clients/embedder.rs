use serde::{Deserialize, Serialize};

use crate::clients::json_or_status;
use crate::error::{RecallError, Result};
use crate::types::Stage;

pub const EMBEDDING_DIM: usize = 768;

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

#[derive(Clone)]
pub struct EmbedderClient {
    http: reqwest::Client,
    base_url: String,
}

impl EmbedderClient {
    pub fn new(http: reqwest::Client, base_url: String) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let resp = self
            .http
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&EmbedRequest {
                input: texts,
                model: "recall-embed",
            })
            .send()
            .await
            .map_err(|source| RecallError::Upstream {
                stage: Stage::Embed,
                source,
            })?;

        let parsed: EmbedResponse = json_or_status(Stage::Embed, resp).await?;

        if parsed.data.len() != texts.len() {
            return Err(RecallError::UpstreamStatus {
                stage: Stage::Embed,
                status: 502,
                body: format!(
                    "embedder returned {} vectors for {} inputs",
                    parsed.data.len(),
                    texts.len()
                ),
            });
        }

        // A dimension mismatch here would otherwise surface much later as an opaque
        // Postgres error, since the column is declared vector(768).
        if let Some(d) = parsed.data.first() {
            if d.embedding.len() != EMBEDDING_DIM {
                return Err(RecallError::UpstreamStatus {
                    stage: Stage::Embed,
                    status: 502,
                    body: format!(
                        "embedder returned dim {} but the index expects {EMBEDDING_DIM}",
                        d.embedding.len()
                    ),
                });
            }
        }

        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(&[text.to_string()]).await?;
        v.pop().ok_or(RecallError::UpstreamStatus {
            stage: Stage::Embed,
            status: 502,
            body: "embedder returned no vectors".into(),
        })
    }

    pub async fn healthy(&self) -> bool {
        self.embed_one("ping").await.is_ok()
    }
}
