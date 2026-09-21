use serde::{Deserialize, Serialize};

use crate::clients::json_or_status;
use crate::error::{RecallError, Result};
use crate::types::Stage;

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
    /// Must be the model Nexus indexed with. A different model produces vectors in a
    /// different space, and the search functions would return confident nonsense
    /// rather than an error.
    model: String,
    dim: usize,
    api_key: Option<String>,
}

impl EmbedderClient {
    pub fn new(
        http: reqwest::Client,
        base_url: String,
        model: String,
        dim: usize,
        api_key: Option<String>,
    ) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            dim,
            api_key,
        }
    }

    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut req = self
            .http
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&EmbedRequest {
                input: texts,
                model: &self.model,
            });
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await.map_err(|source| RecallError::Upstream {
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

        // Catch a dimension mismatch here rather than as an opaque Postgres error,
        // and more importantly rather than as plausible-looking bad results if the
        // dimensions happen to line up under the wrong model.
        if let Some(d) = parsed.data.first() {
            if d.embedding.len() != self.dim {
                return Err(RecallError::UpstreamStatus {
                    stage: Stage::Embed,
                    status: 502,
                    body: format!(
                        "embedder '{}' returned dim {} but Nexus indexed at {}",
                        self.model,
                        d.embedding.len(),
                        self.dim
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
