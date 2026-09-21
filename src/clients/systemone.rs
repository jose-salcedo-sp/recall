use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::clients::json_or_status;
use crate::error::{RecallError, Result};
use crate::types::{Candidate, Stage};

/// The admission instruction. This wording is the product: it asks for sufficiency,
/// not topical relevance, which is the distinction retrieval cannot make.
const NOUL_INSTRUCTIONS: &str = "This candidate is sufficient evidence to answer the question. \
     It is not merely on a related topic.";

#[derive(Serialize)]
struct SystemOneRequest {
    model: String,
    state: serde_json::Value,
    questions: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct SystemOneResponse {
    answers: HashMap<String, Answer>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    calibration: Option<Calibration>,
}

#[derive(Deserialize)]
struct Answer {
    noul: f64,
}

#[derive(Deserialize, Debug)]
struct Calibration {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    fitted: Option<bool>,
}

#[derive(Clone)]
pub struct SystemOneClient {
    http: reqwest::Client,
    base_url: String,
}

impl SystemOneClient {
    pub fn new(http: reqwest::Client, base_url: String) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Score every candidate in ONE call.
    ///
    /// plan.md Finding 3: the wave splitting and `admit_concurrency` in architecture.md
    /// existed to work around hosted Jev's per-call price and state budget. A local
    /// encoder batches these internally, so there is one request and no merge step.
    pub async fn score(
        &self,
        question: &str,
        as_of: chrono::DateTime<chrono::Utc>,
        candidates: &[Candidate],
    ) -> Result<HashMap<String, f64>> {
        if candidates.is_empty() {
            return Ok(HashMap::new());
        }

        // Positional keys keep the payload small and avoid leaking uuids to the model.
        let keyed: Vec<(String, &Candidate)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (format!("m{i}"), c))
            .collect();

        let state = json!({
            "question": question,
            "as_of": as_of.to_rfc3339(),
            "candidates": keyed.iter().map(|(k, c)| json!({
                "id": k,
                "origin": c.origin,
                "grantor": c.grantor_name,
                "text": c.text,
            })).collect::<Vec<_>>(),
        });

        let questions = keyed
            .iter()
            .map(|(k, _)| {
                (
                    k.clone(),
                    json!({ "type": "noul", "instructions": NOUL_INSTRUCTIONS }),
                )
            })
            .collect();

        let resp = self
            .http
            .post(format!("{}/v1/systemone", self.base_url))
            .json(&SystemOneRequest {
                model: "recall-systemone".into(),
                state,
                questions,
            })
            .send()
            .await
            .map_err(|source| RecallError::Upstream {
                stage: Stage::Admit,
                source,
            })?;

        let parsed: SystemOneResponse = json_or_status(Stage::Admit, resp).await?;

        if let Some(c) = &parsed.calibration {
            if c.fitted == Some(false) {
                tracing::debug!(
                    temperature = ?c.temperature,
                    backend = ?parsed.backend,
                    "system one reports an unfitted temperature; noul values are not calibrated"
                );
            }
        }

        // Map positional keys back onto chunk uuids.
        let mut out = HashMap::with_capacity(keyed.len());
        for (k, c) in &keyed {
            match parsed.answers.get(k) {
                Some(a) => {
                    out.insert(c.id.to_string(), a.noul.clamp(0.0, 1.0));
                }
                None => {
                    return Err(RecallError::UpstreamStatus {
                        stage: Stage::Admit,
                        status: 502,
                        body: format!("system one omitted an answer for {k}"),
                    })
                }
            }
        }
        Ok(out)
    }

    pub async fn healthy(&self) -> bool {
        match self
            .http
            .get(format!("{}/healthz", self.base_url))
            .send()
            .await
        {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }
}
