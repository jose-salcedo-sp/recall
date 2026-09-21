pub mod embedder;
pub mod generator;
pub mod systemone;

use crate::error::{RecallError, Result};
use crate::types::Stage;

/// Read an upstream response body as the typed value, turning a non-2xx into an
/// `UpstreamStatus` so the transient/permanent classification in `RecallError` can
/// see the status code.
pub(crate) async fn json_or_status<T: serde::de::DeserializeOwned>(
    stage: Stage,
    resp: reqwest::Response,
) -> Result<T> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(RecallError::UpstreamStatus {
            stage,
            status: status.as_u16(),
            body: body.chars().take(500).collect(),
        });
    }
    resp.json::<T>()
        .await
        .map_err(|source| RecallError::Upstream { stage, source })
}
