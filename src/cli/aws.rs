//! What `recall models check` prints about the ambient AWS settings both Bedrock adapters load.

use std::time::Duration;

use aws_config::{BehaviorVersion, SdkConfig, timeout::TimeoutConfig};
use aws_sdk_bedrockruntime::{
    config::{Credentials, ProvideCredentials},
    error::DisplayErrorContext,
};
use reqwest::Url;

use crate::llm::Endpoint;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Resolving credentials may query instance metadata.
const CREDENTIALS_TIMEOUT: Duration = Duration::from_secs(15);

/// The AWS region and the provider that resolves credentials, as both Bedrock adapters load them.
pub(super) async fn aws_environment() -> String {
    let config = ambient_config().await;
    let region = config
        .region()
        .map_or_else(|| "unset".to_owned(), ToString::to_string);
    let credentials = match config.credentials_provider() {
        None => "none configured".to_owned(),
        Some(provider) => {
            match tokio::time::timeout(CREDENTIALS_TIMEOUT, provider.provide_credentials()).await {
                Ok(Ok(credentials)) => format!("from {}", provider_name(&credentials)),
                Ok(Err(error)) => format!("none resolved: {}", DisplayErrorContext(error)),
                Err(_) => format!("none resolved within {CREDENTIALS_TIMEOUT:?}"),
            }
        }
    };
    format!("AWS region {region}, credentials {credentials}")
}

/// Where Bedrock Runtime sends a request for `path`, as both Bedrock adapters load it: `AWS_ENDPOINT_URL` when set,
/// otherwise the region's endpoint. `None` with no region, or an endpoint URL that does not parse.
pub(super) async fn bedrock_endpoint(path: &str) -> Option<Endpoint> {
    let config = ambient_config().await;
    let base: Url = match config.endpoint_url() {
        Some(url) => url.parse().ok()?,
        None => format!("https://bedrock-runtime.{}.amazonaws.com", config.region()?)
            .parse()
            .ok()?,
    };
    Some(Endpoint::new(&base, path))
}

async fn ambient_config() -> SdkConfig {
    aws_config::defaults(BehaviorVersion::latest())
        .timeout_config(
            TimeoutConfig::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .build(),
        )
        .load()
        .await
}

fn provider_name(credentials: &Credentials) -> String {
    // `Credentials` names its provider only in `Debug`, which redacts the secret; nothing else from it is used.
    let debug = format!("{credentials:?}");
    debug
        .split_once("provider_name: \"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .map_or_else(
            || "an unnamed provider".to_owned(),
            |(name, _)| format!("the {name} provider"),
        )
}
