use aws_sdk_bedrockruntime::{
    Client, Config,
    config::{BehaviorVersion, Credentials, Region},
};
use wiremock::MockServer;

/// A Bedrock runtime client that sends every call to `server`, with fixed test credentials.
pub fn client(server: &MockServer) -> Client {
    Client::from_conf(
        Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new("AKIDTEST", "secret", None, None, "test"))
            .endpoint_url(server.uri())
            .build(),
    )
}
