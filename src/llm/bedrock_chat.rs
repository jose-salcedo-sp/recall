//! `ChatModel` over the Bedrock Converse API, used by `Explain`.

use std::time::Duration;

use async_trait::async_trait;
use aws_config::{BehaviorVersion, SdkConfig, timeout::TimeoutConfig};
use aws_sdk_bedrockruntime::{
    Client,
    config::{Credentials, ProvideCredentials},
    error::DisplayErrorContext,
    operation::converse::ConverseOutput,
    types::{
        ContentBlock, ConversationRole, ConverseOutput as Reply, InferenceConfiguration, Message,
        SpecificToolChoice, StopReason, SystemContentBlock, Tool, ToolChoice, ToolConfiguration,
        ToolInputSchema, ToolSpecification,
    },
};
use aws_smithy_types::{Document, Number};
use eros::Context;
use reqwest::Url;
use serde_json::Value;

use super::{ChatModel, ChatRequest, ChatResponse, ModelRef, backend::Endpoint};

/// The tool the model is forced to call when a request has a `json_schema`; its input is the response.
const RESPOND_TOOL: &str = "respond";
/// A long structured answer can take minutes to generate; this bounds the call including its retries.
const OPERATION_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Resolving credentials may query instance metadata.
const CREDENTIALS_TIMEOUT: Duration = Duration::from_secs(15);

/// [`ChatModel`] over the Converse API. A request with a `json_schema` forces a tool call whose input is the reply.
pub struct BedrockChat {
    client: Client,
    model_id: String,
}

impl BedrockChat {
    /// `model_id` is a Bedrock model or inference profile id; it already carries its version.
    pub fn new(client: Client, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
        }
    }

    /// Region and credentials come from the standard AWS environment and profile chain.
    pub async fn load(model_id: &str) -> Self {
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
}

/// The AWS region and the provider that resolves credentials, as both Bedrock adapters load them.
pub(crate) async fn aws_environment() -> String {
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
pub(crate) async fn bedrock_endpoint(path: &str) -> Option<Endpoint> {
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

#[async_trait]
impl ChatModel for BedrockChat {
    fn model(&self) -> ModelRef {
        // Bedrock model ids already carry their version (`...-v1:0`).
        ModelRef::new(&self.model_id, &self.model_id)
    }

    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse> {
        let converse = converse_request(&request)?;
        let output = self
            .client
            .converse()
            .model_id(&self.model_id)
            .set_system(Some(converse.system))
            .set_messages(Some(converse.messages))
            .inference_config(converse.inference)
            .set_tool_config(converse.tools)
            .send()
            .await
            .map_err(|error| {
                eros::error!(
                    "Bedrock Converse with {} failed: {}",
                    self.model_id,
                    DisplayErrorContext(error)
                )
            })?;
        chat_response(output, &request)
    }
}

/// The parts of a Converse call that come from the `ChatRequest`.
#[derive(Debug)]
struct Converse {
    system: Vec<SystemContentBlock>,
    messages: Vec<Message>,
    inference: InferenceConfiguration,
    tools: Option<ToolConfiguration>,
}

fn converse_request(request: &ChatRequest) -> eros::Result<Converse> {
    let message = Message::builder()
        .role(ConversationRole::User)
        .content(ContentBlock::Text(request.user.clone()))
        .build()?;
    let inference = InferenceConfiguration::builder()
        .max_tokens(i32::try_from(request.max_tokens).context("max_tokens does not fit in i32")?)
        .temperature(request.temperature)
        .build();
    let tools = request.json_schema.as_ref().map(respond_tool).transpose()?;
    Ok(Converse {
        system: vec![SystemContentBlock::Text(request.system.clone())],
        messages: vec![message],
        inference,
        tools,
    })
}

fn respond_tool(schema: &Value) -> eros::Result<ToolConfiguration> {
    let spec = ToolSpecification::builder()
        .name(RESPOND_TOOL)
        .description("Return the response. The input must match the schema.")
        .input_schema(ToolInputSchema::Json(to_document(schema)))
        .build()?;
    let choice = SpecificToolChoice::builder().name(RESPOND_TOOL).build()?;
    Ok(ToolConfiguration::builder()
        .tools(Tool::ToolSpec(spec))
        .tool_choice(ToolChoice::Tool(choice))
        .build()?)
}

/// # Errors
///
/// When the reply stopped at `max_tokens`, even with partial content, or holds no message.
#[eros::context("reading the Converse reply")]
fn chat_response(output: ConverseOutput, request: &ChatRequest) -> eros::Result<ChatResponse> {
    eros::ensure!(
        output.stop_reason != StopReason::MaxTokens,
        "answer cut off at max_tokens={} (stop reason max_tokens); on a reasoning model max_tokens also bounds \
         the reasoning",
        request.max_tokens
    );
    let structured = request.json_schema.is_some();
    let Some(Reply::Message(message)) = output.output else {
        eros::bail!(
            "Bedrock returned no message (stop reason {})",
            output.stop_reason.as_str()
        );
    };
    let tool_input = message.content.iter().find_map(|block| match block {
        ContentBlock::ToolUse(tool) if structured && tool.name == RESPOND_TOOL => Some(&tool.input),
        _ => None,
    });
    let text = match tool_input {
        Some(input) => serde_json::to_string(&from_document(input))?,
        // Without the tool call the text is still returned, so the caller can trace it as invalid output.
        None => message
            .content
            .iter()
            .filter_map(|block| block.as_text().ok())
            .map(String::as_str)
            .collect(),
    };
    Ok(ChatResponse {
        text,
        mean_logprob: None,
    })
}

fn to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(Number::NegInt(i))
            } else {
                n.as_f64()
                    .map_or(Document::Null, |f| Document::Number(Number::Float(f)))
            }
        }
        Value::String(s) => Document::String(s.clone()),
        Value::Array(items) => Document::Array(items.iter().map(to_document).collect()),
        Value::Object(fields) => Document::Object(
            fields
                .iter()
                .map(|(key, value)| (key.clone(), to_document(value)))
                .collect(),
        ),
    }
}

fn from_document(document: &Document) -> Value {
    match document {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::Number(Number::PosInt(u)) => Value::from(*u),
        Document::Number(Number::NegInt(i)) => Value::from(*i),
        Document::Number(Number::Float(f)) => {
            serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number)
        }
        Document::String(s) => Value::String(s.clone()),
        Document::Array(items) => Value::Array(items.iter().map(from_document).collect()),
        Document::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, value)| (key.clone(), from_document(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::types::{ConverseMetrics, StopReason, TokenUsage, ToolUseBlock};
    use serde_json::json;

    use super::*;

    fn request(json_schema: Option<Value>) -> ChatRequest {
        ChatRequest {
            system: "be terse".into(),
            user: "<question>q</question>".into(),
            max_tokens: 512,
            temperature: 0.0,
            json_schema,
        }
    }

    fn output(stop_reason: StopReason, content: Vec<ContentBlock>) -> ConverseOutput {
        let message = Message::builder()
            .role(ConversationRole::Assistant)
            .set_content(Some(content))
            .build()
            .unwrap();
        ConverseOutput::builder()
            .output(Reply::Message(message))
            .stop_reason(stop_reason)
            .usage(
                TokenUsage::builder()
                    .input_tokens(10)
                    .output_tokens(5)
                    .total_tokens(15)
                    .build()
                    .unwrap(),
            )
            .metrics(ConverseMetrics::builder().latency_ms(42).build().unwrap())
            .build()
            .unwrap()
    }

    fn tool_use(name: &str, input: &Value) -> ContentBlock {
        ContentBlock::ToolUse(
            ToolUseBlock::builder()
                .tool_use_id("t1")
                .name(name)
                .input(to_document(input))
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn request_maps_system_user_and_inference() {
        let converse = converse_request(&request(None)).unwrap();

        assert_eq!(
            converse.system,
            [SystemContentBlock::Text("be terse".into())]
        );
        let [message] = converse.messages.as_slice() else {
            panic!("one message");
        };
        assert_eq!(message.role, ConversationRole::User);
        assert_eq!(
            message.content,
            [ContentBlock::Text("<question>q</question>".into())]
        );
        assert_eq!(converse.inference.max_tokens, Some(512));
        assert_eq!(converse.inference.temperature, Some(0.0));
        assert_eq!(converse.tools, None);
    }

    #[test]
    fn json_schema_forces_the_respond_tool() {
        let schema = json!({ "type": "object", "properties": { "claims": { "type": "array" } } });

        let tools = converse_request(&request(Some(schema.clone())))
            .unwrap()
            .tools
            .unwrap();

        let [Tool::ToolSpec(spec)] = tools.tools.as_slice() else {
            panic!("one tool spec");
        };
        assert_eq!(spec.name, RESPOND_TOOL);
        assert_eq!(
            spec.input_schema,
            Some(ToolInputSchema::Json(to_document(&schema)))
        );
        assert!(matches!(
            tools.tool_choice,
            Some(ToolChoice::Tool(ref choice)) if choice.name == RESPOND_TOOL
        ));
    }

    #[test]
    fn structured_response_is_the_tool_input_as_json() {
        let input = json!({ "claims": [{ "text": "A", "chunk_ids": ["x"] }] });
        let output = output(
            StopReason::ToolUse,
            vec![
                ContentBlock::Text("Calling the tool.".into()),
                tool_use(RESPOND_TOOL, &input),
            ],
        );

        let response = chat_response(output, &request(Some(json!({})))).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&response.text).unwrap(),
            input
        );
        assert_eq!(response.mean_logprob, None);
    }

    #[test]
    fn structured_response_without_the_tool_call_returns_the_text() {
        let output = output(
            StopReason::EndTurn,
            vec![
                tool_use("other", &json!({})),
                ContentBlock::Text("I cannot".into()),
                ContentBlock::Text(" comply.".into()),
            ],
        );

        assert_eq!(
            chat_response(output, &request(Some(json!({}))))
                .unwrap()
                .text,
            "I cannot comply."
        );
    }

    #[test]
    fn plain_response_joins_text_blocks() {
        let output = output(
            StopReason::EndTurn,
            vec![ContentBlock::Text("Hello".into())],
        );

        assert_eq!(chat_response(output, &request(None)).unwrap().text, "Hello");
    }

    #[test]
    fn a_reply_stopped_at_max_tokens_is_cut_off_even_with_a_tool_call() {
        let output = output(
            StopReason::MaxTokens,
            vec![tool_use(RESPOND_TOOL, &json!({ "claims": [] }))],
        );

        let error = format!(
            "{:#?}",
            chat_response(output, &request(Some(json!({})))).unwrap_err()
        );
        assert!(
            error.contains("answer cut off at max_tokens=512 (stop reason max_tokens)"),
            "{error}"
        );
    }

    #[test]
    fn documents_round_trip_json() {
        let value = json!({
            "s": "x", "b": true, "n": null, "pos": 3, "neg": -4, "float": 1.5,
            "nested": [{ "a": [1, "two"] }],
        });

        assert_eq!(from_document(&to_document(&value)), value);
    }
}
