//! Workers AI adapters, all under `https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/` with a bearer API
//! token:
//!
//! - [`WorkersAiChat`] posts to the OpenAI-compatible `v1/chat/completions` route, with the body and reply of
//!   [`super::openai`].
//! - [`WorkersAiEmbedder`] runs its model on the native `ai/run/{model}` route, `{"text": [...], "truncate_inputs":
//!   true}`, because the OpenAI-compatible route ignores `truncate_inputs` and rejects an input over the model's input
//!   limit. It packs every request under the model's documented request limit, so no setting can leave a request
//!   unbounded.
//! - [`WorkersAiSystemOne`] runs Cloudflare's Clef models, e.g. `@cf/cloudflare/clef-flash`, on `ai/run/{model}`. The
//!   body is the System One [`DecisionRequest`], with `model` set to the selector the model page names (`clef` or
//!   `clef-flash`, the id's last segment).
//!
//! An `ai/run` reply is wrapped as `{"result": ..., "success": ..., "errors": [...], "messages": [...]}`. A failure
//! answered with that envelope, on any route, is reported by its `errors`; any other body is quoted, with the host
//! that sent it.

use std::{fmt, num::NonZeroUsize, str::FromStr, time::Duration};

use async_trait::async_trait;
use eros::{Context, bail};
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use super::{
    ApiKey, ChatModel, ChatRequest, ChatResponse, EmbedPurpose, Embedder, ModelRef,
    openai::{
        BatchBudget, CHAT_PATH, ChatOptions, batches, chat_body, chat_text, checked_vectors,
        embed_model_ref, estimated_tokens,
    },
    system_one::{DecisionRequest, DecisionResponse, SystemOne},
};

/// The Cloudflare API root that account routes hang off.
const CLOUDFLARE_API: &str = "https://api.cloudflare.com/client/v4";
/// Workers AI's own limit on a synchronous run.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Clef Flash's context window; its text `state` is truncated to fit, without an error.
const CLEF_FLASH_CONTEXT_TOKENS: u64 = 24_576;
/// A request whose input came within this many tokens of its model's context may have lost the end of its state.
const TRUNCATION_MARGIN: u64 = 512;
/// How much of a body outside the envelope an error quotes.
const BODY_PREVIEW_BYTES: usize = 200;
/// bge-m3's documented 60,000-token context, which spans one whole request, counted as its text count times its
/// longest text: 35 sections whose longest was about 2,916 tokens failed with `Max context reached 102060 tokens but
/// model supports only 60000`.
const BGE_M3_REQUEST_TOKENS: usize = 60_000;
/// bge-m3's limit on one input; with `truncate_inputs` a longer input is embedded from its first 8,192 tokens, as
/// Ollama's bge-m3 does, instead of failing with `Sequence too long: 10670 > 8192`.
const BGE_M3_INPUT_TOKENS: usize = 8_192;
/// Tokens a packed request stays under its model's limit by, on top of the 3-bytes-a-token estimate: text dense in
/// digits and symbols, such as a table of limits, can tokenize nearer 2 bytes a token.
const REQUEST_TOKEN_MARGIN: usize = 10_000;
const BGE_M3_BATCH_TOKENS: NonZeroUsize =
    NonZeroUsize::new(BGE_M3_REQUEST_TOKENS - REQUEST_TOKEN_MARGIN)
        .expect("bge-m3's request limit is above the margin");

/// A Cloudflare account id, kept to one URL path segment: ASCII letters, digits, `-` and `_`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountId(String);

/// A value that cannot be a Cloudflare account id.
#[derive(Debug, thiserror::Error)]
#[error("expected a Cloudflare account id (ASCII letters, digits, - and _), got {0:?}")]
pub struct NotAnAccountId(String);

impl FromStr for AccountId {
    type Err = NotAnAccountId;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        let valid = !id.is_empty()
            && id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_');
        match valid {
            true => Ok(Self(id.to_owned())),
            false => Err(NotAnAccountId(id.to_owned())),
        }
    }
}

impl fmt::Display for AccountId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A Workers AI model id, `@cf/{vendor}/{name}` (or `@hf/...`), e.g. `@cf/cloudflare/clef-flash`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkersAiModel {
    id: String,
    name: String,
}

impl WorkersAiModel {
    /// The full id, used in the route and recorded as the [`ModelRef`] id.
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    /// The id's last segment, sent as the System One request's `model` selector.
    fn name(&self) -> &str {
        &self.name
    }

    /// The context window, for the models whose window is documented.
    fn context_tokens(&self) -> Option<u64> {
        match self.id.as_str() {
            "@cf/cloudflare/clef-flash" => Some(CLEF_FLASH_CONTEXT_TOKENS),
            _ => None,
        }
    }
}

impl fmt::Display for WorkersAiModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

/// A model name that is not a Workers AI model id.
#[derive(Debug, thiserror::Error)]
#[error("expected a Workers AI model id such as @cf/cloudflare/clef-flash, got {0:?}")]
pub struct NotAWorkersAiModel(String);

impl FromStr for WorkersAiModel {
    type Err = NotAWorkersAiModel;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        let mut parts = id.split('/');
        match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some("@cf" | "@hf"), Some(vendor), Some(name), None)
                if !vendor.is_empty() && !name.is_empty() =>
            {
                Ok(Self {
                    id: id.to_owned(),
                    name: name.to_owned(),
                })
            }
            _ => Err(NotAWorkersAiModel(id.to_owned())),
        }
    }
}

/// A Workers AI embedding model with [`super::EMBEDDING_DIM`] dimensions and a documented request limit. The set is
/// closed: a request limit is what keeps ingest from failing mid-corpus, so a model without one is refused at startup
/// rather than given a guessed budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkersAiEmbedModel {
    /// `@cf/baai/bge-m3`: 60,000 tokens a request, with every text padded to the longest, and 8,192 an input.
    BgeM3,
}

impl WorkersAiEmbedModel {
    const ALL: [Self; 1] = [Self::BgeM3];

    /// The full id, which names the `ai/run` route.
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::BgeM3 => "@cf/baai/bge-m3",
        }
    }

    /// The padded estimate one request stays under: the model's limit less [`REQUEST_TOKEN_MARGIN`]. The estimate
    /// is of the untruncated texts, since whether Workers AI checks the request limit before or after truncating is
    /// not documented; a text over the budget alone is sent alone.
    fn batch_budget(self) -> BatchBudget {
        match self {
            Self::BgeM3 => BatchBudget::truncating(BGE_M3_BATCH_TOKENS),
        }
    }

    /// The longest input the model embeds whole; the server truncates a longer one.
    fn input_tokens(self) -> usize {
        match self {
            Self::BgeM3 => BGE_M3_INPUT_TOKENS,
        }
    }
}

impl fmt::Display for WorkersAiEmbedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// An embedding model whose request limit Drew does not know.
#[derive(Debug, thiserror::Error)]
#[error(
    "{0:?} is not a Workers AI embedding model with a known request limit; supported: {supported}",
    supported = supported_embed_models()
)]
pub struct UnsupportedEmbedModel(String);

fn supported_embed_models() -> String {
    WorkersAiEmbedModel::ALL
        .map(WorkersAiEmbedModel::id)
        .join(", ")
}

impl FromStr for WorkersAiEmbedModel {
    type Err = UnsupportedEmbedModel;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|model| model.id() == id)
            .ok_or_else(|| UnsupportedEmbedModel(id.to_owned()))
    }
}

/// The public Cloudflare API root.
pub(crate) fn cloudflare_api() -> Url {
    CLOUDFLARE_API
        .parse()
        .expect("CLOUDFLARE_API is a valid URL")
}

/// The `ai/run` route of the model `model_id` under the API root.
pub(crate) fn run_path(account: &AccountId, model_id: &str) -> String {
    format!("accounts/{account}/ai/run/{model_id}")
}

/// An OpenAI-compatible route, such as [`CHAT_PATH`], under the API root.
pub(crate) fn openai_path(account: &AccountId, path: &str) -> String {
    format!("accounts/{account}/ai/v1/{path}")
}

#[eros::context("building the Workers AI HTTP client")]
fn client() -> eros::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()?)
}

/// One account's routes and the token that authorizes them.
struct Account {
    client: reqwest::Client,
    api_base: String,
    id: AccountId,
    key: ApiKey,
}

/// A reply as it came off the wire.
struct Reply {
    status: StatusCode,
    /// Host and port, to name another server that answered in place of Workers AI.
    host: String,
    text: String,
}

impl Reply {
    fn preview(&self) -> &str {
        &self.text[..self.text.floor_char_boundary(BODY_PREVIEW_BYTES)]
    }
}

impl Account {
    fn new(id: AccountId, key: ApiKey) -> eros::Result<Self> {
        Ok(Self {
            client: client()?,
            api_base: CLOUDFLARE_API.to_owned(),
            id,
            key,
        })
    }

    fn with_api_base(self, api_base: &Url) -> Self {
        Self {
            api_base: api_base.as_str().trim_end_matches('/').to_owned(),
            ..self
        }
    }

    fn route(&self, path: &str) -> String {
        format!("{}/{path}", self.api_base)
    }

    #[eros::context("sending the request and reading the reply")]
    async fn send<B: Serialize + Sync>(&self, route: &str, body: &B) -> eros::Result<Reply> {
        let response = self
            .client
            .post(route)
            .bearer_auth(self.key.expose())
            .json(body)
            .send()
            .await?;
        let status = response.status();
        let url = response.url();
        let host = format!(
            "{}{}",
            url.host_str().unwrap_or_default(),
            url.port()
                .map(|port| format!(":{port}"))
                .unwrap_or_default()
        );
        let text = response.text().await?;
        Ok(Reply { status, host, text })
    }

    /// The `result` of running `model` on its `ai/run` route.
    #[eros::context("POST {} for {}", self.route(&run_path(&self.id, model)), model)]
    async fn run<T, B>(&self, model: &str, body: &B) -> eros::Result<T>
    where
        T: DeserializeOwned,
        B: Serialize + Sync,
    {
        let reply = self
            .send(&self.route(&run_path(&self.id, model)), body)
            .await?;
        let status = reply.status;
        let envelope = serde_json::from_str::<Envelope>(&reply.text).with_context(|| {
            format!(
                "{model} at {} returned {status} without the Workers AI envelope: {}",
                reply.host,
                reply.preview()
            )
        })?;
        match envelope {
            Envelope {
                success: true,
                result: Some(result),
                ..
            } if status.is_success() => Ok(T::deserialize(result)
                .with_context(|| format!("reading the result {model} returned"))?),
            Envelope { errors, .. } => {
                bail!("{model} failed with {status}: {}", messages(&errors))
            }
        }
    }

    /// The OpenAI-shaped reply of an OpenAI-compatible route.
    #[eros::context("POST {} for {}", self.route(&openai_path(&self.id, path)), model)]
    async fn post_openai(&self, path: &str, model: &str, body: &Value) -> eros::Result<Value> {
        let reply = self
            .send(&self.route(&openai_path(&self.id, path)), body)
            .await?;
        let value = serde_json::from_str::<Value>(&reply.text).ok();
        if let Some(Failure {
            success: false,
            errors,
        }) = value
            .as_ref()
            .and_then(|value| Failure::deserialize(value).ok())
        {
            bail!(
                "{model} failed with {}: {}",
                reply.status,
                messages(&errors)
            );
        }
        match value {
            Some(value) if reply.status.is_success() => Ok(value),
            _ => bail!(
                "{model} at {} returned {}: {}",
                reply.host,
                reply.status,
                reply.preview()
            ),
        }
    }
}

/// [`Embedder`] over Workers AI's native `ai/run` route with `truncate_inputs`, in requests of at most 64 texts whose
/// count times their longest text's estimated tokens stays a safety margin under the model's request limit. An input
/// over the model's input limit is embedded from its start. Ignores [`EmbedPurpose`]: bge-m3 embeds a query and a
/// document alike, and its `query`/`contexts` input scores contexts against a query instead of embedding.
pub struct WorkersAiEmbedder {
    account: Account,
    model: WorkersAiEmbedModel,
}

impl WorkersAiEmbedder {
    /// Calls the public Cloudflare API.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(account: AccountId, model: WorkersAiEmbedModel, key: ApiKey) -> eros::Result<Self> {
        Ok(Self {
            account: Account::new(account, key)?,
            model,
        })
    }

    /// Another API root in place of `https://api.cloudflare.com/client/v4`, such as a test server. No setting reaches
    /// it: a URL left over from another backend would send the account's token to the wrong server.
    pub fn with_api_base(self, api_base: &Url) -> Self {
        Self {
            account: self.account.with_api_base(api_base),
            ..self
        }
    }
}

#[async_trait]
impl Embedder for WorkersAiEmbedder {
    fn model(&self) -> ModelRef {
        embed_model_ref(self.model.id())
    }

    async fn embed(&self, texts: &[String], _purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        self.embed_texts(texts).await
    }
}

/// The `result` of an embedding run: one vector per input, in input order.
#[derive(Deserialize)]
struct Embedded {
    data: Vec<Vec<f32>>,
}

impl WorkersAiEmbedder {
    #[eros::context("embedding {} texts with {}", texts.len(), self.model)]
    async fn embed_texts(&self, texts: &[String]) -> eros::Result<Vec<Vec<f32>>> {
        let limit = self.model.input_tokens();
        let truncated = texts
            .iter()
            .filter(|text| estimated_tokens(text) > limit)
            .count();
        if truncated > 0 {
            tracing::warn!(
                model = %self.model,
                truncated,
                inputs = texts.len(),
                "{truncated} of {} inputs are estimated over the model's {limit}-token input limit; each is \
                 embedded from its first {limit} tokens",
                texts.len()
            );
        }
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in batches(texts, Some(self.model.batch_budget()))? {
            let body = json!({ "text": batch, "truncate_inputs": true });
            let embedded: Embedded = self.account.run(self.model.id(), &body).await?;
            vectors.extend(checked_vectors(embedded.data, batch.len())?);
        }
        Ok(vectors)
    }
}

/// [`ChatModel`] over Workers AI's OpenAI-compatible `chat/completions` route.
pub struct WorkersAiChat {
    account: Account,
    model: WorkersAiModel,
    options: ChatOptions,
}

impl WorkersAiChat {
    /// Calls the public Cloudflare API.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(
        account: AccountId,
        model: WorkersAiModel,
        key: ApiKey,
        options: ChatOptions,
    ) -> eros::Result<Self> {
        Ok(Self {
            account: Account::new(account, key)?,
            model,
            options,
        })
    }

    /// See [`WorkersAiEmbedder::with_api_base`].
    pub fn with_api_base(self, api_base: &Url) -> Self {
        Self {
            account: self.account.with_api_base(api_base),
            ..self
        }
    }
}

#[async_trait]
impl ChatModel for WorkersAiChat {
    fn model(&self) -> ModelRef {
        ModelRef::new(self.model.id(), self.options.version())
    }

    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse> {
        let body = chat_body(self.model.id(), &request, self.options);
        let response = self
            .account
            .post_openai(CHAT_PATH, self.model.id(), &body)
            .await?;
        Ok(ChatResponse {
            text: chat_text(&response, request.max_tokens)?,
            mean_logprob: None,
        })
    }
}

/// [`SystemOne`] over Workers AI's `ai/run` route.
pub struct WorkersAiSystemOne {
    account: Account,
    model: WorkersAiModel,
}

impl WorkersAiSystemOne {
    /// Calls the public Cloudflare API.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(account: AccountId, model: WorkersAiModel, key: ApiKey) -> eros::Result<Self> {
        Ok(Self {
            account: Account::new(account, key)?,
            model,
        })
    }

    /// See [`WorkersAiEmbedder::with_api_base`].
    pub fn with_api_base(self, api_base: &Url) -> Self {
        Self {
            account: self.account.with_api_base(api_base),
            ..self
        }
    }

    #[cfg(test)]
    fn route(&self) -> String {
        self.account
            .route(&run_path(&self.account.id, self.model.id()))
    }
}

#[async_trait]
impl SystemOne for WorkersAiSystemOne {
    fn model(&self) -> ModelRef {
        ModelRef::new(self.model.id(), "workers-ai-run")
    }

    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        self.post(request).await
    }
}

/// Workers AI's REST envelope. `success` is required, so a JSON body from another server is not read as an empty
/// failure. `result` is read only on success, since a failure may carry `{}` there.
#[derive(Deserialize)]
struct Envelope {
    #[serde(default)]
    result: Option<Value>,
    success: bool,
    #[serde(default)]
    errors: Vec<ApiMessage>,
}

/// The envelope without its `result`, to recognize a failure on a route whose success is not enveloped.
#[derive(Deserialize)]
struct Failure {
    success: bool,
    #[serde(default)]
    errors: Vec<ApiMessage>,
}

#[derive(Deserialize)]
struct RunResult {
    #[serde(flatten)]
    response: DecisionResponse,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u64,
}

#[derive(Deserialize)]
struct ApiMessage {
    code: Option<i64>,
    message: String,
}

fn messages(errors: &[ApiMessage]) -> String {
    if errors.is_empty() {
        return "no error message".to_owned();
    }
    errors
        .iter()
        .map(|error| match error.code {
            Some(code) => format!("{code}: {}", error.message),
            None => error.message.clone(),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

impl WorkersAiSystemOne {
    #[eros::context("asking {} for decisions", self.model)]
    async fn post(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        let request = DecisionRequest {
            model: self.model.name().to_owned(),
            ..request
        };
        let result: RunResult = self.account.run(self.model.id(), &request).await?;
        if let (Some(usage), Some(context)) = (&result.usage, self.model.context_tokens())
            && usage.input_tokens + TRUNCATION_MARGIN >= context
        {
            tracing::warn!(
                model = %self.model,
                input_tokens = usage.input_tokens,
                "the System One state may have been truncated to fit the {context}-token context"
            );
        }
        Ok(result.response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_id_has_a_scope_a_vendor_and_a_name() {
        let model: WorkersAiModel = "@cf/cloudflare/clef-flash".parse().unwrap();
        assert_eq!(model.id(), "@cf/cloudflare/clef-flash");
        assert_eq!(model.name(), "clef-flash");
        assert_eq!(model.context_tokens(), Some(CLEF_FLASH_CONTEXT_TOKENS));
        let clef: WorkersAiModel = "@cf/cloudflare/clef".parse().unwrap();
        assert_eq!(clef.context_tokens(), None);
        for bad in [
            "clef-flash",
            "kev-latest",
            "@cf/clef-flash",
            "@cf//clef",
            "@cf/cloudflare/",
            "@cf/a/b/c",
        ] {
            assert!(bad.parse::<WorkersAiModel>().is_err(), "{bad}");
        }
    }

    #[test]
    fn an_account_id_is_one_path_segment() {
        let account: AccountId = "0123abcd-ef_9".parse().unwrap();
        assert_eq!(account.to_string(), "0123abcd-ef_9");
        for bad in ["", " ", "acc/../other", "acc?x=1", "acc 1"] {
            assert!(bad.parse::<AccountId>().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn only_embedding_models_with_a_known_request_limit_are_accepted() {
        let model: WorkersAiEmbedModel = "@cf/baai/bge-m3".parse().unwrap();
        assert_eq!(model, WorkersAiEmbedModel::BgeM3);
        assert_eq!(model.batch_budget().tokens(), 50_000);
        assert_eq!(model.input_tokens(), 8_192);
        let error = "@cf/baai/bge-large-en-v1.5"
            .parse::<WorkersAiEmbedModel>()
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "\"@cf/baai/bge-large-en-v1.5\" is not a Workers AI embedding model with a known request limit; \
             supported: @cf/baai/bge-m3"
        );
    }

    #[test]
    fn the_route_uses_the_public_api_unless_overridden() {
        let client = WorkersAiSystemOne::new(
            "acc".parse().unwrap(),
            "@cf/cloudflare/clef-flash".parse().unwrap(),
            "token".parse().unwrap(),
        )
        .unwrap();
        assert_eq!(
            client.route(),
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
        );
        let client = client.with_api_base(&"http://127.0.0.1:9/client/v4/".parse().unwrap());
        assert_eq!(
            client.route(),
            "http://127.0.0.1:9/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
        );
    }
}
