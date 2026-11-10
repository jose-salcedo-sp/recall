//! Which backend serves each model port, read once from the command line and environment.
//!
//! Each port names its backend, with no default: `DREW_EMBED_BACKEND`, `DREW_EXPLAIN_BACKEND` and
//! `DREW_SYSTEM_ONE_BACKEND`. A missing env file is then a startup error, not a silent fall back to Bedrock. The
//! flat command-line fields are parsed into one enum per port whose variants hold exactly the settings that backend
//! uses; a missing, unparsable or unused setting is an error naming its variable. Each arm destructures every field,
//! so a new field does not compile until every backend accepts or rejects it. [`embedder`], [`explain_model`] and
//! [`system_one_client`] map those enums to the port traits, which is all the rest of the crate sees.
//!
//! Every port has a `workers-ai` backend, which takes the Cloudflare account id, the API token and the model id, and
//! builds the Cloudflare URL itself, so no URL setting reaches it. Its embedder packs requests under the model's
//! documented request limit, so that limit does not depend on a setting.

use std::{error::Error, num::NonZeroUsize, str::FromStr, sync::Arc};

use reqwest::Url;

use super::{
    ApiKey, ChatModel, Embedder,
    bedrock_chat::{BedrockChat, aws_environment, bedrock_endpoint},
    bedrock_embed::BedrockEmbedder,
    openai::{
        CHAT_PATH, ChatOptions, EMBEDDINGS_PATH, OpenAiChat, OpenAiEmbedder, ReasoningEffort,
        SchemaFormat,
    },
    system_one::{DECIDE_PATH, HttpSystemOne, SystemOne},
    workers_ai::{
        AccountId, WorkersAiChat, WorkersAiEmbedModel, WorkersAiEmbedder, WorkersAiModel,
        WorkersAiSystemOne, cloudflare_api, openai_path, run_path,
    },
};

/// The embedding model and the backend that serves it.
#[derive(Clone, Debug, clap::Args)]
pub struct EmbedArgs {
    /// Which backend serves embeddings.
    #[arg(long, env = "DREW_EMBED_BACKEND", value_enum)]
    pub embed_backend: EmbedBackendKind,
    /// Embedding model, with 1024 dimensions: on Bedrock Titan Text Embeddings v2 or Cohere Embed v3, on Workers AI
    /// `@cf/baai/bge-m3`, otherwise the server's name, e.g. `bge-m3` in Ollama.
    #[arg(long, env = "DREW_EMBED_MODEL")]
    pub embed_model: String,
    /// `openai-compatible` only, and required there: base URL, e.g. `http://localhost:11434/v1`.
    #[arg(long, env = "DREW_EMBED_URL")]
    pub embed_url: Option<Url>,
    /// `workers-ai` only, and required there: the Cloudflare account id.
    #[arg(long, env = "DREW_EMBED_ACCOUNT_ID")]
    pub embed_account_id: Option<AccountId>,
    /// Bearer key: optional on `openai-compatible`; the Workers AI API token, required, on `workers-ai`.
    #[arg(long, env = "DREW_EMBED_KEY", hide_env_values = true)]
    pub embed_key: Option<ApiKey>,
    /// `openai-compatible` only: the most estimated tokens one request carries, counted as its text count times its
    /// longest text's estimate, for a server whose limit spans a request. Unset: 64 texts a request whatever their
    /// size. `workers-ai` knows its model's limit.
    #[arg(long, env = "DREW_EMBED_MAX_BATCH_TOKENS")]
    pub embed_max_batch_tokens: Option<NonZeroUsize>,
}

/// `DREW_EMBED_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum EmbedBackendKind {
    /// Amazon Bedrock; region and credentials from the default AWS chain.
    Bedrock,
    /// `POST {url}/embeddings`: Ollama, vLLM, or llama.cpp.
    OpenaiCompatible,
    /// `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/embeddings`, packed under the model's
    /// request limit.
    WorkersAi,
}

/// The `Explain` LLM and the backend that serves it.
#[derive(Clone, Debug, clap::Args)]
pub struct ChatArgs {
    /// Which backend serves `Explain`.
    #[arg(long, env = "DREW_EXPLAIN_BACKEND", value_enum)]
    pub explain_backend: ChatBackendKind,
    /// Model for `Explain`: a Bedrock model or inference profile id, e.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`,
    /// a Workers AI model id, e.g. `@cf/qwen/qwen3.8-27b`, or the server's model name, e.g. `qwen3:14b`.
    #[arg(long, env = "DREW_EXPLAIN_MODEL")]
    pub explain_model: String,
    /// `openai-compatible` only, and required there: base URL, e.g. `http://localhost:11434/v1`.
    #[arg(long, env = "DREW_EXPLAIN_URL")]
    pub explain_url: Option<Url>,
    /// `workers-ai` only, and required there: the Cloudflare account id.
    #[arg(long, env = "DREW_EXPLAIN_ACCOUNT_ID")]
    pub explain_account_id: Option<AccountId>,
    /// Bearer key: optional on `openai-compatible`; the Workers AI API token, required, on `workers-ai`.
    #[arg(long, env = "DREW_EXPLAIN_KEY", hide_env_values = true)]
    pub explain_key: Option<ApiKey>,
    /// `openai-compatible` and `workers-ai`: `reasoning_effort` to send; unset sends none. Use `medium` on Workers
    /// AI: `low` with a `json_schema` hangs there (cloudflare/workers-sdk#16165).
    #[arg(long, env = "DREW_EXPLAIN_REASONING_EFFORT", value_enum)]
    pub explain_reasoning_effort: Option<ReasoningEffort>,
    /// `openai-compatible` and `workers-ai`: how the response schema is sent. Unset: `openai`.
    #[arg(long, env = "DREW_EXPLAIN_SCHEMA_FORMAT", value_enum)]
    pub explain_schema_format: Option<SchemaFormat>,
}

/// `DREW_EXPLAIN_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ChatBackendKind {
    /// Amazon Bedrock's Converse API; region and credentials from the default AWS chain.
    Bedrock,
    /// `POST {url}/chat/completions`: Ollama, vLLM, or llama.cpp.
    OpenaiCompatible,
    /// `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions`.
    WorkersAi,
}

/// System One and the backend that serves it.
#[derive(Clone, Debug, clap::Args)]
#[group(id = "system_one")]
pub struct SystemOneArgs {
    /// Which backend serves System One; `none` classifies every question `open`.
    #[arg(long, env = "DREW_SYSTEM_ONE_BACKEND", value_enum)]
    pub system_one_backend: SystemOneBackendKind,
    /// Model: e.g. `kev-latest`, `clef-flash` on system-one.dev, or `@cf/cloudflare/clef-flash` on Workers AI.
    #[arg(long, env = "DREW_SYSTEM_ONE_MODEL")]
    pub system_one_model: Option<String>,
    /// `system-one` only, and required there: the server root, e.g. `http://127.0.0.1:8009` or
    /// `https://system-one.dev`.
    #[arg(long, env = "DREW_SYSTEM_ONE_URL")]
    pub system_one_url: Option<Url>,
    /// `workers-ai` only: the Cloudflare account id.
    #[arg(long, env = "DREW_SYSTEM_ONE_ACCOUNT_ID")]
    pub system_one_account_id: Option<AccountId>,
    /// Bearer key: Kev's `KEV_API_KEY` or a system-one.dev key, or the Workers AI API token (required there).
    #[arg(long, env = "DREW_SYSTEM_ONE_KEY", hide_env_values = true)]
    pub system_one_key: Option<ApiKey>,
}

/// `DREW_SYSTEM_ONE_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SystemOneBackendKind {
    /// No System One: every question is classified `open`, and nothing is answered yes or no.
    None,
    /// `POST {url}/v1/systemone`: Kev, or system-one.dev.
    SystemOne,
    /// Clef on Workers AI: `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`.
    WorkersAi,
}

/// A backend setting that is missing, unparsable, or set for a backend that does not use it.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InvalidBackend {
    #[error("{var} is required with {backend}")]
    Missing { var: &'static str, backend: String },
    #[error("{var} is set, but {backend} does not use it; unset it")]
    Unused { var: &'static str, backend: String },
    #[error("{var}: {source}")]
    Unparsable {
        var: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Where a backend sends its requests: scheme, host, port and path. Credentials, query and fragment are dropped, so
/// it is safe to print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint(String);

impl Endpoint {
    /// `path` under `base`, as the adapters build their routes.
    pub(crate) fn new(base: &Url, path: &str) -> Self {
        let port = base
            .port()
            .map(|port| format!(":{port}"))
            .unwrap_or_default();
        Self(format!(
            "{}://{}{port}{}/{path}",
            base.scheme(),
            base.host_str().unwrap_or_default(),
            base.path().trim_end_matches('/'),
        ))
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The backend selection that settings are checked against.
struct Selected(String);

impl Selected {
    fn new(var: &str, value: &impl clap::ValueEnum) -> Self {
        let value = value
            .to_possible_value()
            .expect("no backend variant is skipped, so each has a value");
        Self(format!("{var}={}", value.get_name()))
    }

    fn require<T: Clone>(&self, value: Option<&T>, var: &'static str) -> Result<T, InvalidBackend> {
        value.cloned().ok_or_else(|| InvalidBackend::Missing {
            var,
            backend: self.0.clone(),
        })
    }

    fn reject<T>(&self, value: Option<&T>, var: &'static str) -> Result<(), InvalidBackend> {
        match value {
            Some(_) => Err(InvalidBackend::Unused {
                var,
                backend: self.0.clone(),
            }),
            None => Ok(()),
        }
    }
}

/// `value` of `var`, parsed.
fn parse<T>(value: &str, var: &'static str) -> Result<T, InvalidBackend>
where
    T: FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    value.parse().map_err(|error| InvalidBackend::Unparsable {
        var,
        source: Box::new(error),
    })
}

/// [`EmbedArgs`], checked.
pub(crate) enum EmbedBackend {
    Bedrock {
        model: String,
    },
    OpenAiCompatible {
        url: Url,
        model: String,
        key: Option<ApiKey>,
        max_batch_tokens: Option<NonZeroUsize>,
    },
    WorkersAi {
        account_id: AccountId,
        model: WorkersAiEmbedModel,
        key: ApiKey,
    },
}

impl TryFrom<&EmbedArgs> for EmbedBackend {
    type Error = InvalidBackend;

    fn try_from(args: &EmbedArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_EMBED_BACKEND", &args.embed_backend);
        Ok(match args.embed_backend {
            EmbedBackendKind::Bedrock => {
                let EmbedArgs {
                    embed_backend: _,
                    embed_model,
                    embed_url,
                    embed_account_id,
                    embed_key,
                    embed_max_batch_tokens,
                } = args;
                selected.reject(embed_url.as_ref(), "DREW_EMBED_URL")?;
                selected.reject(embed_account_id.as_ref(), "DREW_EMBED_ACCOUNT_ID")?;
                selected.reject(embed_key.as_ref(), "DREW_EMBED_KEY")?;
                selected.reject(
                    embed_max_batch_tokens.as_ref(),
                    "DREW_EMBED_MAX_BATCH_TOKENS",
                )?;
                Self::Bedrock {
                    model: embed_model.clone(),
                }
            }
            EmbedBackendKind::OpenaiCompatible => {
                let EmbedArgs {
                    embed_backend: _,
                    embed_model,
                    embed_url,
                    embed_account_id,
                    embed_key,
                    embed_max_batch_tokens,
                } = args;
                selected.reject(embed_account_id.as_ref(), "DREW_EMBED_ACCOUNT_ID")?;
                Self::OpenAiCompatible {
                    url: selected.require(embed_url.as_ref(), "DREW_EMBED_URL")?,
                    model: embed_model.clone(),
                    key: embed_key.clone(),
                    max_batch_tokens: *embed_max_batch_tokens,
                }
            }
            EmbedBackendKind::WorkersAi => {
                let EmbedArgs {
                    embed_backend: _,
                    embed_model,
                    embed_url,
                    embed_account_id,
                    embed_key,
                    embed_max_batch_tokens,
                } = args;
                selected.reject(embed_url.as_ref(), "DREW_EMBED_URL")?;
                selected.reject(
                    embed_max_batch_tokens.as_ref(),
                    "DREW_EMBED_MAX_BATCH_TOKENS",
                )?;
                Self::WorkersAi {
                    account_id: selected
                        .require(embed_account_id.as_ref(), "DREW_EMBED_ACCOUNT_ID")?,
                    model: parse(embed_model, "DREW_EMBED_MODEL")?,
                    key: selected.require(embed_key.as_ref(), "DREW_EMBED_KEY")?,
                }
            }
        })
    }
}

impl EmbedBackend {
    /// What `drew models check` prints before calling a backend that depends on ambient settings.
    pub(crate) async fn note(&self) -> Option<String> {
        match self {
            Self::Bedrock { .. } => Some(format!(
                "embeddings on Bedrock (DREW_EMBED_BACKEND=bedrock): {}",
                aws_environment().await
            )),
            Self::OpenAiCompatible { .. } | Self::WorkersAi { .. } => None,
        }
    }

    /// Where the embedder posts; `None` on Bedrock with no AWS region.
    pub(crate) async fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Self::Bedrock { model } => bedrock_endpoint(&format!("model/{model}/invoke")).await,
            Self::OpenAiCompatible { url, .. } => Some(Endpoint::new(url, EMBEDDINGS_PATH)),
            Self::WorkersAi {
                account_id, model, ..
            } => Some(Endpoint::new(
                &cloudflare_api(),
                &run_path(account_id, model.id()),
            )),
        }
    }

    #[eros::context("setting up the embeddings backend")]
    pub(crate) async fn build(self) -> eros::Result<Arc<dyn Embedder>> {
        Ok(match self {
            Self::Bedrock { model } => Arc::new(BedrockEmbedder::load(&model).await?),
            Self::OpenAiCompatible {
                url,
                model,
                key,
                max_batch_tokens,
            } => Arc::new(
                OpenAiEmbedder::new(&url, model)?
                    .with_key(key)
                    .with_max_batch_tokens(max_batch_tokens),
            ),
            Self::WorkersAi {
                account_id,
                model,
                key,
            } => Arc::new(WorkersAiEmbedder::new(account_id, model, key)?),
        })
    }
}

/// [`ChatArgs`], checked.
pub(crate) enum ChatBackend {
    Bedrock {
        model: String,
    },
    OpenAiCompatible {
        url: Url,
        model: String,
        key: Option<ApiKey>,
        options: ChatOptions,
    },
    WorkersAi {
        account_id: AccountId,
        model: WorkersAiModel,
        key: ApiKey,
        options: ChatOptions,
    },
}

impl TryFrom<&ChatArgs> for ChatBackend {
    type Error = InvalidBackend;

    fn try_from(args: &ChatArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_EXPLAIN_BACKEND", &args.explain_backend);
        Ok(match args.explain_backend {
            ChatBackendKind::Bedrock => {
                let ChatArgs {
                    explain_backend: _,
                    explain_model,
                    explain_url,
                    explain_account_id,
                    explain_key,
                    explain_reasoning_effort,
                    explain_schema_format,
                } = args;
                selected.reject(explain_url.as_ref(), "DREW_EXPLAIN_URL")?;
                selected.reject(explain_account_id.as_ref(), "DREW_EXPLAIN_ACCOUNT_ID")?;
                selected.reject(explain_key.as_ref(), "DREW_EXPLAIN_KEY")?;
                selected.reject(
                    explain_reasoning_effort.as_ref(),
                    "DREW_EXPLAIN_REASONING_EFFORT",
                )?;
                selected.reject(explain_schema_format.as_ref(), "DREW_EXPLAIN_SCHEMA_FORMAT")?;
                Self::Bedrock {
                    model: explain_model.clone(),
                }
            }
            ChatBackendKind::OpenaiCompatible => {
                let ChatArgs {
                    explain_backend: _,
                    explain_model,
                    explain_url,
                    explain_account_id,
                    explain_key,
                    explain_reasoning_effort,
                    explain_schema_format,
                } = args;
                selected.reject(explain_account_id.as_ref(), "DREW_EXPLAIN_ACCOUNT_ID")?;
                Self::OpenAiCompatible {
                    url: selected.require(explain_url.as_ref(), "DREW_EXPLAIN_URL")?,
                    model: explain_model.clone(),
                    key: explain_key.clone(),
                    options: ChatOptions {
                        reasoning_effort: *explain_reasoning_effort,
                        schema_format: explain_schema_format.unwrap_or_default(),
                    },
                }
            }
            ChatBackendKind::WorkersAi => {
                let ChatArgs {
                    explain_backend: _,
                    explain_model,
                    explain_url,
                    explain_account_id,
                    explain_key,
                    explain_reasoning_effort,
                    explain_schema_format,
                } = args;
                selected.reject(explain_url.as_ref(), "DREW_EXPLAIN_URL")?;
                Self::WorkersAi {
                    account_id: selected
                        .require(explain_account_id.as_ref(), "DREW_EXPLAIN_ACCOUNT_ID")?,
                    model: parse(explain_model, "DREW_EXPLAIN_MODEL")?,
                    key: selected.require(explain_key.as_ref(), "DREW_EXPLAIN_KEY")?,
                    options: ChatOptions {
                        reasoning_effort: *explain_reasoning_effort,
                        schema_format: explain_schema_format.unwrap_or_default(),
                    },
                }
            }
        })
    }
}

impl ChatBackend {
    /// What `drew models check` prints before calling a backend that depends on ambient settings.
    pub(crate) async fn note(&self) -> Option<String> {
        match self {
            Self::Bedrock { .. } => Some(format!(
                "explain on Bedrock (DREW_EXPLAIN_BACKEND=bedrock): {}",
                aws_environment().await
            )),
            Self::OpenAiCompatible { .. } | Self::WorkersAi { .. } => None,
        }
    }

    /// Where `Explain` posts; `None` on Bedrock with no AWS region.
    pub(crate) async fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Self::Bedrock { model } => bedrock_endpoint(&format!("model/{model}/converse")).await,
            Self::OpenAiCompatible { url, .. } => Some(Endpoint::new(url, CHAT_PATH)),
            Self::WorkersAi { account_id, .. } => Some(Endpoint::new(
                &cloudflare_api(),
                &openai_path(account_id, CHAT_PATH),
            )),
        }
    }

    #[eros::context("setting up the Explain backend")]
    pub(crate) async fn build(self) -> eros::Result<Arc<dyn ChatModel>> {
        Ok(match self {
            Self::Bedrock { model } => Arc::new(BedrockChat::load(&model).await),
            Self::OpenAiCompatible {
                url,
                model,
                key,
                options,
            } => Arc::new(
                OpenAiChat::new(&url, model)?
                    .with_key(key)
                    .with_options(options),
            ),
            Self::WorkersAi {
                account_id,
                model,
                key,
                options,
            } => Arc::new(WorkersAiChat::new(account_id, model, key, options)?),
        })
    }
}

/// [`SystemOneArgs`], checked.
pub(crate) enum SystemOneBackend {
    None,
    SystemOne {
        url: Url,
        model: String,
        key: Option<ApiKey>,
    },
    WorkersAi {
        account_id: AccountId,
        model: WorkersAiModel,
        key: ApiKey,
    },
}

impl TryFrom<&SystemOneArgs> for SystemOneBackend {
    type Error = InvalidBackend;

    fn try_from(args: &SystemOneArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_SYSTEM_ONE_BACKEND", &args.system_one_backend);
        Ok(match args.system_one_backend {
            SystemOneBackendKind::None => {
                let SystemOneArgs {
                    system_one_backend: _,
                    system_one_model,
                    system_one_url,
                    system_one_account_id,
                    system_one_key,
                } = args;
                selected.reject(system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?;
                selected.reject(system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?;
                selected.reject(system_one_account_id.as_ref(), "DREW_SYSTEM_ONE_ACCOUNT_ID")?;
                selected.reject(system_one_key.as_ref(), "DREW_SYSTEM_ONE_KEY")?;
                Self::None
            }
            SystemOneBackendKind::SystemOne => {
                let SystemOneArgs {
                    system_one_backend: _,
                    system_one_model,
                    system_one_url,
                    system_one_account_id,
                    system_one_key,
                } = args;
                selected.reject(system_one_account_id.as_ref(), "DREW_SYSTEM_ONE_ACCOUNT_ID")?;
                Self::SystemOne {
                    url: selected.require(system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?,
                    model: selected.require(system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?,
                    key: system_one_key.clone(),
                }
            }
            SystemOneBackendKind::WorkersAi => {
                let SystemOneArgs {
                    system_one_backend: _,
                    system_one_model,
                    system_one_url,
                    system_one_account_id,
                    system_one_key,
                } = args;
                selected.reject(system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?;
                let model = parse(
                    &selected.require(system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?,
                    "DREW_SYSTEM_ONE_MODEL",
                )?;
                Self::WorkersAi {
                    account_id: selected
                        .require(system_one_account_id.as_ref(), "DREW_SYSTEM_ONE_ACCOUNT_ID")?,
                    model,
                    key: selected.require(system_one_key.as_ref(), "DREW_SYSTEM_ONE_KEY")?,
                }
            }
        })
    }
}

impl SystemOneBackend {
    /// What `drew models check` prints before calling a backend that depends on ambient settings, or instead of
    /// calling none.
    pub(crate) fn note(&self) -> Option<String> {
        match self {
            Self::None => Some(
                "system one: none (DREW_SYSTEM_ONE_BACKEND=none); every question is classified open"
                    .to_owned(),
            ),
            Self::SystemOne { .. } | Self::WorkersAi { .. } => None,
        }
    }

    /// Where System One posts; `None` for [`SystemOneBackend::None`].
    pub(crate) fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Self::None => None,
            Self::SystemOne { url, .. } => Some(Endpoint::new(url, DECIDE_PATH)),
            Self::WorkersAi {
                account_id, model, ..
            } => Some(Endpoint::new(
                &cloudflare_api(),
                &run_path(account_id, model.id()),
            )),
        }
    }

    /// `None` for [`SystemOneBackend::None`].
    #[eros::context("setting up the System One backend")]
    pub(crate) fn build(self) -> eros::Result<Option<Arc<dyn SystemOne>>> {
        Ok(match self {
            Self::None => None,
            Self::SystemOne { url, model, key } => {
                Some(Arc::new(HttpSystemOne::new(&url, model, key)?))
            }
            Self::WorkersAi {
                account_id,
                model,
                key,
            } => Some(Arc::new(WorkersAiSystemOne::new(account_id, model, key)?)),
        })
    }
}

/// The embedder `DREW_EMBED_BACKEND` selects.
///
/// # Errors
///
/// When a backend setting is missing, unparsable or unused, the HTTP client cannot be built, or on Bedrock when the
/// model is not Titan Text Embeddings v2 or Cohere Embed v3.
pub async fn embedder(args: &EmbedArgs) -> eros::Result<Arc<dyn Embedder>> {
    EmbedBackend::try_from(args)?.build().await
}

/// The `Explain` LLM `DREW_EXPLAIN_BACKEND` selects.
///
/// # Errors
///
/// When a backend setting is missing, unparsable or unused, or the HTTP client cannot be built.
pub async fn explain_model(args: &ChatArgs) -> eros::Result<Arc<dyn ChatModel>> {
    ChatBackend::try_from(args)?.build().await
}

/// The System One `DREW_SYSTEM_ONE_BACKEND` selects; `None` for `none`.
///
/// # Errors
///
/// When a backend setting is missing, unparsable or unused, or the HTTP client cannot be built.
pub fn system_one_client(args: &SystemOneArgs) -> eros::Result<Option<Arc<dyn SystemOne>>> {
    SystemOneBackend::try_from(args)?.build()
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        embed: EmbedArgs,
        #[command(flatten)]
        chat: ChatArgs,
        #[command(flatten)]
        system_one: SystemOneArgs,
    }

    #[test]
    fn every_backend_is_required_with_no_default() {
        let command = Cli::command();
        for (id, env) in [
            ("embed_backend", "DREW_EMBED_BACKEND"),
            ("explain_backend", "DREW_EXPLAIN_BACKEND"),
            ("system_one_backend", "DREW_SYSTEM_ONE_BACKEND"),
            ("explain_model", "DREW_EXPLAIN_MODEL"),
            ("embed_model", "DREW_EMBED_MODEL"),
        ] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == id)
                .unwrap();
            assert_eq!(arg.get_env().and_then(|env| env.to_str()), Some(env));
            assert!(arg.is_required_set(), "{id}");
            assert!(arg.get_default_values().is_empty(), "{id}");
        }
    }

    #[test]
    fn keys_are_hidden_from_help() {
        let command = Cli::command();
        for id in ["embed_key", "explain_key", "system_one_key"] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == id)
                .unwrap();
            assert!(arg.is_hide_env_values_set(), "{id}");
        }
    }

    fn try_parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let base = [
            "drew",
            "--embed-backend",
            "openai-compatible",
            "--embed-model",
            "bge-m3",
            "--explain-model",
            "qwen3:14b",
        ];
        Cli::try_parse_from(base.iter().chain(args))
    }

    fn parse(args: &[&str]) -> Cli {
        try_parse(args).unwrap()
    }

    fn missing(error: InvalidBackend) -> &'static str {
        match error {
            InvalidBackend::Missing { var, .. } => var,
            other => panic!("expected a missing setting, got {other}"),
        }
    }

    #[test]
    fn a_backend_names_the_setting_it_lacks() {
        let cli = parse(&[
            "--explain-backend",
            "openai-compatible",
            "--system-one-backend",
            "workers-ai",
            "--system-one-model",
            "@cf/cloudflare/clef-flash",
            "--system-one-key",
            "token",
        ]);
        assert_eq!(
            missing(EmbedBackend::try_from(&cli.embed).err().unwrap()),
            "DREW_EMBED_URL"
        );
        assert_eq!(
            missing(ChatBackend::try_from(&cli.chat).err().unwrap()),
            "DREW_EXPLAIN_URL"
        );
        let error = SystemOneBackend::try_from(&cli.system_one).err().unwrap();
        assert_eq!(
            error.to_string(),
            "DREW_SYSTEM_ONE_ACCOUNT_ID is required with DREW_SYSTEM_ONE_BACKEND=workers-ai"
        );
        assert_eq!(missing(error), "DREW_SYSTEM_ONE_ACCOUNT_ID");
    }

    #[test]
    fn a_setting_for_another_backend_is_rejected() {
        let cli = parse(&[
            "--explain-backend",
            "bedrock",
            "--explain-reasoning-effort",
            "medium",
            "--system-one-backend",
            "none",
            "--system-one-url",
            "http://127.0.0.1:8009",
        ]);
        assert_eq!(
            ChatBackend::try_from(&cli.chat).err().unwrap().to_string(),
            "DREW_EXPLAIN_REASONING_EFFORT is set, but DREW_EXPLAIN_BACKEND=bedrock does not use it; unset it"
        );
        assert!(matches!(
            SystemOneBackend::try_from(&cli.system_one),
            Err(InvalidBackend::Unused {
                var: "DREW_SYSTEM_ONE_URL",
                ..
            })
        ));
    }

    #[test]
    fn workers_ai_needs_a_workers_ai_model_id_named_by_its_variable() {
        let workers_ai = |model: &str| {
            parse(&[
                "--explain-backend",
                "openai-compatible",
                "--system-one-backend",
                "workers-ai",
                "--system-one-model",
                model,
                "--system-one-account-id",
                "acc",
                "--system-one-key",
                "token",
            ])
        };
        let cli = workers_ai("@cf/cloudflare/clef-flash");
        let backend = SystemOneBackend::try_from(&cli.system_one).unwrap();
        assert!(matches!(backend, SystemOneBackend::WorkersAi { .. }));
        assert_eq!(
            backend.endpoint().unwrap().to_string(),
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
        );
        assert!(system_one_client(&cli.system_one).unwrap().is_some());

        let error = SystemOneBackend::try_from(&workers_ai("kev-latest").system_one)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.starts_with("DREW_SYSTEM_ONE_MODEL: ")
                && error.contains("@cf/cloudflare/clef-flash"),
            "{error}"
        );
    }

    #[test]
    fn workers_ai_rejects_a_system_one_url_left_over_from_kev() {
        let cli = parse(&[
            "--explain-backend",
            "openai-compatible",
            "--system-one-backend",
            "workers-ai",
            "--system-one-model",
            "@cf/cloudflare/clef-flash",
            "--system-one-account-id",
            "acc",
            "--system-one-key",
            "token",
            "--system-one-url",
            "http://127.0.0.1:8009",
        ]);
        assert_eq!(
            SystemOneBackend::try_from(&cli.system_one)
                .err()
                .unwrap()
                .to_string(),
            "DREW_SYSTEM_ONE_URL is set, but DREW_SYSTEM_ONE_BACKEND=workers-ai does not use it; unset it"
        );
        assert!(system_one_client(&cli.system_one).is_err());
    }

    #[test]
    fn an_endpoint_keeps_scheme_host_port_and_path_only() {
        let base: Url = "http://user:secret@127.0.0.1:8009/root/?key=secret#frag"
            .parse()
            .unwrap();
        assert_eq!(
            Endpoint::new(&base, DECIDE_PATH).to_string(),
            "http://127.0.0.1:8009/root/v1/systemone"
        );
        let base: Url = "https://system-one.dev".parse().unwrap();
        assert_eq!(
            Endpoint::new(&base, DECIDE_PATH).to_string(),
            "https://system-one.dev/v1/systemone"
        );
    }

    #[test]
    fn the_embed_batch_budget_is_a_positive_integer_for_openai_compatible_only() {
        for bad in ["0", "-1", "50k"] {
            assert!(
                try_parse(&[
                    "--explain-backend",
                    "bedrock",
                    "--system-one-backend",
                    "none",
                    "--embed-max-batch-tokens",
                    bad,
                ])
                .is_err(),
                "{bad}"
            );
        }
        let mut cli = parse(&[
            "--explain-backend",
            "bedrock",
            "--system-one-backend",
            "none",
            "--embed-url",
            "http://localhost:11434/v1",
            "--embed-max-batch-tokens",
            "50000",
        ]);
        assert!(matches!(
            EmbedBackend::try_from(&cli.embed),
            Ok(EmbedBackend::OpenAiCompatible {
                max_batch_tokens: Some(budget),
                ..
            }) if budget.get() == 50_000
        ));
        cli.embed.embed_backend = EmbedBackendKind::Bedrock;
        cli.embed.embed_url = None;
        assert_eq!(
            EmbedBackend::try_from(&cli.embed)
                .err()
                .unwrap()
                .to_string(),
            "DREW_EMBED_MAX_BATCH_TOKENS is set, but DREW_EMBED_BACKEND=bedrock does not use it; unset it"
        );
    }

    #[test]
    fn a_bad_url_fails_at_parse_time() {
        assert!(
            try_parse(&[
                "--explain-backend",
                "openai-compatible",
                "--explain-url",
                "localhost 11434",
                "--system-one-backend",
                "none",
            ])
            .is_err()
        );
    }

    #[test]
    fn an_empty_key_is_rejected_at_parse_time() {
        assert!(
            try_parse(&[
                "--explain-backend",
                "bedrock",
                "--system-one-backend",
                "system-one",
                "--system-one-key",
                " ",
            ])
            .is_err()
        );
    }

    /// Embeddings and Explain on Workers AI, plus `extra`.
    fn try_workers_ai(embed_model: &str, extra: &[&str]) -> Result<Cli, clap::Error> {
        let base = [
            "drew",
            "--embed-backend",
            "workers-ai",
            "--embed-model",
            embed_model,
            "--embed-account-id",
            "acc",
            "--embed-key",
            "token",
            "--explain-backend",
            "workers-ai",
            "--explain-model",
            "@cf/qwen/qwen3.8-27b",
            "--explain-account-id",
            "acc",
            "--explain-key",
            "token",
            "--explain-reasoning-effort",
            "medium",
            "--system-one-backend",
            "none",
        ];
        Cli::try_parse_from(base.iter().chain(extra))
    }

    #[tokio::test]
    async fn workers_ai_builds_the_cloudflare_urls_from_the_account_id() {
        let cli = try_workers_ai("@cf/baai/bge-m3", &[]).unwrap();

        let embed = EmbedBackend::try_from(&cli.embed).unwrap();
        assert!(matches!(
            embed,
            EmbedBackend::WorkersAi {
                model: WorkersAiEmbedModel::BgeM3,
                ..
            }
        ));
        assert_eq!(
            embed.endpoint().await.unwrap().to_string(),
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/baai/bge-m3"
        );
        let chat = ChatBackend::try_from(&cli.chat).unwrap();
        assert_eq!(
            chat.endpoint().await.unwrap().to_string(),
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/v1/chat/completions"
        );
        assert_eq!(
            explain_model(&cli.chat).await.unwrap().model().version,
            "openai-compatible;reasoning_effort=medium"
        );
        assert_eq!(
            embedder(&cli.embed).await.unwrap().model(),
            crate::llm::ModelRef::new("@cf/baai/bge-m3", "dimensions=1024")
        );
    }

    #[test]
    fn workers_ai_rejects_a_url_and_a_batch_budget_as_unused() {
        for (extra, message) in [
            (
                ["--embed-url", "http://localhost:11434/v1"],
                "DREW_EMBED_URL is set, but DREW_EMBED_BACKEND=workers-ai does not use it; unset it",
            ),
            (
                ["--embed-max-batch-tokens", "50000"],
                "DREW_EMBED_MAX_BATCH_TOKENS is set, but DREW_EMBED_BACKEND=workers-ai does not use it; unset it",
            ),
        ] {
            let cli = try_workers_ai("@cf/baai/bge-m3", &extra).unwrap();
            assert_eq!(
                EmbedBackend::try_from(&cli.embed)
                    .err()
                    .unwrap()
                    .to_string(),
                message
            );
        }
        let cli = try_workers_ai(
            "@cf/baai/bge-m3",
            &["--explain-url", "http://localhost:11434/v1"],
        )
        .unwrap();
        assert_eq!(
            ChatBackend::try_from(&cli.chat).err().unwrap().to_string(),
            "DREW_EXPLAIN_URL is set, but DREW_EXPLAIN_BACKEND=workers-ai does not use it; unset it"
        );
    }

    #[test]
    fn workers_ai_names_what_it_lacks_or_cannot_parse() {
        let mut cli = try_workers_ai("@cf/baai/bge-m3", &[]).unwrap();
        cli.embed.embed_account_id = None;
        cli.chat.explain_key = None;
        assert_eq!(
            missing(EmbedBackend::try_from(&cli.embed).err().unwrap()),
            "DREW_EMBED_ACCOUNT_ID"
        );
        assert_eq!(
            missing(ChatBackend::try_from(&cli.chat).err().unwrap()),
            "DREW_EXPLAIN_KEY"
        );

        let cli = try_workers_ai("bge-m3", &[]).unwrap();
        let error = EmbedBackend::try_from(&cli.embed)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.starts_with("DREW_EMBED_MODEL: ")
                && error.ends_with("supported: @cf/baai/bge-m3"),
            "{error}"
        );
        let mut cli = try_workers_ai("@cf/baai/bge-m3", &[]).unwrap();
        cli.chat.explain_model = "qwen3:14b".into();
        assert!(
            ChatBackend::try_from(&cli.chat)
                .err()
                .unwrap()
                .to_string()
                .starts_with("DREW_EXPLAIN_MODEL: ")
        );
        assert!(try_workers_ai("@cf/baai/bge-m3", &["--embed-account-id", "acc/../x"]).is_err());
    }

    #[test]
    fn an_account_id_is_only_for_workers_ai() {
        let cli = parse(&[
            "--embed-url",
            "http://localhost:11434/v1",
            "--embed-account-id",
            "acc",
            "--explain-backend",
            "bedrock",
            "--explain-account-id",
            "acc",
            "--system-one-backend",
            "none",
        ]);
        assert!(matches!(
            EmbedBackend::try_from(&cli.embed),
            Err(InvalidBackend::Unused {
                var: "DREW_EMBED_ACCOUNT_ID",
                ..
            })
        ));
        assert!(matches!(
            ChatBackend::try_from(&cli.chat),
            Err(InvalidBackend::Unused {
                var: "DREW_EXPLAIN_ACCOUNT_ID",
                ..
            })
        ));
    }
}
