//! `drew models check`: one call to each model backend `drew serve` and `drew eval run` would use, with the same
//! arguments, so a wrong URL, key, model id or AWS profile shows in seconds instead of mid-run.
//!
//! Backends are parsed and built by [`crate::llm::backend`], which also says what to print about a backend's
//! ambient settings; this module only calls the ports.

use std::{fmt, time::Duration};

use eros::{Context, ensure};
use serde_json::{Value, json};
use tokio::time::Instant;

use crate::{
    llm::{
        ChatModel, ChatRequest, EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef,
        backend::{ChatBackend, EmbedBackend, Endpoint, SystemOneBackend},
        system_one::{Decision, DecisionRequest, SystemOne},
    },
    pipeline::answerer::PipelineArgs,
};

/// `Explain`'s own budget: a reasoning effort that spends it all before answering fails here too.
const PROBE_MAX_TOKENS: u32 = 2048;
const QUESTION_ID: &str = "check";

/// `drew models` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// `check`.
    #[command(subcommand)]
    pub command: Command,
}

/// The `drew models` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Call the embedder, the Explain LLM and System One once each, and report each one's model, latency and
    /// outcome; exits non-zero when any fails.
    Check(PipelineArgs),
}

/// A model port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    /// [`Embedder`].
    Embeddings,
    /// [`ChatModel`] for `Explain`.
    Explain,
    /// [`SystemOne`].
    SystemOne,
}

impl fmt::Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Embeddings => "embeddings",
            Self::Explain => "explain",
            Self::SystemOne => "system one",
        })
    }
}

/// One port's probe.
#[derive(Debug)]
pub struct Probe {
    /// The port probed.
    pub port: Port,
    /// What happened.
    pub outcome: Outcome,
}

/// Whether the adapter was built, and if so how its one call went.
#[derive(Debug)]
pub enum Outcome {
    /// A backend setting was missing, unparsable or unused, or the client could not be built; nothing was called.
    NotBuilt(eros::ErrorUnion),
    /// The adapter was called once.
    Called {
        /// The adapter's model.
        model: ModelRef,
        /// Where the adapter sent its call; `None` on Bedrock with no AWS region.
        endpoint: Option<Endpoint>,
        /// Time spent in the call.
        latency: Duration,
        /// The call's outcome.
        result: eros::Result<()>,
    },
}

impl Probe {
    /// Not built, or called and failed.
    pub fn failed(&self) -> bool {
        match &self.outcome {
            Outcome::NotBuilt(_) => true,
            Outcome::Called { result, .. } => result.is_err(),
        }
    }
}

/// What `drew models check` found.
#[derive(Debug)]
pub struct Check {
    /// What each backend says about its ambient settings, such as Bedrock's AWS region, or why a port is not called.
    pub notes: Vec<String>,
    /// One per called port, in port order.
    pub probes: Vec<Probe>,
}

impl Check {
    /// How many probes failed.
    pub fn failed(&self) -> usize {
        self.probes.iter().filter(|probe| probe.failed()).count()
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for note in &self.notes {
            writeln!(f, "{note}")?;
        }
        for Probe { port, outcome } in &self.probes {
            match outcome {
                Outcome::NotBuilt(error) => writeln!(f, "{port:<10}  not built\n{error:#?}")?,
                Outcome::Called {
                    model: ModelRef { id, version },
                    endpoint,
                    latency,
                    result,
                } => {
                    write!(f, "{port:<10}  {id} ({version})  ")?;
                    match endpoint {
                        Some(endpoint) => write!(f, "{endpoint}  ")?,
                        None => write!(f, "endpoint unknown  ")?,
                    }
                    write!(f, "{latency:.1?}  ")?;
                    match result {
                        Ok(()) => writeln!(f, "ok")?,
                        Err(error) => writeln!(f, "error\n{error:#?}")?,
                    }
                }
            }
        }
        Ok(())
    }
}

/// Runs `drew models`.
///
/// # Errors
///
/// When any backend fails; each failure is printed with the report.
pub async fn run(args: Args) -> eros::Result<()> {
    match args.command {
        Command::Check(args) => {
            let check = check(&args).await;
            print!("{check}");
            let failed = check.failed();
            ensure!(
                failed == 0,
                "{failed} of {} model backends failed; see above",
                check.probes.len()
            );
            Ok(())
        }
    }
}

/// Calls each backend `args` selects once, concurrently. A port whose backend calls nothing (System One `none`)
/// gets a note instead of a probe.
pub async fn check(args: &PipelineArgs) -> Check {
    let (embeddings, explain, system_one) = tokio::join!(
        probe_embeddings(args),
        probe_explain(args),
        probe_system_one(args),
    );
    let mut notes = Vec::new();
    let mut probes = Vec::new();
    for (note, probe) in [embeddings, explain, system_one] {
        notes.extend(note);
        probes.extend(probe);
    }
    Check { notes, probes }
}

type Probed = (Option<String>, Option<Probe>);

fn not_built(port: Port, error: eros::ErrorUnion) -> Probed {
    (
        None,
        Some(Probe {
            port,
            outcome: Outcome::NotBuilt(error),
        }),
    )
}

/// Times `call`, which does nothing until awaited here.
async fn timed(
    port: Port,
    model: ModelRef,
    endpoint: Option<Endpoint>,
    call: impl Future<Output = eros::Result<()>>,
) -> Probe {
    let start = Instant::now();
    let result = call.await;
    Probe {
        port,
        outcome: Outcome::Called {
            model,
            endpoint,
            latency: start.elapsed(),
            result,
        },
    }
}

async fn probe_embeddings(args: &PipelineArgs) -> Probed {
    let backend = match EmbedBackend::try_from(&args.embed) {
        Ok(backend) => backend,
        Err(error) => return not_built(Port::Embeddings, error.into()),
    };
    let note = backend.note().await;
    let endpoint = backend.endpoint().await;
    let probe = match backend.build().await {
        Ok(embedder) => {
            timed(
                Port::Embeddings,
                embedder.model(),
                endpoint,
                embed_once(&*embedder),
            )
            .await
        }
        Err(error) => return (note, not_built(Port::Embeddings, error).1),
    };
    (note, Some(probe))
}

#[eros::context("embedding one query")]
async fn embed_once(embedder: &dyn Embedder) -> eros::Result<()> {
    let vectors = embedder
        .embed(&["drew models check".to_owned()], EmbedPurpose::Query)
        .await?;
    let dimensions: Vec<usize> = vectors.iter().map(Vec::len).collect();
    ensure!(
        dimensions == [EMBEDDING_DIM],
        "expected one {EMBEDDING_DIM}-dimension vector, got dimensions {dimensions:?}"
    );
    Ok(())
}

async fn probe_explain(args: &PipelineArgs) -> Probed {
    let backend = match ChatBackend::try_from(&args.chat) {
        Ok(backend) => backend,
        Err(error) => return not_built(Port::Explain, error.into()),
    };
    let note = backend.note().await;
    let endpoint = backend.endpoint().await;
    let probe = match backend.build().await {
        Ok(chat) => timed(Port::Explain, chat.model(), endpoint, explain_once(&*chat)).await,
        Err(error) => return (note, not_built(Port::Explain, error).1),
    };
    (note, Some(probe))
}

#[eros::context("asking for a one-field JSON reply")]
async fn explain_once(chat: &dyn ChatModel) -> eros::Result<()> {
    let response = chat
        .complete(ChatRequest {
            system: "Reply with JSON that matches the schema.".to_owned(),
            user: "Is this message a connectivity check? Answer in `ok`.".to_owned(),
            max_tokens: PROBE_MAX_TOKENS,
            temperature: 0.0,
            json_schema: Some(json!({
                "type": "object",
                "properties": { "ok": { "type": "boolean" } },
                "required": ["ok"],
                "additionalProperties": false,
            })),
        })
        .await?;
    let reply: Value = serde_json::from_str(&response.text).context("the reply is not JSON")?;
    ensure!(
        reply["ok"].is_boolean(),
        "the reply does not match the schema: {}",
        response.text
    );
    Ok(())
}

async fn probe_system_one(args: &PipelineArgs) -> Probed {
    let backend = match SystemOneBackend::try_from(&args.system_one) {
        Ok(backend) => backend,
        Err(error) => return not_built(Port::SystemOne, error.into()),
    };
    let note = backend.note();
    let endpoint = backend.endpoint();
    let probe = match backend.build() {
        Ok(Some(system_one)) => Some(
            timed(
                Port::SystemOne,
                system_one.model(),
                endpoint,
                decide_once(&*system_one),
            )
            .await,
        ),
        Ok(None) => None,
        Err(error) => not_built(Port::SystemOne, error).1,
    };
    (note, probe)
}

#[eros::context("asking one yes/no question")]
async fn decide_once(system_one: &dyn SystemOne) -> eros::Result<()> {
    let response = system_one
        .decide(DecisionRequest {
            state: "This message is a connectivity check from drew models check.".to_owned(),
            model: String::new(),
            questions: [(
                QUESTION_ID.to_owned(),
                Decision::Noul {
                    instructions: "Is this message a connectivity check?".to_owned(),
                },
            )]
            .into(),
        })
        .await?;
    response.yes_no(QUESTION_ID)?;
    Ok(())
}
