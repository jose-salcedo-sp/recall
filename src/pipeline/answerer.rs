//! The real `Answerer`: classify and retrieve in parallel, join on the category, then drive its pipeline.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::{
    config::{ConfigArgs, PipelineConfig, StepName},
    domain::{CategoryKind, Chunk, Provenance, Question, RunId},
    llm::{
        self, ChatModel, ModelRef,
        backend::{ChatArgs, EmbedArgs, SystemOneArgs},
        system_one::SystemOne,
    },
    retrieval::{Retrieved, Retriever, hybrid::HybridRetriever},
};

use super::{
    Boolean, Category, Choice, Halt, Halted, Open, Pipeline, Ranking, Run, Seq, Unclassified,
    report::{Answer, Answerer, Citation, Outcome, RunReport, Verdict},
    steps::{
        Answer as AnswerStep, Candidates, Classify, Decisive, Explain, OpenEnded, Rerank, Reranked,
        Retrieve, Verified, Verify,
    },
    trace::Trace,
};

const EXCERPT_CHARS: usize = 200;

/// The one place where runtime data becomes a type. See `docs/design.md` section 5.4.
pub enum Classified {
    /// Continues through [`Decisive<Boolean>`].
    Boolean(Run<Boolean, Reranked>),
    /// Not produced until `Answer<Choice>` exists; `choice` questions take the `Open` arm.
    Choice(Run<Choice, Reranked>),
    /// Not produced until `Answer<Ranking>` exists; `ranking` questions take the `Open` arm.
    Ranking(Run<Ranking, Reranked>),
    /// Continues through [`OpenEnded`].
    Open(Run<Open, Reranked>),
}

/// Everything [`PipelineAnswerer::from_args`] needs besides the database.
#[derive(Clone, Debug, clap::Args)]
pub struct PipelineArgs {
    /// The tunables file.
    #[command(flatten)]
    pub config: ConfigArgs,
    /// The query embedder; must match the corpus's model.
    #[command(flatten)]
    pub embed: EmbedArgs,
    /// The `Explain` LLM.
    #[command(flatten)]
    pub chat: ChatArgs,
    /// System One, for `Classify`, `Answer`, and `Rerank` when `rerank.judge` names it.
    #[command(flatten)]
    pub system_one: SystemOneArgs,
}

/// The models a pipeline calls, recorded with every `drew eval` run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// `None` when the retriever was built outside `from_args`.
    pub embedding: Option<ModelRef>,
    /// The `Explain` LLM.
    pub explain: ModelRef,
    /// Classify and Answer, and Rerank when `rerank` is set. `None`: every question is `open`.
    pub system_one: Option<ModelRef>,
    /// The Rerank judge. `None`: reranking keeps retrieval order, because `rerank.judge` is `none` or System One
    /// is not configured. Also `None` in files written before it was recorded, whose config says who judged.
    #[serde(default)]
    pub rerank: Option<ModelRef>,
}

/// The production [`Answerer`]: builds every step and pipeline once, then answers questions concurrently.
pub struct PipelineAnswerer {
    config: PipelineConfig,
    embedding: Option<ModelRef>,
    explain: Arc<dyn ChatModel>,
    system_one: Option<Arc<dyn SystemOne>>,
    classify: Classify,
    retrieve: Retrieve,
    rerank_open: Rerank<Open>,
    rerank_boolean: Rerank<Boolean>,
    open: OpenEnded,
    boolean: Decisive<Boolean>,
}

impl PipelineAnswerer {
    /// Classifies every question as `open`; see `with_system_one`.
    pub fn new(retriever: Arc<dyn Retriever>, explain: Arc<dyn ChatModel>) -> Self {
        Self::build(
            PipelineConfig::default(),
            None,
            Retrieve::new(retriever),
            explain,
            None,
        )
    }

    fn build(
        config: PipelineConfig,
        embedding: Option<ModelRef>,
        retrieve: Retrieve,
        explain: Arc<dyn ChatModel>,
        system_one: Option<Arc<dyn SystemOne>>,
    ) -> Self {
        let threshold = |category, step| config.threshold(category, step);
        Self {
            classify: Classify::new(system_one.clone())
                .with_min_confidence(config.classify.min_confidence),
            retrieve,
            rerank_open: Rerank::from_config(
                &config.rerank,
                threshold(CategoryKind::Open, StepName::Rerank),
                system_one.clone(),
            ),
            rerank_boolean: Rerank::from_config(
                &config.rerank,
                threshold(CategoryKind::Boolean, StepName::Rerank),
                system_one.clone(),
            ),
            open: Seq(
                Explain::new(Arc::clone(&explain), config.explain.max_tokens),
                Verify::new(),
            ),
            boolean: Seq(
                AnswerStep::new(
                    system_one.clone(),
                    threshold(CategoryKind::Boolean, StepName::Answer),
                ),
                Seq(
                    Explain::new(Arc::clone(&explain), config.explain.max_tokens),
                    Verify::new(),
                ),
            ),
            config,
            embedding,
            explain,
            system_one,
        }
    }

    /// Classify and Answer on System One, and Rerank when `rerank.judge` names it.
    pub fn with_system_one(self, system_one: Arc<dyn SystemOne>) -> Self {
        Self::build(
            self.config,
            self.embedding,
            self.retrieve,
            self.explain,
            Some(system_one),
        )
    }

    /// Rebuilds the steps with `config`'s thresholds and rerank settings. Retrieval settings are fixed when the
    /// retriever is built.
    pub fn with_config(self, config: PipelineConfig) -> Self {
        Self::build(
            config,
            self.embedding,
            self.retrieve,
            self.explain,
            self.system_one,
        )
    }

    /// The config the steps were built with.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// The models the steps call.
    pub fn models(&self) -> Models {
        Models {
            embedding: self.embedding.clone(),
            explain: self.explain.model(),
            system_one: self
                .system_one
                .as_ref()
                .map(|system_one| system_one.model()),
            rerank: self.rerank_open.judged_by(),
        }
    }

    /// Hybrid retrieval and `Explain` on Bedrock or OpenAI-compatible servers; `Classify` and `Answer` on System
    /// One when configured, and `Rerank` too when `rerank.judge` names it. Every tunable comes from the pipeline
    /// config file.
    ///
    /// # Errors
    ///
    /// When the config does not load or a model client cannot be built.
    pub async fn from_args(pool: PgPool, args: &PipelineArgs) -> eros::Result<Self> {
        let config = args.config.load()?;
        let embedder = llm::embedder(&args.embed).await?;
        let embedding = Some(embedder.model());
        let retriever = HybridRetriever::new(pool, embedder, config.retrieval.clone());
        let explain = llm::explain_model(&args.chat).await?;
        let system_one = llm::system_one_client(&args.system_one)?;
        Ok(Self::build(
            config,
            embedding,
            Retrieve::new(Arc::new(retriever)),
            explain,
            system_one,
        ))
    }

    /// `choice` and `ranking` take the open path until their `Answer` steps exist; the returned category is still
    /// the one `Classify` chose.
    async fn classify(&self, question: Question) -> Result<(CategoryKind, Classified), Halted> {
        let run = Run::new(question)
            .both("classify_retrieve", &self.classify, &self.retrieve)
            .await?;
        let category = run.state().0;
        let run: Run<Unclassified, Candidates> = run.map(|(_, candidates)| candidates);
        let classified = match category {
            CategoryKind::Boolean => Classified::Boolean(
                run.classified::<Boolean>()
                    .then(&self.rerank_boolean)
                    .await?,
            ),
            CategoryKind::Choice | CategoryKind::Ranking | CategoryKind::Open => {
                Classified::Open(run.classified::<Open>().then(&self.rerank_open).await?)
            }
        };
        Ok((category, classified))
    }
}

#[async_trait]
impl Answerer for PipelineAnswerer {
    async fn answer(&self, question: Question) -> RunReport {
        let (category, classified) = match self.classify(question).await {
            Ok(classified) => classified,
            Err(halted) => return halted_report(halted, &[]),
        };
        // `Halted` carries no state, so a later decline lists the sources kept here.
        let mut report = match classified {
            Classified::Boolean(run) => {
                let sources = run.state().sources().to_vec();
                match self.boolean.drive(run).await {
                    Ok(run) => {
                        let (id, question, decided, trace) = run.into_parts();
                        let verdict = Verdict::Boolean {
                            yes: decided.yes(),
                            confidence: decided.confidence(),
                        };
                        answered_report(id, question, decided.into_verified(), trace, Some(verdict))
                    }
                    Err(halted) => halted_report(halted, &sources),
                }
            }
            Classified::Choice(run) => halted_report(answer_not_implemented(run), &[]),
            Classified::Ranking(run) => halted_report(answer_not_implemented(run), &[]),
            Classified::Open(run) => {
                let sources = run.state().sources().to_vec();
                match self.open.drive(run).await {
                    Ok(run) => {
                        let (id, question, verified, trace) = run.into_parts();
                        answered_report(id, question, verified, trace, None)
                    }
                    Err(halted) => halted_report(halted, &sources),
                }
            }
        };
        report.category = Some(category);
        report
    }
}

fn answer_not_implemented<TCategory: Category>(run: Run<TCategory, Reranked>) -> Halted {
    let (id, question, _, trace) = run.into_parts();
    Halted {
        id,
        question,
        trace,
        category: TCategory::KIND,
        at: "answer",
        halt: Halt::Failed {
            error: "Answer is not implemented until milestone 6".into(),
        },
    }
}

fn answered_report(
    id: RunId,
    question: Question,
    verified: Verified,
    trace: Trace,
    verdict: Option<Verdict>,
) -> RunReport {
    let (answer, sources) = verified.into_parts();
    let why = answer.into_claims();
    let mut cited = HashSet::new();
    let citations = why
        .iter()
        .flat_map(|claim| claim.sources.as_slice())
        .filter_map(|source| match source {
            Provenance::Chunk(id) => Some(id),
            Provenance::UserPrompt(_) => None,
        })
        .filter(|id| cited.insert(*id))
        .filter_map(|id| sources.sources().iter().find(|hit| &hit.chunk.id == id))
        .map(|hit| citation(&hit.chunk))
        .collect();
    RunReport {
        id,
        question,
        category: None,
        outcome: Outcome::Answered(Answer {
            verdict,
            why,
            citations,
        }),
        trace,
    }
}

fn halted_report(halted: Halted, sources: &[Retrieved]) -> RunReport {
    let at = halted.at.to_owned();
    let outcome = match halted.halt {
        Halt::Declined {
            reason,
            score,
            threshold,
        } => Outcome::Declined {
            at,
            reason,
            score,
            threshold,
            sources: sources.iter().map(|hit| citation(&hit.chunk)).collect(),
        },
        Halt::Failed { error } => Outcome::Failed { at, error },
    };
    RunReport {
        id: halted.id,
        question: halted.question,
        category: halted.category,
        outcome,
        trace: halted.trace,
    }
}

fn citation(chunk: &Chunk) -> Citation {
    Citation {
        chunk: chunk.reference(),
        title: chunk.title.clone(),
        url: chunk.url.clone(),
        excerpt: excerpt(&chunk.text),
    }
}

fn excerpt(text: &str) -> String {
    let text = text.trim();
    match text.char_indices().nth(EXCERPT_CHARS) {
        Some((end, _)) => format!("{}…", text[..end].trim_end()),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests;
