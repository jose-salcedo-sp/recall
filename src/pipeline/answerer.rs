//! The real `Answerer`: classify and retrieve in parallel, join on the category, rerank and, when configured, admit
//! the sources, then drive the category's pipeline.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    access::{Authorizer, PolicyRef, Principal},
    config::PipelineConfig,
    domain::{CategoryKind, Chunk, Provenance, Question, RunId},
    llm::{ModelRef, judge::Judge},
    retrieval::Retrieved,
};

use super::{
    Boolean, Category, Choice, Halt, Halted, Open, Pipeline, Ranking, Run, Unclassified,
    assembly::{self, Assembled, Path, Plan, Ports, Unassembled},
    report::{Answer, Answerer, Citation, Outcome, RunReport, Verdict},
    steps::{Admit, Candidates, Decisive, Reranked, Verified},
    trace::Trace,
};

const EXCERPT_CHARS: usize = 200;

/// The one place where runtime data becomes a type. See `docs/design.md` section 5.4.
pub enum Classified<'a> {
    /// Continues through the [`Decisive<Boolean>`] it carries, which exists only when a judge classifies.
    Boolean(Run<Boolean, Reranked>, &'a Decisive<Boolean>),
    /// Not produced until `Answer<Choice>` exists; `choice` questions take the `Open` arm.
    Choice(Run<Choice, Reranked>),
    /// Not produced until `Answer<Ranking>` exists; `ranking` questions take the `Open` arm.
    Ranking(Run<Ranking, Reranked>),
    /// Continues through [`OpenEnded`](super::steps::OpenEnded).
    Open(Run<Open, Reranked>),
}

/// The models a pipeline calls, recorded with every `recall eval` run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The retriever's embedding model; `None` when [`Ports::embedding`] does not name it.
    pub embedding: Option<ModelRef>,
    /// The `Explain` LLM.
    pub explain: ModelRef,
    /// The judge: Classify and Answer, and every check whose config names a judge. `None`: every question is
    /// `open`. Files written before it was renamed call it `system_one`.
    #[serde(alias = "system_one")]
    pub judge: Option<ModelRef>,
    /// The Rerank judge. `None`: reranking keeps retrieval order, because `rerank.judge` is `none`. Also `None` in
    /// files written before it was recorded, whose config says who judged.
    #[serde(default)]
    pub rerank: Option<ModelRef>,
    /// The `Admit` judge. `None`: runs have no `admit` step, because `admit.judge` is `none`. Also `None` in files
    /// written before admission existed.
    #[serde(default)]
    pub admit: Option<ModelRef>,
    /// The `Verify` support judge. `None`: claims are checked by membership, or quoted spans, only.
    #[serde(default)]
    pub verify: Option<ModelRef>,
    /// The access policy. `None` in files written before access was enforced.
    #[serde(default)]
    pub policy: Option<PolicyRef>,
}

/// The production [`Answerer`]: assembles every step and pipeline once, then answers questions concurrently.
pub struct PipelineAnswerer {
    config: PipelineConfig,
    ports: Ports,
    steps: Assembled,
}

impl PipelineAnswerer {
    /// Every step `config` names, built from `ports`.
    ///
    /// # Errors
    ///
    /// [`Unassembled`] when a check `config` names needs a judge and `ports` has none, or a rerank threshold is set
    /// while rerank is unscored.
    pub fn assemble(config: PipelineConfig, ports: Ports) -> Result<Self, Unassembled> {
        let steps = assembly::assemble(&config, &ports)?;
        Ok(Self {
            config,
            ports,
            steps,
        })
    }

    /// The default config over `ports`, judge included: with a judge, Classify and Answer run on it; without one,
    /// every question is `open`.
    pub fn new(ports: Ports) -> Self {
        Self::assemble(PipelineConfig::default(), ports)
            .expect("the default config names no judged check and no rerank threshold")
    }

    /// Reassembles with `judge`: Classify and Answer on it, and every check whose config names a judge.
    pub fn with_judge(self, judge: Arc<dyn Judge>) -> Self {
        let ports = Ports {
            judge: Some(judge),
            ..self.ports
        };
        Self::assemble(self.config, ports)
            .expect("a config that assembled without a judge assembles with one")
    }

    /// Reassembles with `config`. Retrieval settings are fixed when the retriever is built.
    ///
    /// # Errors
    ///
    /// As [`PipelineAnswerer::assemble`].
    pub fn with_config(self, config: PipelineConfig) -> Result<Self, Unassembled> {
        Self::assemble(config, self.ports)
    }

    /// The config the steps were built with.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// What each step does, as assembled.
    pub fn plan(&self) -> Plan {
        self.steps.plan
    }

    /// The policy every run is authorized by; the server reads persisted runs through it too.
    pub fn authorizer(&self) -> Arc<dyn Authorizer> {
        Arc::clone(&self.ports.authorizer)
    }

    /// The models the steps call.
    pub fn models(&self) -> Models {
        Models {
            embedding: self.ports.embedding.clone(),
            explain: self.ports.explain.model(),
            judge: self.ports.judge.as_ref().map(|judge| judge.model()),
            rerank: self.steps.open.rerank.judged_by(),
            admit: self.steps.open.admit.as_ref().map(Admit::judged_by),
            verify: self.steps.open.pipeline.1.judged_by(),
            policy: Some(self.ports.authorizer.policy()),
        }
    }

    /// `choice` and `ranking` take the open path until their `Answer` steps exist; the returned category is still
    /// the one `Classify` chose.
    async fn classify(
        &self,
        principal: &Principal,
        question: Question,
    ) -> Result<(CategoryKind, Classified<'_>), Halted> {
        let steps = &self.steps;
        let run = Run::authorize(
            self.ports.ids.next(),
            question,
            principal.clone(),
            self.ports.authorizer.as_ref(),
            Arc::clone(&self.ports.clock),
        )
        .await?
        .both("classify_retrieve", &steps.classify, &steps.retrieve)
        .await?;
        let category = run.state().0;
        let run: Run<Unclassified, Candidates> = run.map(|(_, candidates)| candidates);
        let classified = match (category, &steps.boolean) {
            (CategoryKind::Boolean, Some(boolean)) => Classified::Boolean(
                reranked(run.classified(), boolean).await?,
                &boolean.pipeline,
            ),
            (CategoryKind::Boolean, None)
            | (CategoryKind::Choice | CategoryKind::Ranking | CategoryKind::Open, _) => {
                Classified::Open(reranked(run.classified(), &steps.open).await?)
            }
        };
        Ok((category, classified))
    }
}

#[async_trait]
impl Answerer for PipelineAnswerer {
    async fn answer(&self, principal: &Principal, question: Question) -> RunReport {
        let (category, classified) = match self.classify(principal, question).await {
            Ok(classified) => classified,
            Err(halted) => return halted_report(halted, &[]),
        };
        // `Halted` carries no state, so a later decline lists the sources kept here.
        let mut report = match classified {
            Classified::Boolean(run, boolean) => {
                let sources = run.state().sources().to_vec();
                match boolean.drive(run).await {
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
                match self.steps.open.pipeline.drive(run).await {
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

/// The run through `path`'s rerank, then its `admit` step when the pipeline has one.
async fn reranked<TCategory: Category, TPipeline>(
    run: Run<TCategory, Candidates>,
    path: &Path<TCategory, TPipeline>,
) -> Result<Run<TCategory, Reranked>, Halted> {
    let run = run.then(&path.rerank).await?;
    match &path.admit {
        Some(admit) => run.then(admit).await,
        None => Ok(run),
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
