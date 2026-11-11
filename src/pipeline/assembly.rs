//! Pipeline assembly: the config resolved against the model backends available, once, before any question.
//!
//! A check the config names runs as named, or the pipeline does not assemble: `rerank.judge`, `admit.judge` or
//! `verify.support` asking for a judge the pipeline does not have is an [`Unassembled`] error, never a quieter
//! check, and so is a rerank threshold on an unscored rerank. Without a judge every question is classified `open`;
//! that follows from the backends, since no config key asks for classification. What was assembled is the [`Plan`],
//! and each step's trace record says the mode it ran in.

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    access::Authorizer,
    clock::{Clock, RunIds},
    config::{AdmitJudge, PipelineConfig, RerankJudge, StepName, SupportCheck, SupportCutoff},
    domain::{AtLeastOne, CategoryKind},
    llm::{ChatModel, ModelRef, judge::Judge},
    retrieval::Retriever,
};

use super::{
    Boolean, Open, Seq,
    steps::{
        Admit, Answer, Classify, Decisive, Explain, OpenEnded, Rerank, RerankBy, Retrieve, Support,
        Verify,
    },
    trace::{ClassifyMode, RerankMode, SupportMode},
};

/// The model backends a pipeline is assembled from.
#[derive(Clone)]
pub struct Ports {
    /// Finds the candidate chunks.
    pub retriever: Arc<dyn Retriever>,
    /// The retriever's embedding model, recorded in [`super::answerer::Models`]; `None` when it is not known.
    pub embedding: Option<ModelRef>,
    /// The `Explain` LLM.
    pub explain: Arc<dyn ChatModel>,
    /// Classifies, answers yes or no, and judges for every check whose config names a judge. `None`: every
    /// question is `open`, and a config whose checks need a judge does not assemble.
    pub judge: Option<Arc<dyn Judge>>,
    /// Decides what each run's principal may read; asked once per run, before retrieval.
    pub authorizer: Arc<dyn Authorizer>,
    /// Stamps every trace entry.
    pub clock: Arc<dyn Clock>,
    /// Names every run.
    pub ids: Arc<dyn RunIds>,
}

/// What each step does, as assembled: the config's checks exactly, since a check that cannot run as configured
/// fails assembly. Recorded with every `recall eval` run and logged by `recall serve` at startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Whether a judge classifies, or every question is `open`.
    pub classify: ClassifyMode,
    /// Whether a judge orders the matches.
    pub rerank: RerankMode,
    /// Whether runs have an `admit` step.
    pub admit: AdmitMode,
    /// The check `Verify` gives each cited claim.
    pub support: SupportMode,
}

/// Whether runs have an `admit` step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmitMode {
    /// No `admit` step: every reranked source reaches `Explain`.
    Off,
    /// A judge screens every reranked source.
    Judged,
}

/// A config check that needs a judge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JudgedCheck {
    /// `rerank.judge`.
    Rerank,
    /// `admit.judge`.
    Admit,
    /// `verify.support`.
    Support,
}

impl JudgedCheck {
    /// The config key.
    pub fn key(self) -> &'static str {
        match self {
            Self::Rerank => "rerank.judge",
            Self::Admit => "admit.judge",
            Self::Support => "verify.support",
        }
    }

    /// The values of [`JudgedCheck::key`] that need no judge.
    fn without_judge(self) -> &'static str {
        match self {
            Self::Rerank | Self::Admit => r#""none""#,
            Self::Support => r#""quotes" or "membership""#,
        }
    }
}

impl fmt::Display for JudgedCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} needs one (or set it to {})",
            self.key(),
            self.without_judge()
        )
    }
}

/// A config the backends available cannot run as written.
#[derive(Debug, thiserror::Error)]
pub enum Unassembled {
    /// Checks the config names that need a judge, with no judge configured.
    #[error("no judge is configured, but {}; configure a judge backend", list(.0))]
    NoJudge(AtLeastOne<JudgedCheck>),
    /// A rerank threshold while `rerank.judge` is `"none"`, which leaves the step unscored.
    #[error(
        r#"a rerank threshold is set, but rerank.judge is "none", so rerank is unscored; remove the threshold or name a judge"#
    )]
    UnscoredRerank,
}

impl Unassembled {
    /// The checks that need a judge; empty for any other error.
    pub fn checks(&self) -> &[JudgedCheck] {
        match self {
            Self::NoJudge(checks) => checks.as_slice(),
            Self::UnscoredRerank => &[],
        }
    }
}

fn list(checks: &AtLeastOne<JudgedCheck>) -> String {
    checks
        .as_slice()
        .iter()
        .map(JudgedCheck::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// The steps a category runs after `Classify` and `Retrieve`.
pub(super) struct Path<TCategory, TPipeline> {
    pub(super) rerank: Rerank<TCategory>,
    /// `None` under [`AdmitMode::Off`].
    pub(super) admit: Option<Admit<TCategory>>,
    pub(super) pipeline: TPipeline,
}

/// Every step, built once.
pub(super) struct Assembled {
    pub(super) plan: Plan,
    pub(super) classify: Classify,
    pub(super) retrieve: Retrieve,
    pub(super) open: Path<Open, OpenEnded>,
    /// Set exactly when a judge classifies, since only a judged `Classify` chooses `boolean`.
    pub(super) boolean: Option<Path<Boolean, Decisive<Boolean>>>,
}

/// Builds every step `config` names from `ports`.
pub(super) fn assemble(config: &PipelineConfig, ports: &Ports) -> Result<Assembled, Unassembled> {
    let judge = ports.judge.as_ref();
    let rerank_judge = match (config.rerank.judge(), judge) {
        (RerankJudge::None, _) => Ok(None),
        (RerankJudge::Judged(_), Some(judge)) => Ok(Some(judge)),
        (RerankJudge::Judged(_), None) => Err(JudgedCheck::Rerank),
    };
    let admit = match (config.admit.judge, judge) {
        (AdmitJudge::None, _) => Ok(None),
        (AdmitJudge::Judged(cutoffs), Some(judge)) => Ok(Some((judge, cutoffs))),
        (AdmitJudge::Judged(_), None) => Err(JudgedCheck::Admit),
    };
    let support = match (config.verify.support, judge) {
        (SupportCheck::Membership, _) => Ok(Support::Membership),
        (SupportCheck::Quotes, _) => Ok(Support::Quotes),
        (SupportCheck::Judged(SupportCutoff { min_support }), Some(judge)) => Ok(Support::Judged {
            judge: Arc::clone(judge),
            min_support,
        }),
        (SupportCheck::Judged(_), None) => Err(JudgedCheck::Support),
    };
    let (rerank_judge, admit, support) = match (rerank_judge, admit, support) {
        (Ok(rerank_judge), Ok(admit), Ok(support)) => (rerank_judge, admit, support),
        (rerank_judge, admit, support) => {
            let missing = [rerank_judge.err(), admit.err(), support.err()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            return Err(Unassembled::NoJudge(
                AtLeastOne::try_from(missing)
                    .expect("the first arm matches when every check resolved"),
            ));
        }
    };
    if rerank_judge.is_none()
        && CategoryKind::ALL
            .into_iter()
            .any(|category| config.threshold(category, StepName::Rerank).is_some())
    {
        return Err(Unassembled::UnscoredRerank);
    }

    let open = Path {
        rerank: rerank(config, rerank_judge, CategoryKind::Open),
        admit: admit.map(|(judge, cutoffs)| Admit::new(Arc::clone(judge), cutoffs)),
        pipeline: Seq(
            explain(config, ports),
            Verify::new(support.clone(), config.verify.publish),
        ),
    };
    let boolean = judge.map(|judge| Path {
        rerank: rerank(config, rerank_judge, CategoryKind::Boolean),
        admit: admit.map(|(judge, cutoffs)| Admit::new(Arc::clone(judge), cutoffs)),
        pipeline: Seq(
            Answer::new(
                Arc::clone(judge),
                config.threshold(CategoryKind::Boolean, StepName::Answer),
            ),
            Seq(
                explain(config, ports),
                Verify::new(support, config.verify.publish),
            ),
        ),
    });
    let classify = match judge {
        Some(judge) => Classify::judged(Arc::clone(judge), config.classify.min_confidence),
        None => Classify::all_open(),
    };
    let plan = Plan {
        classify: classify.mode(),
        rerank: open.rerank.mode(),
        admit: match open.admit {
            Some(_) => AdmitMode::Judged,
            None => AdmitMode::Off,
        },
        support: open.pipeline.1.support(),
    };
    Ok(Assembled {
        plan,
        classify,
        retrieve: Retrieve::new(Arc::clone(&ports.retriever)),
        open,
        boolean,
    })
}

fn rerank<TCategory>(
    config: &PipelineConfig,
    judge: Option<&Arc<dyn Judge>>,
    category: CategoryKind,
) -> Rerank<TCategory> {
    let by = match judge {
        None => RerankBy::Retrieval,
        Some(judge) => RerankBy::Judge {
            judge: Arc::clone(judge),
            threshold: config.threshold(category, StepName::Rerank),
        },
    };
    Rerank::new(&config.rerank, by)
}

fn explain<TCategory>(config: &PipelineConfig, ports: &Ports) -> Explain<TCategory> {
    Explain::new(Arc::clone(&ports.explain), config.explain.max_tokens)
        .with_instructions(config.explain.instructions.clone())
}

#[cfg(test)]
mod tests;
