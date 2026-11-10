use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use futures::{StreamExt, TryStreamExt, stream};

use crate::{
    config::PublishPolicy,
    domain::{AtLeastOne, ChunkId, Cited, Probability, Provenance},
    llm::{
        ModelRef,
        judge::{Judge, JudgeOption, JudgeQuestion, JudgeRequest},
    },
    pipeline::{
        Ask, Boolean, DeclineReason, Open, Scored, Step, StepError,
        trace::{ClaimCheck, ClaimVerdict, Relation, RelationKind, SupportMode, VerifyRecord},
    },
    retrieval::Retrieved,
};

use super::{CONCURRENT_REQUESTS, Explained, ExplainedVerdict, Reranked, judged_text, with_access};

/// Recorded as the trace entry's prompt version when a judge checks support; bump on any change to the question.
pub const SUPPORT_PROMPT_VERSION: &str = "support-v1";

/// In [`RelationKind`] order.
const RELATIONS: [(RelationKind, JudgeOption); 3] = [
    (
        RelationKind::Supports,
        JudgeOption {
            name: "supports",
            description: "The text supports the claim.",
        },
    ),
    (
        RelationKind::Contradicts,
        JudgeOption {
            name: "contradicts",
            description: "The text contradicts the claim.",
        },
    ),
    (
        RelationKind::SaysNothing,
        JudgeOption {
            name: "says_nothing",
            description: "The text is silent on the claim.",
        },
    ),
];

/// Checks every claim before anything is shown. A claim must cite at least one chunk, and only this run's reranked
/// sources; with [`Support::Quotes`] or [`Support::Judged`], every span it puts in backticks or double quotes must
/// also appear in a chunk it cites, and with a judge, a cited chunk must support it. Under [`PublishPolicy::Decline`]
/// any failed claim declines the run; under [`PublishPolicy::Strip`] failed claims are left out, and the run
/// declines only when none is left.
pub struct Verify<TCategory> {
    support: Support,
    publish: PublishPolicy,
    category: PhantomData<TCategory>,
}

/// The check each cited claim gets beyond membership: `verify.support` resolved against the judge the pipeline has.
#[derive(Clone)]
pub enum Support {
    /// Citations only.
    Membership,
    /// Also every quoted span; no model call.
    Quotes,
    /// Also the quote check, then `judge` reads the cited chunks in order until one supports the claim with at least
    /// `min_support`.
    Judged {
        /// Asked one one-of question per cited chunk.
        judge: Arc<dyn Judge>,
        /// The probability of "supports" a claim needs.
        min_support: Probability,
    },
}

impl<TCategory> Verify<TCategory> {
    /// Checks claims with `support`, and publishes by `publish`.
    pub fn new(support: Support, publish: PublishPolicy) -> Self {
        Self {
            support,
            publish,
            category: PhantomData,
        }
    }

    /// The check each cited claim gets beyond membership.
    pub fn support(&self) -> SupportMode {
        match self.support {
            Support::Membership => SupportMode::Membership,
            Support::Quotes => SupportMode::Quotes,
            Support::Judged { .. } => SupportMode::Judged,
        }
    }

    /// The model that judges support; `None` without a judge.
    pub fn judged_by(&self) -> Option<ModelRef> {
        match &self.support {
            Support::Judged { judge, .. } => Some(judge.model()),
            Support::Membership | Support::Quotes => None,
        }
    }
}

/// Output of `Verify`: the answer and the sources its citations point into.
pub struct Verified {
    answer: VerifiedAnswer,
    sources: Reranked,
}

impl Verified {
    /// The answer, and the sources it was checked against.
    pub fn into_parts(self) -> (VerifiedAnswer, Reranked) {
        (self.answer, self.sources)
    }
}

/// At least one claim. Every claim cites at least one chunk, every cited chunk is among the run's reranked sources,
/// and each claim passed the run's support check.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedAnswer {
    claims: Vec<Cited<String>>,
}

/// Output of `Verify<Boolean>`: the verified explanation and the verdict it supports.
pub struct VerifiedVerdict {
    verified: Verified,
    yes: bool,
    confidence: Probability,
}

impl VerifiedVerdict {
    /// The judge's verdict.
    pub fn yes(&self) -> bool {
        self.yes
    }

    /// Probability of the verdict given, so at least `0.5`.
    pub fn confidence(&self) -> Probability {
        self.confidence
    }

    /// The verified explanation, without the verdict.
    pub fn into_verified(self) -> Verified {
        self.verified
    }
}

impl VerifiedAnswer {
    /// In the model's order; each cites at least one of the run's chunks.
    pub fn claims(&self) -> &[Cited<String>] {
        &self.claims
    }

    /// See [`VerifiedAnswer::claims`].
    pub fn into_claims(self) -> Vec<Cited<String>> {
        self.claims
    }
}

impl Step for Verify<Open> {
    const NAME: &'static str = "verify";
    type Category = Open;
    type Input = Explained;
    type Output = Verified;
    type Record = VerifyRecord;

    async fn run(
        &self,
        ask: &Ask,
        input: Explained,
    ) -> Result<Scored<Verified, VerifyRecord>, StepError> {
        self.enforced(ask, input).await
    }
}

impl Step for Verify<Boolean> {
    const NAME: &'static str = "verify";
    type Category = Boolean;
    type Input = ExplainedVerdict;
    type Output = VerifiedVerdict;
    type Record = VerifyRecord;

    async fn run(
        &self,
        ask: &Ask,
        input: ExplainedVerdict,
    ) -> Result<Scored<VerifiedVerdict, VerifyRecord>, StepError> {
        let ExplainedVerdict {
            explained,
            yes,
            confidence,
        } = input;
        let Scored {
            output,
            record,
            score,
            model,
            prompt_version,
            access,
        } = self.enforced(ask, explained).await?;
        Ok(Scored {
            output: VerifiedVerdict {
                verified: output,
                yes,
                confidence,
            },
            record,
            score,
            model,
            prompt_version,
            access,
        })
    }
}

impl<TCategory> Verify<TCategory> {
    /// Rechecks the sources first, so a claim citing a chunk outside the run's scope cites an unknown chunk.
    async fn enforced(
        &self,
        ask: &Ask,
        input: Explained,
    ) -> Result<Scored<Verified, VerifyRecord>, StepError> {
        let Explained {
            explanation,
            sources,
        } = input;
        let (sources, access) = sources.enforce(ask.access());
        let input = Explained {
            explanation,
            sources,
        };
        with_access(self.verify(input).await, access)
    }

    async fn verify(&self, input: Explained) -> Result<Scored<Verified, VerifyRecord>, StepError> {
        let by_id: HashMap<&ChunkId, &Retrieved> = input
            .sources
            .0
            .iter()
            .map(|hit| (&hit.chunk.id, hit))
            .collect();
        let mut record = VerifyRecord {
            uncited_claims: Vec::new(),
            unknown_citations: Vec::new(),
            support: self.support(),
            claims: Vec::new(),
            stripped: 0,
        };
        let claims = input.explanation.claims();
        let mut cited = Vec::new();
        for claim in claims {
            let unknown: Vec<&ChunkId> = claim
                .chunks
                .iter()
                .filter(|id| !by_id.contains_key(id))
                .collect();
            for id in &unknown {
                if !record.unknown_citations.contains(id) {
                    record.unknown_citations.push((*id).clone());
                }
            }
            let sources: Vec<_> = claim
                .chunks
                .iter()
                .cloned()
                .map(Provenance::Chunk)
                .collect();
            match AtLeastOne::try_from(sources) {
                Ok(sources) if unknown.is_empty() => cited.push(Cited {
                    value: claim.text.clone(),
                    sources,
                }),
                Ok(_) => {}
                Err(_) => record.uncited_claims.push(claim.text.clone()),
            }
        }
        let model = self.judged_by();
        let decline = |reason: String, record: VerifyRecord| {
            let error = StepError::decline(DeclineReason::Unverified(reason)).with_record(record);
            match &model {
                Some(model) => error.with_model(model.clone(), SUPPORT_PROMPT_VERSION),
                None => error,
            }
        };
        if self.publish == PublishPolicy::Decline
            && (!record.uncited_claims.is_empty() || !record.unknown_citations.is_empty())
        {
            let reason = format!(
                "{} claim(s) without a citation, {} citation(s) outside this run's sources",
                record.uncited_claims.len(),
                record.unknown_citations.len(),
            );
            return Err(decline(reason, record));
        }

        let requests: Vec<_> = cited
            .iter()
            .map(|claim| self.check(claim, &by_id))
            .collect();
        let checks: Vec<Option<ClaimCheck>> = stream::iter(requests)
            .buffered(CONCURRENT_REQUESTS)
            .try_collect()
            .await
            .map_err(|error| match &model {
                Some(model) => error.with_model(model.clone(), SUPPORT_PROMPT_VERSION),
                None => error,
            })?;
        let passed: Vec<bool> = checks
            .iter()
            .map(|check| check.as_ref().is_none_or(|check| check.verdict.passed()))
            .collect();
        record.claims = checks.into_iter().flatten().collect();
        let failed = passed.iter().filter(|passed| !**passed).count();
        if self.publish == PublishPolicy::Decline && failed > 0 {
            let reason = format!("{failed} claim(s) not backed by the chunks they cite");
            return Err(decline(reason, record));
        }
        let kept: Vec<Cited<String>> = cited
            .into_iter()
            .zip(passed)
            .filter_map(|(claim, passed)| passed.then_some(claim))
            .collect();
        record.stripped = claims.len() - kept.len();
        if kept.is_empty() {
            return Err(decline(
                "no claim is backed by the chunks it cites".into(),
                record,
            ));
        }
        let scored = Scored::new(
            Verified {
                answer: VerifiedAnswer { claims: kept },
                sources: input.sources,
            },
            record,
        );
        Ok(match model {
            Some(model) => scored.with_model(model, SUPPORT_PROMPT_VERSION),
            None => scored,
        })
    }

    /// `None` under membership only, where a cited claim has no further check.
    async fn check(
        &self,
        claim: &Cited<String>,
        by_id: &HashMap<&ChunkId, &Retrieved>,
    ) -> Result<Option<ClaimCheck>, StepError> {
        let chunks: Vec<&Retrieved> = claim
            .sources
            .as_slice()
            .iter()
            .filter_map(|source| match source {
                Provenance::Chunk(id) => by_id.get(id).copied(),
                Provenance::UserPrompt(_) => None,
            })
            .collect();
        let verdict = match &self.support {
            Support::Membership => return Ok(None),
            Support::Quotes => quote_check(&claim.value, &chunks)
                .map_or(ClaimVerdict::QuotesFound, |span| ClaimVerdict::Fabricated {
                    span,
                }),
            Support::Judged { judge, min_support } => match quote_check(&claim.value, &chunks) {
                Some(span) => ClaimVerdict::Fabricated { span },
                None => support(judge.as_ref(), *min_support, &claim.value, &chunks).await?,
            },
        };
        Ok(Some(ClaimCheck {
            claim: claim.value.clone(),
            verdict,
        }))
    }
}

/// The first quoted span of `claim` that none of `chunks` contains; `None` when every span is found.
fn quote_check(claim: &str, chunks: &[&Retrieved]) -> Option<String> {
    let texts: Vec<String> = chunks
        .iter()
        .map(|hit| normalize(&hit.chunk.text))
        .collect();
    quoted_spans(claim)
        .into_iter()
        .find(|span| {
            let span = normalize(span);
            !span.is_empty() && !texts.iter().any(|text| text.contains(&span))
        })
        .map(str::to_owned)
}

/// Spans inside backticks, straight double quotes, or curly double quotes. Single quotes are left alone, since they
/// are also apostrophes.
fn quoted_spans(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some((start, open)) = rest
        .char_indices()
        .find(|(_, c)| matches!(c, '`' | '"' | '“'))
    {
        let close = if open == '“' { '”' } else { open };
        let after = &rest[start + open.len_utf8()..];
        let Some(end) = after.find(close) else {
            break;
        };
        let span = &after[..end];
        if !span.trim().is_empty() {
            spans.push(span);
        }
        rest = &after[end + close.len_utf8()..];
    }
    spans
}

/// Lowercase, without whitespace or quotation marks, so line breaks and markdown code spans do not hide a match.
fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '`' | '"' | '“' | '”'))
        .flat_map(char::to_lowercase)
        .collect()
}

/// Judges the cited chunks in citation order until one supports the claim with at least `min_support`.
async fn support(
    judge: &dyn Judge,
    min_support: Probability,
    claim: &str,
    chunks: &[&Retrieved],
) -> Result<ClaimVerdict, StepError> {
    let mut relations = Vec::new();
    for hit in chunks {
        let judgements = judge
            .judge(JudgeRequest {
                text: judged_text(&hit.chunk),
                questions: vec![JudgeQuestion::OneOf {
                    instructions: format!("How does this text relate to the claim: {claim}"),
                    options: RELATIONS.iter().map(|(_, option)| *option).collect(),
                }],
            })
            .await?;
        let (chosen, probability) = judgements.one_of(0).map_err(|error| {
            StepError::decline(DeclineReason::InvalidModelOutput(format!(
                "{}: {error}",
                hit.chunk.id.0
            )))
        })?;
        let relation = Relation {
            chunk: hit.chunk.id.clone(),
            relation: RELATIONS[chosen].0,
            probability,
        };
        if relation.relation == RelationKind::Supports && probability >= min_support {
            return Ok(ClaimVerdict::Supported { by: relation });
        }
        relations.push(relation);
    }
    Ok(
        match relations.iter().find(|relation| {
            relation.relation == RelationKind::Contradicts && relation.probability >= min_support
        }) {
            Some(by) => ClaimVerdict::Contradicted { by: by.clone() },
            None => ClaimVerdict::Unsupported { relations },
        },
    )
}

#[cfg(test)]
mod tests;
