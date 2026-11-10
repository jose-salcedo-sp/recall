use std::{collections::HashSet, marker::PhantomData};

use crate::{
    domain::{AtLeastOne, ChunkId, Cited, Probability, Provenance, Question},
    pipeline::{Boolean, DeclineReason, Open, Scored, Step, StepError, trace::VerifyRecord},
};

use super::{Explained, ExplainedVerdict, Reranked};

/// Rejects any claim without a citation and any citation outside this run's reranked sources.
pub struct Verify<TCategory>(PhantomData<TCategory>);

impl<TCategory> Verify<TCategory> {
    /// Verify has no settings or clients.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<TCategory> Default for Verify<TCategory> {
    fn default() -> Self {
        Self::new()
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

/// Every claim cites at least one chunk, and every cited chunk is among the run's reranked sources.
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
    /// System One's verdict.
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
        _question: &Question,
        input: Explained,
    ) -> Result<Scored<Verified, VerifyRecord>, StepError> {
        verify(input)
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
        _question: &Question,
        input: ExplainedVerdict,
    ) -> Result<Scored<VerifiedVerdict, VerifyRecord>, StepError> {
        let ExplainedVerdict {
            explained,
            yes,
            confidence,
        } = input;
        let scored = verify(explained)?;
        Ok(Scored::new(
            VerifiedVerdict {
                verified: scored.output,
                yes,
                confidence,
            },
            scored.record,
        ))
    }
}

fn verify(input: Explained) -> Result<Scored<Verified, VerifyRecord>, StepError> {
    let known: HashSet<&ChunkId> = input.sources.0.iter().map(|hit| &hit.chunk.id).collect();
    let mut claims = Vec::new();
    let mut record = VerifyRecord {
        uncited_claims: Vec::new(),
        unknown_citations: Vec::new(),
    };
    for claim in input.explanation.claims() {
        for id in &claim.chunks {
            if !known.contains(id) && !record.unknown_citations.contains(id) {
                record.unknown_citations.push(id.clone());
            }
        }
        let sources: Vec<_> = claim
            .chunks
            .iter()
            .cloned()
            .map(Provenance::Chunk)
            .collect();
        match AtLeastOne::try_from(sources) {
            Ok(sources) => claims.push(Cited {
                value: claim.text.clone(),
                sources,
            }),
            Err(_) => record.uncited_claims.push(claim.text.clone()),
        }
    }

    if !record.uncited_claims.is_empty() || !record.unknown_citations.is_empty() {
        let reason = DeclineReason::Unverified(format!(
            "{} claim(s) without a citation, {} citation(s) outside this run's sources",
            record.uncited_claims.len(),
            record.unknown_citations.len(),
        ));
        return Err(StepError::decline(reason).with_record(record));
    }
    Ok(Scored::new(
        Verified {
            answer: VerifiedAnswer { claims },
            sources: input.sources,
        },
        record,
    ))
}
