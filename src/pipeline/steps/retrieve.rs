use std::sync::Arc;

use crate::{
    domain::AtLeastOne,
    pipeline::{
        Ask, Asked, DeclineReason, Scored, Step, StepError, Unclassified, trace::RetrieveRecord,
    },
    retrieval::{Retrieval, Retriever},
};

use super::{Candidates, enforce, ranked};

/// Runs the [`Retriever`] on the question, in parallel with `Classify`, within the run's chunk scope, and rechecks
/// every hit against it. Declines with [`DeclineReason::NoSources`] when nothing readable is found.
pub struct Retrieve {
    retriever: Arc<dyn Retriever>,
}

impl Retrieve {
    /// Wraps the retriever.
    pub fn new(retriever: Arc<dyn Retriever>) -> Self {
        Self { retriever }
    }
}

impl Step for Retrieve {
    const NAME: &'static str = "retrieve";
    type Category = Unclassified;
    type Input = Asked;
    type Output = Candidates;
    type Record = RetrieveRecord;

    async fn run(
        &self,
        ask: &Ask,
        _input: Asked,
    ) -> Result<Scored<Candidates, RetrieveRecord>, StepError> {
        let Retrieval { hits, expansion } = self
            .retriever
            .retrieve_expanded(ask.question().as_str(), ask.access().chunks())
            .await?;
        let (hits, access) = enforce(ask.access(), hits);
        let denied = access.len() - hits.len();
        if denied > 0 {
            tracing::warn!(
                denied,
                "the retriever returned chunks outside the run's scope; they were dropped"
            );
        }
        let record = RetrieveRecord {
            chunks: ranked(&hits),
            expansion,
        };
        let Ok(hits) = AtLeastOne::try_from(hits) else {
            return Err(StepError::decline(DeclineReason::NoSources)
                .with_record(record)
                .with_access(access));
        };
        Ok(Scored::new(Candidates(hits), record).with_access(access))
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::{
        access::{
            Access, ChunkScope, Decision, DenyReason,
            fakes::{tester, unrestricted},
        },
        domain::{ChunkId, ChunkRef, ChunkVersion, Question, Sensitivity},
        pipeline::{StepErrorKind, steps::fixtures::chunk},
        retrieval::{
            Relevance, Retrieved,
            fakes::{FailingRetriever, LeakyRetriever},
            glossary::Glossary,
        },
    };

    fn confidential(locator: &str) -> crate::domain::Chunk {
        let mut chunk = chunk(locator, "secrets", "the confidential plan");
        chunk.labels.sensitivity = Sensitivity::Confidential;
        chunk
    }

    /// Reads internal chunks only.
    fn reader() -> Access {
        let role = crate::access::Role::new("reader").unwrap();
        Access::new(
            ChunkScope::new(
                tester().id().clone(),
                [],
                [crate::access::ChunkGrant::new(role, Sensitivity::Internal)],
            ),
            Default::default(),
        )
    }

    #[tokio::test]
    async fn chunks_a_retriever_leaks_are_dropped_and_recorded() {
        let open = chunk("open", "page", "the plan");
        let retriever = LeakyRetriever {
            chunks: vec![confidential("secret"), open.clone()],
        };
        let ask = Ask::new(Question::new("the plan?"), tester(), reader());

        let scored = Retrieve::new(Arc::new(retriever))
            .run(&ask, Asked)
            .await
            .unwrap();

        let kept: Vec<_> = scored
            .output
            .0
            .as_slice()
            .iter()
            .map(|hit| &hit.chunk.id)
            .collect();
        assert_eq!(kept, [&open.id]);
        assert_eq!(scored.record.chunks.len(), 1);
        let decisions: Vec<_> = scored.access.iter().map(|check| &check.decision).collect();
        assert_eq!(
            decisions,
            [
                &Decision::Denied {
                    reason: DenyReason::NoGrant
                },
                &Decision::Allowed {
                    role: crate::access::Role::new("reader").unwrap()
                },
            ]
        );
    }

    #[tokio::test]
    async fn nothing_readable_declines_with_the_denials_recorded() {
        let retriever = LeakyRetriever {
            chunks: vec![confidential("secret")],
        };
        let ask = Ask::new(Question::new("the plan?"), tester(), Access::none());

        let error = Retrieve::new(Arc::new(retriever))
            .run(&ask, Asked)
            .await
            .err()
            .unwrap();

        assert!(matches!(
            error.kind,
            StepErrorKind::Decline(DeclineReason::NoSources)
        ));
        assert_eq!(error.access.len(), 1);
        assert!(!error.access[0].decision.is_allowed());
    }

    #[tokio::test]
    async fn a_failing_retriever_fails_the_step() {
        let ask = Ask::new(Question::new("q"), tester(), unrestricted());
        let error = Retrieve::new(Arc::new(FailingRetriever))
            .run(&ask, Asked)
            .await
            .err()
            .unwrap();
        assert!(matches!(error.kind, StepErrorKind::Failed(_)));
    }

    struct Fixed(Vec<Retrieved>);

    #[async_trait]
    impl Retriever for Fixed {
        async fn retrieve(
            &self,
            _query: &str,
            _scope: &ChunkScope,
        ) -> eros::Result<Vec<Retrieved>> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn the_record_marks_context_chunks() {
        let matched = chunk("match", "page", "the match");
        let next = chunk("next", "page", "the section after it");
        let hits = vec![
            Retrieved {
                chunk: matched.clone(),
                score: Relevance::new(0.5).unwrap(),
                context_for: None,
            },
            Retrieved {
                chunk: next,
                score: Relevance::new(0.4).unwrap(),
                context_for: Some(matched.id.clone()),
            },
        ];
        let scored = Retrieve::new(Arc::new(Fixed(hits)))
            .run(&Ask::unrestricted("q"), Asked)
            .await
            .unwrap();
        let context: Vec<_> = scored
            .record
            .chunks
            .iter()
            .map(|ranked| ranked.context_for.as_ref())
            .collect();
        assert_eq!(context, [None, Some(&matched.id)]);
    }

    struct Expanding(Glossary);

    #[async_trait]
    impl Retriever for Expanding {
        async fn retrieve(&self, query: &str, scope: &ChunkScope) -> eros::Result<Vec<Retrieved>> {
            Ok(self.retrieve_expanded(query, scope).await?.hits)
        }

        async fn retrieve_expanded(
            &self,
            query: &str,
            _scope: &ChunkScope,
        ) -> eros::Result<Retrieval> {
            Ok(Retrieval {
                hits: vec![Retrieved {
                    chunk: chunk("hit", "page", "the earlier screens"),
                    score: Relevance::new(0.5).unwrap(),
                    context_for: None,
                }],
                expansion: self.0.expand(query, 8),
            })
        }
    }

    #[tokio::test]
    async fn the_record_lists_the_expansion() {
        let source = ChunkRef {
            id: ChunkId::try_from(String::from("markdown:LEXICON.md#generations")).unwrap(),
            version: ChunkVersion::try_from(String::from("v1")).unwrap(),
        };
        let glossary = Glossary::parse([(
            source.clone(),
            "| Internal | Say instead |\n|---|---|\n| Legacy / NEW generation | the earlier screens |",
        )]);
        let scored = Retrieve::new(Arc::new(Expanding(glossary)))
            .run(&Ask::unrestricted("Only the NEW screen?"), Asked)
            .await
            .unwrap();

        let matches = scored.record.expansion.matches();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched, "new");
        assert_eq!(matches[0].source, source);
        assert_eq!(
            matches[0].terms,
            ["legacy", "generation", "earlier", "screens"]
        );
        let json = serde_json::to_value(&scored.record).unwrap();
        assert_eq!(json["expansion"][0]["matched"], "new");
    }
}
