use std::sync::Arc;

use crate::{
    domain::{AtLeastOne, Question},
    pipeline::{
        Asked, DeclineReason, Scored, Step, StepError, Unclassified, trace::RetrieveRecord,
    },
    retrieval::{Retrieval, Retriever},
};

use super::{Candidates, ranked};

/// Runs the [`Retriever`] on the question, in parallel with `Classify`. Declines with
/// [`DeclineReason::NoSources`] when it returns nothing.
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
        question: &Question,
        _input: Asked,
    ) -> Result<Scored<Candidates, RetrieveRecord>, StepError> {
        let Retrieval { hits, expansion } =
            self.retriever.retrieve_expanded(question.as_str()).await?;
        let record = RetrieveRecord {
            chunks: ranked(&hits),
            expansion,
        };
        let Ok(hits) = AtLeastOne::try_from(hits) else {
            return Err(StepError::decline(DeclineReason::NoSources).with_record(record));
        };
        Ok(Scored::new(Candidates(hits), record))
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::{
        domain::{ChunkId, ChunkRef, ChunkVersion},
        pipeline::steps::fixtures::chunk,
        retrieval::{Relevance, Retrieved, glossary::Glossary},
    };

    struct Fixed(Vec<Retrieved>);

    #[async_trait]
    impl Retriever for Fixed {
        async fn retrieve(&self, _query: &str) -> eros::Result<Vec<Retrieved>> {
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
                score: Relevance(0.5),
                context_for: None,
            },
            Retrieved {
                chunk: next,
                score: Relevance(0.4),
                context_for: Some(matched.id.clone()),
            },
        ];
        let scored = Retrieve::new(Arc::new(Fixed(hits)))
            .run(&Question::new("q"), Asked)
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
        async fn retrieve(&self, query: &str) -> eros::Result<Vec<Retrieved>> {
            Ok(self.retrieve_expanded(query).await?.hits)
        }

        async fn retrieve_expanded(&self, query: &str) -> eros::Result<Retrieval> {
            Ok(Retrieval {
                hits: vec![Retrieved {
                    chunk: chunk("hit", "page", "the earlier screens"),
                    score: Relevance(0.5),
                    context_for: None,
                }],
                expansion: self.0.expand(query, 8),
            })
        }
    }

    #[tokio::test]
    async fn the_record_lists_the_expansion() {
        let source = ChunkRef {
            id: ChunkId("markdown:LEXICON.md#generations".into()),
            version: ChunkVersion("v1".into()),
        };
        let glossary = Glossary::parse([(
            source.clone(),
            "| Internal | Say instead |\n|---|---|\n| Legacy / NEW generation | the earlier screens |",
        )]);
        let scored = Retrieve::new(Arc::new(Expanding(glossary)))
            .run(&Question::new("Only the NEW screen?"), Asked)
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
