use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    sync::Arc,
};

use futures::{StreamExt, TryStreamExt, stream};

use crate::{
    config::{RerankConfig, RerankJudge, SystemOneJudge},
    domain::{ChunkId, Probability, Question},
    llm::{
        ModelRef,
        system_one::{Decision, DecisionRequest, SystemOne},
    },
    pipeline::{Category, DeclineReason, Scored, Step, StepError, trace::RerankRecord},
    retrieval::Retrieved,
};

use super::{Candidates, MAX_STATE_CHARS, Reranked, ranked};

/// Recorded as the trace entry's prompt version; bump on any change to the relevance question.
pub const RERANK_PROMPT_VERSION: &str = "rerank-v2";

const QUESTION_ID: &str = "relevant";
/// Requests in flight at once; the server batches them.
const CONCURRENT_REQUESTS: usize = 8;
/// Reciprocal rank fusion constant, as in hybrid retrieval.
const RRF_K: f32 = 60.0;

/// Picks the `top_k` matches that best answer the question, each followed by the context retrieval attached to it.
///
/// With System One as the judge, the best `candidates` matches by fused score are each judged by one yes/no
/// question. Matches are then ordered by reciprocal rank fusion of their retrieval rank and their probability-of-yes
/// rank, so the judgement refines retrieval instead of replacing it; the trace records each probability of yes, and
/// the best kept one scores the step. Without a judge, matches are ordered by retrieval rank alone and the step is
/// unscored. Matches always keep their fused score. Either way an evidence match is ranked
/// `evidence_rank_penalty` places lower, so Drew's settled answers lead unless the evidence is clearly better.
pub struct Rerank<TCategory> {
    top_k: usize,
    evidence_rank_penalty: usize,
    threshold: Option<Probability>,
    judge: Option<Judge>,
    category: PhantomData<TCategory>,
}

/// System One and the number of best matches by fused score it judges.
struct Judge {
    system_one: Arc<dyn SystemOne>,
    candidates: usize,
}

impl<TCategory> Rerank<TCategory> {
    /// The default evidence penalty and no threshold. `candidates` applies only with System One.
    pub fn new(top_k: usize, candidates: usize, system_one: Option<Arc<dyn SystemOne>>) -> Self {
        Self {
            top_k,
            evidence_rank_penalty: RerankConfig::default().evidence_rank_penalty(),
            threshold: None,
            judge: system_one.map(|system_one| Judge {
                system_one,
                candidates,
            }),
            category: PhantomData,
        }
    }

    /// Every setting from the config, plus the step's threshold for its category. `system_one` judges only when
    /// the config's judge is [`RerankJudge::SystemOne`].
    pub fn from_config(
        config: &RerankConfig,
        threshold: Option<Probability>,
        system_one: Option<Arc<dyn SystemOne>>,
    ) -> Self {
        let judge = match config.judge() {
            RerankJudge::None => None,
            RerankJudge::SystemOne(SystemOneJudge { candidates }) => {
                system_one.map(|system_one| Judge {
                    system_one,
                    candidates,
                })
            }
        };
        Self {
            top_k: config.top_k(),
            evidence_rank_penalty: config.evidence_rank_penalty(),
            threshold,
            judge,
            category: PhantomData,
        }
    }

    /// Matches kept.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Matches System One judges; `None` when no model judges.
    pub fn candidates(&self) -> Option<usize> {
        self.judge.as_ref().map(|judge| judge.candidates)
    }

    /// The model that judges the matches; `None` when they keep retrieval order.
    pub fn judged_by(&self) -> Option<ModelRef> {
        self.judge.as_ref().map(|judge| judge.system_one.model())
    }
}

impl<TCategory> Default for Rerank<TCategory> {
    fn default() -> Self {
        Self::from_config(&RerankConfig::default(), None, None)
    }
}

fn state(hit: &Retrieved) -> String {
    let chunk = &hit.chunk;
    let heading = chunk
        .heading
        .as_deref()
        .map(|heading| format!(" > {heading}"))
        .unwrap_or_default();
    let text = match chunk.text.char_indices().nth(MAX_STATE_CHARS) {
        Some((end, _)) => &chunk.text[..end],
        None => &chunk.text,
    };
    format!("{}{heading}\n\n{text}", chunk.title)
}

fn request(question: &Question, hit: &Retrieved) -> DecisionRequest {
    DecisionRequest {
        state: state(hit),
        model: String::new(),
        questions: BTreeMap::from([(
            QUESTION_ID.to_owned(),
            Decision::Noul {
                instructions: format!(
                    "Does this text contain information that helps answer the question: {}",
                    question.as_str()
                ),
            },
        )]),
    }
}

/// One list's reciprocal rank fusion term for a match at a 0-based `rank`.
fn rrf(hit: &Retrieved, rank: usize, evidence_rank_penalty: usize) -> f32 {
    let penalty = if hit.chunk.authoritative {
        0
    } else {
        evidence_rank_penalty
    };
    1.0 / (RRF_K + (rank + penalty) as f32)
}

async fn judge(
    system_one: &dyn SystemOne,
    request: DecisionRequest,
    chunk: &ChunkId,
) -> Result<Probability, StepError> {
    let response = system_one.decide(request).await?;
    response.yes_no(QUESTION_ID).map_err(|error| {
        StepError::decline(DeclineReason::InvalidModelOutput(format!(
            "{}: {error}",
            chunk.0
        )))
    })
}

impl<TCategory: Category> Step for Rerank<TCategory> {
    const NAME: &'static str = "rerank";
    type Category = TCategory;
    type Input = Candidates;
    type Output = Reranked;
    type Record = RerankRecord;

    /// Without System One the step is unscored, so no threshold applies.
    fn threshold(&self) -> Option<Probability> {
        self.threshold.filter(|_| self.judge.is_some())
    }

    async fn run(
        &self,
        question: &Question,
        input: Candidates,
    ) -> Result<Scored<Reranked, RerankRecord>, StepError> {
        let (mut matches, context): (Vec<Retrieved>, Vec<Retrieved>) = input
            .0
            .into_vec()
            .into_iter()
            .partition(|hit| hit.context_for.is_none());
        matches.sort_by(|a, b| b.score.0.total_cmp(&a.score.0));

        let penalty = self.evidence_rank_penalty;
        let mut raw = None;
        let mut model = None;
        let mut judged: HashMap<ChunkId, Probability> = HashMap::new();
        if let Some(Judge {
            system_one,
            candidates,
        }) = &self.judge
        {
            matches.truncate(*candidates);
            let judged_by = model.insert(system_one.model()).clone();
            let requests: Vec<_> = matches
                .iter()
                .map(|hit| {
                    let (system_one, request) = (Arc::clone(system_one), request(question, hit));
                    let id = hit.chunk.id.clone();
                    async move { judge(system_one.as_ref(), request, &id).await }
                })
                .collect();
            let probabilities: Vec<Probability> = stream::iter(requests)
                .buffered(CONCURRENT_REQUESTS)
                .try_collect()
                .await
                .map_err(|error| error.with_model(judged_by, RERANK_PROMPT_VERSION))?;
            let by_chunk: BTreeMap<&str, Probability> = matches
                .iter()
                .zip(&probabilities)
                .map(|(hit, p)| (hit.chunk.id.0.as_str(), *p))
                .collect();
            raw = serde_json::to_string(&by_chunk).ok();
            let mut by_probability: Vec<usize> = (0..matches.len()).collect();
            by_probability
                .sort_by(|&a, &b| probabilities[b].get().total_cmp(&probabilities[a].get()));
            let mut judged_rank = vec![0; matches.len()];
            for (rank, &index) in by_probability.iter().enumerate() {
                judged_rank[index] = rank;
            }
            let mut fused: Vec<(f32, Retrieved)> = matches
                .drain(..)
                .zip(probabilities)
                .enumerate()
                .map(|(retrieval_rank, (hit, probability))| {
                    let key = rrf(&hit, retrieval_rank, penalty)
                        + rrf(&hit, judged_rank[retrieval_rank], penalty);
                    judged.insert(hit.chunk.id.clone(), probability);
                    (key, hit)
                })
                .collect();
            fused.sort_by(|a, b| b.0.total_cmp(&a.0));
            matches = fused.into_iter().map(|(_, hit)| hit).collect();
        } else {
            let mut ordered: Vec<(f32, Retrieved)> = matches
                .drain(..)
                .enumerate()
                .map(|(rank, hit)| (rrf(&hit, rank, penalty), hit))
                .collect();
            ordered.sort_by(|a, b| b.0.total_cmp(&a.0));
            matches = ordered.into_iter().map(|(_, hit)| hit).collect();
        }
        matches.truncate(self.top_k);
        let best = matches
            .iter()
            .filter_map(|hit| judged.get(&hit.chunk.id).copied())
            .max_by(|a, b| a.get().total_cmp(&b.get()));

        let mut context_of: HashMap<ChunkId, Vec<Retrieved>> = HashMap::new();
        for chunk in context {
            if let Some(hit) = chunk.context_for.clone() {
                context_of.entry(hit).or_default().push(chunk);
            }
        }
        let mut sources = Vec::new();
        let mut kept_context = Vec::new();
        for hit in &matches {
            let around = context_of.remove(&hit.chunk.id).unwrap_or_default();
            sources.push(hit.clone());
            sources.extend(around.iter().cloned());
            kept_context.extend(around);
        }
        // Matches first, so eval's hit@5 reads the top five matches.
        let mut chunks = ranked(&matches);
        for chunk in &mut chunks {
            chunk.probability = judged.get(&chunk.chunk.id).copied();
        }
        chunks.extend(ranked(&kept_context));
        let record = RerankRecord { raw, chunks };
        let scored = Scored::new(Reranked(sources), record);
        Ok(match (model, best) {
            (Some(model), Some(best)) => scored
                .with_score(best)
                .with_model(model, RERANK_PROMPT_VERSION),
            _ => scored,
        })
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::{
        domain::Chunk,
        llm::{
            ModelRef,
            fakes::ScriptedSystemOne,
            system_one::{DecisionAnswer, DecisionResponse, SystemOne},
        },
        pipeline::{Open, StepErrorKind, steps::fixtures::chunk},
        retrieval::Relevance,
    };

    fn hit(locator: &str, text: &str, score: f32) -> Retrieved {
        Retrieved {
            chunk: chunk(locator, "Roles", text),
            score: Relevance(score),
            context_for: None,
        }
    }

    fn context(of: &Retrieved, locator: &str, score: f32) -> Retrieved {
        Retrieved {
            chunk: chunk(locator, "Roles", "context"),
            score: Relevance(score),
            context_for: Some(of.chunk.id.clone()),
        }
    }

    fn ids(reranked: &Reranked) -> Vec<&str> {
        reranked
            .0
            .iter()
            .map(|hit| hit.chunk.heading.as_deref().unwrap())
            .collect()
    }

    /// Says yes with the probability written in the state's text, e.g. `p=0.9`.
    struct ByText;

    #[async_trait]
    impl SystemOne for ByText {
        fn model(&self) -> ModelRef {
            ModelRef::new("fake-system-one", "1")
        }

        async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
            let p = request
                .state
                .split("p=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|p| p.parse().ok())
                .unwrap_or(0.0);
            Ok(DecisionResponse {
                answers: BTreeMap::from([(QUESTION_ID.into(), DecisionAnswer::Noul { noul: p })]),
            })
        }
    }

    async fn rerank(rerank: Rerank<Open>, hits: Vec<Retrieved>) -> Scored<Reranked, RerankRecord> {
        rerank
            .run(&Question::new("q"), candidates(hits))
            .await
            .unwrap()
    }

    fn candidates(hits: Vec<Retrieved>) -> Candidates {
        Candidates(crate::domain::AtLeastOne::try_from(hits).unwrap())
    }

    #[tokio::test]
    async fn context_rides_with_its_match_instead_of_taking_a_slot() {
        let first = hit("a", "", 0.9);
        let second = hit("b", "", 0.8);
        let hits = vec![
            context(&first, "a-next", 0.89),
            context(&first, "a-prev", 0.89),
            first.clone(),
            second.clone(),
            hit("c", "", 0.1),
        ];

        let scored = rerank(Rerank::new(2, 25, None), hits).await;

        assert_eq!(ids(&scored.output), ["a", "a-next", "a-prev", "b"]);
        assert_eq!(scored.score, None);
        let recorded: Vec<_> = scored
            .record
            .chunks
            .iter()
            .map(|c| c.chunk.id.0.as_str())
            .collect();
        assert_eq!(
            recorded[..2],
            [first.chunk.id.0.as_str(), second.chunk.id.0.as_str()]
        );
        assert_eq!(recorded.len(), 4);
    }

    fn authoritative(hit: Retrieved) -> Retrieved {
        Retrieved {
            chunk: Chunk {
                authoritative: true,
                ..hit.chunk
            },
            ..hit
        }
    }

    #[tokio::test]
    async fn system_one_refines_retrieval_order_and_scores_the_step() {
        let hits = vec![
            hit("fused-best", "p=0.2", 0.9),
            hit("fused-second", "p=0.95", 0.8),
            hit("fused-third", "p=0.6", 0.7),
            hit("beyond-candidates", "p=1.0", 0.1),
        ];

        let scored = rerank(Rerank::new(2, 3, Some(Arc::new(ByText))), hits).await;

        // Judged ranks second, third, best; fused with retrieval ranks best, second, third.
        assert_eq!(ids(&scored.output), ["fused-second", "fused-best"]);
        assert_eq!(scored.score, Some(Probability::new(0.95).unwrap()));
        assert_eq!(scored.prompt_version, Some(RERANK_PROMPT_VERSION));
        let raw: BTreeMap<String, f32> =
            serde_json::from_str(scored.record.raw.as_deref().unwrap()).unwrap();
        assert_eq!(raw.len(), 3);
        assert_eq!(scored.output.0[0].score, Relevance(0.8));
        let judged: Vec<_> = scored
            .record
            .chunks
            .iter()
            .map(|c| (c.score, c.probability.map(Probability::get)))
            .collect();
        assert_eq!(
            judged,
            [(Relevance(0.8), Some(0.95)), (Relevance(0.9), Some(0.2))]
        );
    }

    #[tokio::test]
    async fn a_probability_outside_zero_to_one_declines_as_invalid_output() {
        for bad in ["p=1.7", "p=-0.3", "p=NaN"] {
            let error = Rerank::<Open>::new(1, 25, Some(Arc::new(ByText)))
                .run(&Question::new("q"), candidates(vec![hit("a", bad, 0.9)]))
                .await
                .err()
                .unwrap();
            assert!(
                matches!(
                    error.kind,
                    StepErrorKind::Decline(DeclineReason::InvalidModelOutput(_))
                ),
                "{bad}"
            );
        }
    }

    #[tokio::test]
    async fn authoritative_matches_lead_unless_evidence_is_clearly_better() {
        let with_system_one = || Some(Arc::new(ByText) as Arc<dyn SystemOne>);
        let close = vec![
            hit("evidence", "p=0.6", 0.9),
            authoritative(hit("settled", "p=0.5", 0.8)),
        ];
        let scored = rerank(Rerank::new(1, 25, with_system_one()), close.clone()).await;
        assert_eq!(ids(&scored.output), ["settled"]);
        let scored = rerank(Rerank::new(1, 25, None), close).await;
        assert_eq!(ids(&scored.output), ["settled"]);

        let mut far = vec![hit("evidence", "p=0.99", 0.9)];
        far.extend((0..40).map(|i| hit(&format!("filler-{i}"), "p=0.0", 0.5)));
        far.push(authoritative(hit("settled", "p=0.01", 0.1)));
        let scored = rerank(Rerank::new(1, 50, with_system_one()), far).await;
        assert_eq!(ids(&scored.output), ["evidence"]);
    }

    fn config(json: &str) -> RerankConfig {
        serde_json::from_str(json).unwrap()
    }

    fn scripted(probabilities: &[f32]) -> Arc<ScriptedSystemOne> {
        Arc::new(ScriptedSystemOne::new(probabilities.iter().map(|&noul| {
            Ok(DecisionResponse {
                answers: BTreeMap::from([(QUESTION_ID.into(), DecisionAnswer::Noul { noul })]),
            })
        })))
    }

    fn many() -> Vec<Retrieved> {
        vec![
            hit("fused-best", "", 0.9),
            hit("fused-second", "", 0.8),
            authoritative(hit("fused-third", "", 0.7)),
            hit("fused-fourth", "", 0.6),
        ]
    }

    #[tokio::test]
    async fn without_a_judge_system_one_is_never_called() {
        let system_one = scripted(&[]);
        let step = Rerank::<Open>::from_config(
            &config(r#"{"top_k": 2, "judge": "none"}"#),
            Some(Probability::new(0.9).unwrap()),
            Some(Arc::<ScriptedSystemOne>::clone(&system_one) as Arc<dyn SystemOne>),
        );
        assert_eq!(step.threshold(), None);
        assert_eq!(step.judged_by(), None);
        assert_eq!(step.candidates(), None);

        let scored = rerank(step, many()).await;

        assert!(system_one.requests().is_empty());
        // Retrieval order, the evidence penalty lifting the authoritative third match to the top.
        assert_eq!(ids(&scored.output), ["fused-third", "fused-best"]);
        assert_eq!(scored.score, None);
        assert_eq!(scored.model, None);
        assert_eq!(scored.record.raw, None);
        assert!(scored.record.chunks.iter().all(|c| c.probability.is_none()));
    }

    #[tokio::test]
    async fn a_system_one_judge_works_as_before() {
        let system_one = scripted(&[0.2, 0.95, 0.6]);
        let threshold = Some(Probability::new(0.5).unwrap());
        let step = Rerank::<Open>::from_config(
            &config(r#"{"top_k": 2, "judge": {"system_one": {"candidates": 3}}}"#),
            threshold,
            Some(Arc::<ScriptedSystemOne>::clone(&system_one) as Arc<dyn SystemOne>),
        );
        assert_eq!(step.threshold(), threshold);
        assert_eq!(step.candidates(), Some(3));
        assert_eq!(
            step.judged_by(),
            Some(ModelRef::new("fake-system-one", "1"))
        );

        let scored = rerank(step, many()).await;

        assert_eq!(system_one.requests().len(), 3);
        let same = rerank(
            Rerank::new(2, 3, Some(Arc::new(ByText))),
            vec![
                hit("fused-best", "p=0.2", 0.9),
                hit("fused-second", "p=0.95", 0.8),
                authoritative(hit("fused-third", "p=0.6", 0.7)),
                hit("fused-fourth", "p=1.0", 0.6),
            ],
        )
        .await;
        assert_eq!(ids(&scored.output), ids(&same.output));
        assert_eq!(scored.score, Some(Probability::new(0.95).unwrap()));
        assert_eq!(scored.score, same.score);
        assert_eq!(scored.prompt_version, Some(RERANK_PROMPT_VERSION));
    }

    #[tokio::test]
    async fn a_system_one_judge_without_system_one_keeps_retrieval_order() {
        let step = Rerank::<Open>::from_config(
            &config(r#"{"top_k": 2, "judge": {"system_one": {"candidates": 3}}}"#),
            Some(Probability::new(0.5).unwrap()),
            None,
        );
        assert_eq!(step.threshold(), None);
        let scored = rerank(step, many()).await;
        assert_eq!(ids(&scored.output), ["fused-third", "fused-best"]);
    }

    #[test]
    fn a_long_section_is_judged_on_its_beginning() {
        let long = Retrieved {
            chunk: Chunk {
                text: "x".repeat(MAX_STATE_CHARS * 2),
                ..chunk("long", "Roles", "")
            },
            score: Relevance(1.0),
            context_for: None,
        };
        let state = state(&long);
        assert!(state.starts_with("Roles > long\n\n"));
        assert_eq!(state.chars().filter(|c| *c == 'x').count(), MAX_STATE_CHARS);
    }
}
