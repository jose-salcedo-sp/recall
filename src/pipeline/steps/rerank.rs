use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    marker::PhantomData,
    sync::Arc,
};

use futures::{StreamExt, TryStreamExt, stream};

use crate::{
    config::{RerankConfig, RerankJudge},
    domain::Standing,
    domain::{AtLeastOne, ChunkId, Probability, Question},
    llm::{
        ModelRef,
        judge::{Judge, JudgeQuestion, JudgeRequest},
    },
    pipeline::{
        Ask, Category, DeclineReason, Scored, Step, StepError,
        trace::{RerankMode, RerankOrder, RerankRecord},
    },
    retrieval::{Relevance, Retrieved},
};

use super::{CONCURRENT_REQUESTS, Candidates, Reranked, enforce, judged_text, ranked, with_access};

/// Recorded as the trace entry's prompt version; bump on any change to the relevance question.
pub const RERANK_PROMPT_VERSION: &str = "rerank-v2";

/// Reciprocal rank fusion constant, as in hybrid retrieval.
const RRF_K: f32 = 60.0;

/// Picks the `top_k` matches that best answer the question, each followed by the context retrieval attached to it.
///
/// With a judge, the best `candidates` matches by fused score are each judged by one yes/no
/// question. Matches are then ordered by reciprocal rank fusion of their retrieval rank and their probability-of-yes
/// rank, so the judgement refines retrieval instead of replacing it; the trace records each probability of yes, and
/// the best kept one scores the step. Without a judge, matches are ordered by retrieval rank alone and the step is
/// unscored. Matches always keep their fused score. Either way an evidence match is ranked
/// `evidence_rank_penalty` places lower, so settled answers lead unless the evidence is clearly better.
pub struct Rerank<TCategory> {
    top_k: usize,
    /// At least `top_k`: the config's judge candidates, or `top_k` when the config names no judge.
    candidates: usize,
    evidence_rank_penalty: usize,
    by: RerankBy,
    category: PhantomData<TCategory>,
}

/// What orders the matches `Rerank` keeps: `rerank.judge` resolved against the judge the pipeline has.
pub enum RerankBy {
    /// Retrieval rank alone. The step is unscored, so it has no threshold.
    Retrieval,
    /// The judge asks about the best matches by fused score: as many as the config's `rerank.judge` candidates.
    Judge {
        /// The judge.
        judge: Arc<dyn Judge>,
        /// Applies to the best kept match's probability.
        threshold: Option<Probability>,
    },
}

impl<TCategory> fmt::Debug for Rerank<TCategory> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rerank")
            .field("top_k", &self.top_k)
            .field("candidates", &self.candidates)
            .field("evidence_rank_penalty", &self.evidence_rank_penalty)
            .field("by", &self.by)
            .finish()
    }
}

impl fmt::Debug for RerankBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retrieval => f.write_str("Retrieval"),
            Self::Judge { judge, threshold } => f
                .debug_struct("Judge")
                .field("judge", &judge.model())
                .field("threshold", threshold)
                .finish(),
        }
    }
}

impl<TCategory> Rerank<TCategory> {
    /// `config`'s `top_k` and evidence penalty, ordered `by` what the pipeline's assembly resolved its judge to.
    pub fn new(config: &RerankConfig, by: RerankBy) -> Self {
        let candidates = match config.judge() {
            RerankJudge::Judged(candidates) => candidates.get(),
            RerankJudge::None => config.top_k(),
        };
        Self {
            top_k: config.top_k(),
            candidates,
            evidence_rank_penalty: config.evidence_rank_penalty(),
            by,
            category: PhantomData,
        }
    }

    /// How the matches are ordered.
    pub fn mode(&self) -> RerankMode {
        match self.by {
            RerankBy::Retrieval => RerankMode::RetrievalOrder,
            RerankBy::Judge { .. } => RerankMode::Judged,
        }
    }

    /// The model that judges the matches; `None` when they keep retrieval order.
    pub fn judged_by(&self) -> Option<ModelRef> {
        match &self.by {
            RerankBy::Retrieval => None,
            RerankBy::Judge { judge, .. } => Some(judge.model()),
        }
    }
}

fn request(question: &Question, hit: &Retrieved) -> JudgeRequest {
    JudgeRequest {
        text: judged_text(&hit.chunk),
        questions: vec![JudgeQuestion::YesNo {
            instructions: format!(
                "Does this text contain information that helps answer the question: {}",
                question.as_str()
            ),
        }],
    }
}

/// One list's reciprocal rank fusion term for a match at a 0-based `rank`.
fn rrf(hit: &Retrieved, rank: usize, evidence_rank_penalty: usize) -> f32 {
    let penalty = match hit.chunk.standing {
        Standing::Authoritative => 0,
        Standing::Evidence => evidence_rank_penalty,
    };
    1.0 / (RRF_K + (rank + penalty) as f32)
}

async fn relevance(
    judge: &dyn Judge,
    request: JudgeRequest,
    chunk: &ChunkId,
) -> Result<Probability, StepError> {
    let judgements = judge.judge(request).await?;
    judgements.yes(0).map_err(|error| {
        StepError::decline(DeclineReason::InvalidModelOutput(format!(
            "{}: {error}",
            chunk.as_str()
        )))
    })
}

impl<TCategory: Category> Step for Rerank<TCategory> {
    const NAME: &'static str = "rerank";
    type Category = TCategory;
    type Input = Candidates;
    type Output = Reranked;
    type Record = RerankRecord;

    fn threshold(&self) -> Option<Probability> {
        match self.by {
            RerankBy::Retrieval => None,
            RerankBy::Judge { threshold, .. } => threshold,
        }
    }

    async fn run(
        &self,
        ask: &Ask,
        input: Candidates,
    ) -> Result<Scored<Reranked, RerankRecord>, StepError> {
        let (hits, access) = enforce(ask.access(), input.0.into_vec());
        let result = match AtLeastOne::try_from(hits) {
            Ok(hits) => self.rerank(ask.question(), Candidates(hits)).await,
            Err(_) => Err(StepError::decline(DeclineReason::NoSources)),
        };
        with_access(result, access)
    }
}

impl<TCategory> Rerank<TCategory> {
    async fn rerank(
        &self,
        question: &Question,
        input: Candidates,
    ) -> Result<Scored<Reranked, RerankRecord>, StepError> {
        let (mut matches, context): (Vec<Retrieved>, Vec<Retrieved>) = input
            .0
            .into_vec()
            .into_iter()
            .partition(|hit| hit.context_for.is_none());
        matches.sort_by(|a, b| Relevance::descending(&a.score, &b.score));

        let penalty = self.evidence_rank_penalty;
        let mut raw = None;
        let mut model = None;
        let mut judged: HashMap<ChunkId, Probability> = HashMap::new();
        if let RerankBy::Judge { judge, .. } = &self.by {
            matches.truncate(self.candidates);
            let judged_by = model.insert(judge.model()).clone();
            let requests: Vec<_> = matches
                .iter()
                .map(|hit| {
                    let (judge, request) = (Arc::clone(judge), request(question, hit));
                    let id = hit.chunk.id.clone();
                    async move { relevance(judge.as_ref(), request, &id).await }
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
                .map(|(hit, p)| (hit.chunk.id.as_str(), *p))
                .collect();
            raw = Some(
                serde_json::to_string(&by_chunk)
                    .expect("probabilities by chunk id are plain data, which always serializes"),
            );
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
        let record = RerankRecord {
            order: match raw {
                Some(raw) => RerankOrder::Judged { raw },
                None => RerankOrder::RetrievalOrder,
            },
            chunks,
        };
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
    use super::*;
    use crate::{
        domain::Chunk,
        llm::{
            ModelRef,
            judge::{FnJudge, RawJudgement},
        },
        pipeline::steps::MAX_STATE_CHARS,
        pipeline::{Open, StepErrorKind, steps::fixtures::chunk},
    };

    fn keeping(top_k: usize) -> RerankConfig {
        RerankConfig::new(
            top_k,
            crate::config::RerankJudge::None,
            RerankConfig::default().evidence_rank_penalty(),
        )
        .unwrap()
    }

    fn by_retrieval(top_k: usize) -> Rerank<Open> {
        Rerank::new(&keeping(top_k), RerankBy::Retrieval)
    }

    fn judging(top_k: usize, candidates: usize) -> RerankConfig {
        RerankConfig::new(
            top_k,
            RerankJudge::Judged(crate::config::RerankCandidates::new(candidates)),
            RerankConfig::default().evidence_rank_penalty(),
        )
        .unwrap()
    }

    fn judged(top_k: usize, candidates: usize, judge: Arc<dyn Judge>) -> Rerank<Open> {
        Rerank::new(
            &judging(top_k, candidates),
            RerankBy::Judge {
                judge,
                threshold: None,
            },
        )
    }

    fn hit(locator: &str, text: &str, score: f32) -> Retrieved {
        Retrieved {
            chunk: chunk(locator, "Roles", text),
            score: Relevance::new(score).unwrap(),
            context_for: None,
        }
    }

    fn context(of: &Retrieved, locator: &str, score: f32) -> Retrieved {
        Retrieved {
            chunk: chunk(locator, "Roles", "context"),
            score: Relevance::new(score).unwrap(),
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

    /// Says yes with the probability written in the text, e.g. `p=0.9`.
    fn by_text() -> Arc<FnJudge> {
        Arc::new(FnJudge::new(|text, _| {
            RawJudgement::YesNo(
                text.split("p=")
                    .nth(1)
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(0.0),
            )
        }))
    }

    async fn rerank(rerank: Rerank<Open>, hits: Vec<Retrieved>) -> Scored<Reranked, RerankRecord> {
        rerank
            .run(&Ask::unrestricted("q"), candidates(hits))
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

        let scored = rerank(by_retrieval(2), hits).await;

        assert_eq!(ids(&scored.output), ["a", "a-next", "a-prev", "b"]);
        assert_eq!(scored.score, None);
        let recorded: Vec<_> = scored
            .record
            .chunks
            .iter()
            .map(|c| c.chunk.id.as_str())
            .collect();
        assert_eq!(
            recorded[..2],
            [first.chunk.id.as_str(), second.chunk.id.as_str()]
        );
        assert_eq!(recorded.len(), 4);
    }

    fn authoritative(hit: Retrieved) -> Retrieved {
        Retrieved {
            chunk: Chunk {
                standing: crate::domain::Standing::Authoritative,
                ..hit.chunk
            },
            ..hit
        }
    }

    #[tokio::test]
    async fn a_judge_refines_retrieval_order_and_scores_the_step() {
        let hits = vec![
            hit("fused-best", "p=0.2", 0.9),
            hit("fused-second", "p=0.95", 0.8),
            hit("fused-third", "p=0.6", 0.7),
            hit("beyond-candidates", "p=1.0", 0.1),
        ];

        let scored = rerank(judged(2, 3, by_text()), hits).await;

        // Judged ranks second, third, best; fused with retrieval ranks best, second, third.
        assert_eq!(ids(&scored.output), ["fused-second", "fused-best"]);
        assert_eq!(scored.score, Some(Probability::new(0.95).unwrap()));
        assert_eq!(scored.prompt_version, Some(RERANK_PROMPT_VERSION));
        let RerankOrder::Judged { raw } = &scored.record.order else {
            panic!("{:?}", scored.record.order);
        };
        let raw: BTreeMap<String, f32> = serde_json::from_str(raw).unwrap();
        assert_eq!(raw.len(), 3);
        assert_eq!(scored.output.0[0].score, Relevance::new(0.8).unwrap());
        let judged: Vec<_> = scored
            .record
            .chunks
            .iter()
            .map(|c| (c.score, c.probability.map(Probability::get)))
            .collect();
        assert_eq!(
            judged,
            [
                (Relevance::new(0.8).unwrap(), Some(0.95)),
                (Relevance::new(0.9).unwrap(), Some(0.2))
            ]
        );
    }

    #[tokio::test]
    async fn a_probability_outside_zero_to_one_declines_as_invalid_output() {
        for bad in ["p=1.7", "p=-0.3", "p=NaN"] {
            let error = judged(1, 25, by_text())
                .run(
                    &Ask::unrestricted("q"),
                    candidates(vec![hit("a", bad, 0.9)]),
                )
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
        let close = vec![
            hit("evidence", "p=0.6", 0.9),
            authoritative(hit("settled", "p=0.5", 0.8)),
        ];
        let scored = rerank(judged(1, 25, by_text()), close.clone()).await;
        assert_eq!(ids(&scored.output), ["settled"]);
        let scored = rerank(by_retrieval(1), close).await;
        assert_eq!(ids(&scored.output), ["settled"]);

        let mut far = vec![hit("evidence", "p=0.99", 0.9)];
        far.extend((0..40).map(|i| hit(&format!("filler-{i}"), "p=0.0", 0.5)));
        far.push(authoritative(hit("settled", "p=0.01", 0.1)));
        let scored = rerank(judged(1, 50, by_text()), far).await;
        assert_eq!(ids(&scored.output), ["evidence"]);
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
    async fn retrieval_order_is_unscored_and_records_its_mode() {
        let step = by_retrieval(2);
        assert_eq!(step.threshold(), None);
        assert_eq!(step.judged_by(), None);
        assert_eq!(step.mode(), RerankMode::RetrievalOrder);

        let scored = rerank(step, many()).await;

        // Retrieval order, the evidence penalty lifting the authoritative third match to the top.
        assert_eq!(ids(&scored.output), ["fused-third", "fused-best"]);
        assert_eq!(scored.score, None);
        assert_eq!(scored.model, None);
        assert_eq!(scored.record.order, RerankOrder::RetrievalOrder);
        assert!(scored.record.chunks.iter().all(|c| c.probability.is_none()));
    }

    #[tokio::test]
    async fn a_judge_scores_the_step_against_its_threshold_and_records_its_mode() {
        let judge = by_text();
        let threshold = Some(Probability::new(0.5).unwrap());
        let step = Rerank::<Open>::new(
            &judging(2, 3),
            RerankBy::Judge {
                judge: Arc::<FnJudge>::clone(&judge),
                threshold,
            },
        );
        assert_eq!(step.threshold(), threshold);
        assert_eq!(step.mode(), RerankMode::Judged);
        assert_eq!(step.judged_by(), Some(ModelRef::new("fake-fn-judge", "1")));

        let scored = rerank(
            step,
            vec![
                hit("fused-best", "p=0.2", 0.9),
                hit("fused-second", "p=0.95", 0.8),
                authoritative(hit("fused-third", "p=0.6", 0.7)),
                hit("fused-fourth", "p=1.0", 0.6),
            ],
        )
        .await;

        assert_eq!(judge.requests().len(), 3);
        assert_eq!(scored.record.order.mode(), RerankMode::Judged);
        assert_eq!(ids(&scored.output), ["fused-third", "fused-second"]);
        assert_eq!(scored.score, Some(Probability::new(0.95).unwrap()));
        assert_eq!(scored.prompt_version, Some(RERANK_PROMPT_VERSION));
    }

    #[test]
    fn a_long_section_is_judged_on_its_beginning() {
        let long = Retrieved {
            chunk: Chunk {
                text: "x".repeat(MAX_STATE_CHARS * 2),
                ..chunk("long", "Roles", "")
            },
            score: Relevance::new(1.0).unwrap(),
            context_for: None,
        };
        let state = judged_text(&long.chunk);
        assert!(state.starts_with("Roles > long\n\n"));
        assert_eq!(state.chars().filter(|c| *c == 'x').count(), MAX_STATE_CHARS);
    }
}
