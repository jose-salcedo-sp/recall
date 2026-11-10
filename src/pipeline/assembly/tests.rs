use super::*;
use crate::llm::{
    fakes::ScriptedChat,
    judge::{FnJudge, RawJudgement},
};
use crate::retrieval::fakes::KeywordRetriever;

fn ports(judge: Option<Arc<dyn Judge>>) -> Ports {
    Ports {
        retriever: Arc::new(KeywordRetriever { chunks: Vec::new() }),
        embedding: None,
        explain: Arc::new(ScriptedChat::replying(Vec::<String>::new())),
        judge,
        authorizer: Arc::new(crate::access::fakes::StaticAuthorizer::tester()),
        clock: Arc::new(crate::clock::fakes::FixedClock::default()),
        ids: Arc::new(crate::clock::fakes::SequentialIds::default()),
    }
}

fn with_judge() -> Ports {
    ports(Some(Arc::new(FnJudge::new(|_, _| RawJudgement::Missing))))
}

fn config(json: &str) -> PipelineConfig {
    PipelineConfig::parse(json).unwrap()
}

const EVERY_JUDGED_CHECK: &str = r#"{
    "rerank": {"judge": {"candidates": 25}},
    "admit": {"judge": {}},
    "verify": {"support": {"judged": {}}}
}"#;

#[test]
fn the_shipped_configs_assemble_without_a_judge() {
    for shipped in [
        include_str!("../../../config/recall.json"),
        include_str!("../../../config/recall-workers-ai.json"),
    ] {
        let plan = assemble(&config(shipped), &ports(None)).unwrap().plan;
        assert_eq!(
            plan,
            Plan {
                classify: ClassifyMode::AllOpen,
                rerank: RerankMode::RetrievalOrder,
                admit: AdmitMode::Off,
                support: SupportMode::Quotes,
            }
        );
    }
}

#[test]
fn every_judged_check_without_a_judge_is_named_in_one_error() {
    let error = assemble(&config(EVERY_JUDGED_CHECK), &ports(None))
        .err()
        .unwrap();

    assert_eq!(
        error.checks(),
        [
            JudgedCheck::Rerank,
            JudgedCheck::Admit,
            JudgedCheck::Support
        ]
    );
    assert_eq!(
        error.to_string(),
        r#"no judge is configured, but rerank.judge needs one (or set it to "none"); admit.judge needs one (or set it to "none"); verify.support needs one (or set it to "quotes" or "membership"); configure a judge backend"#
    );
}

#[test]
fn one_judged_check_without_a_judge_fails_alone() {
    for (json, check) in [
        (
            r#"{"rerank": {"judge": {"candidates": 25}}}"#,
            JudgedCheck::Rerank,
        ),
        (r#"{"admit": {"judge": {}}}"#, JudgedCheck::Admit),
        (
            r#"{"verify": {"support": {"judged": {"min_support": 0.9}}}}"#,
            JudgedCheck::Support,
        ),
    ] {
        let error = assemble(&config(json), &ports(None)).err().unwrap();
        assert_eq!(error.checks(), [check], "{json}");
    }
}

#[test]
fn with_a_judge_every_check_runs_as_configured() {
    let assembled = assemble(&config(EVERY_JUDGED_CHECK), &with_judge()).unwrap();

    assert_eq!(
        assembled.plan,
        Plan {
            classify: ClassifyMode::Judged,
            rerank: RerankMode::Judged,
            admit: AdmitMode::Judged,
            support: SupportMode::Judged,
        }
    );
    let boolean = assembled.boolean.unwrap();
    assert_eq!(boolean.rerank.mode(), RerankMode::Judged);
    assert!(boolean.admit.is_some());
    assert_eq!(boolean.pipeline.1.1.support(), SupportMode::Judged);
}

#[test]
fn without_a_judge_there_is_no_boolean_path() {
    let assembled = assemble(&PipelineConfig::default(), &ports(None)).unwrap();
    assert!(assembled.boolean.is_none());
    assert_eq!(assembled.classify.mode(), ClassifyMode::AllOpen);
    assert_eq!(assembled.plan.support, SupportMode::Membership);
}

#[test]
fn a_rerank_threshold_needs_a_judged_rerank() {
    for json in [
        r#"{"thresholds": {"rerank": 0.5}}"#,
        r#"{"per_category": {"open": {"rerank": 0.5}}}"#,
        r#"{"per_category": {"boolean": {"rerank": 0.5}}}"#,
    ] {
        for ports in [ports(None), with_judge()] {
            assert!(
                matches!(
                    assemble(&config(json), &ports),
                    Err(Unassembled::UnscoredRerank)
                ),
                "{json}"
            );
        }
    }
    let judged =
        config(r#"{"rerank": {"judge": {"candidates": 25}}, "thresholds": {"rerank": 0.5}}"#);
    let assembled = assemble(&judged, &with_judge()).unwrap();
    assert_eq!(
        crate::pipeline::Step::threshold(&assembled.open.rerank),
        Some(crate::domain::Probability::new(0.5).unwrap())
    );
}
