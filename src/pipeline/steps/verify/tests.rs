use serde_json::json;

use super::*;
use crate::{
    config::MaxTokens,
    llm::{
        fakes::ScriptedChat,
        judge::{FnJudge, RawJudgement},
    },
    pipeline::{StepErrorKind, steps::Explain, steps::fixtures::chunk, trace::TraceRecord},
    retrieval::Relevance,
};

fn hit(locator: &str, text: &str) -> Retrieved {
    Retrieved {
        chunk: chunk(locator, "Roles", text),
        score: Relevance(0.5),
        context_for: None,
    }
}

fn id(locator: &str) -> String {
    hit(locator, "").chunk.id.0
}

/// Explains with a scripted reply of `claims`, each `(text, cited locators)`, over `sources`.
async fn explained(sources: Vec<Retrieved>, claims: &[(&str, &[&str])]) -> Explained {
    let reply = json!({
        "claims": claims
            .iter()
            .map(|(text, cited)| json!({
                "text": text,
                "chunk_ids": cited.iter().map(|locator| id(locator)).collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>(),
    });
    Explain::<Open>::new(
        Arc::new(ScriptedChat::replying([reply.to_string()])),
        MaxTokens::DEFAULT,
    )
    .run(&Question::new("q"), Reranked(sources))
    .await
    .unwrap()
    .output
}

/// Relates a claim to a text by the `relation=<option>:<p>` written in the text; says nothing without one.
fn by_text() -> Arc<FnJudge> {
    Arc::new(FnJudge::new(|text, question| {
        let JudgeQuestion::OneOf { options, .. } = question else {
            return RawJudgement::Missing;
        };
        let (name, p) = text
            .split("relation=")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|pair| pair.split_once(':'))
            .and_then(|(name, p)| Some((name, p.parse::<f32>().ok()?)))
            .unwrap_or(("says_nothing", 0.9));
        let rest = (1.0 - p) / (options.len() - 1) as f32;
        RawJudgement::OneOf(
            options
                .iter()
                .map(|option| if option.name == name { p } else { rest })
                .collect(),
        )
    }))
}

fn config(support: SupportCheck, publish: PublishPolicy) -> VerifyConfig {
    VerifyConfig { support, publish }
}

fn judged() -> SupportCheck {
    SupportCheck::SystemOne(SupportCutoff::default())
}

fn verify(config: VerifyConfig, judge: &Arc<FnJudge>) -> Verify<Open> {
    Verify::from_config(&config, Some(Arc::clone(judge) as Arc<dyn Judge>))
}

fn record(error: StepError) -> VerifyRecord {
    match error.record {
        Some(TraceRecord::Verify(record)) => record,
        other => panic!("expected a verify record, got {other:?}"),
    }
}

fn verdicts(record: &VerifyRecord) -> Vec<&ClaimVerdict> {
    record.claims.iter().map(|check| &check.verdict).collect()
}

#[tokio::test]
async fn membership_only_never_calls_the_judge_and_records_no_verdicts() {
    let judge = by_text();
    let step = verify(
        config(SupportCheck::Membership, PublishPolicy::Decline),
        &judge,
    );
    let input = explained(vec![hit("a", "")], &[("Roles expire `NEVER`.", &["a"])]).await;

    let scored = step.verify(input).await.unwrap();

    assert!(judge.requests().is_empty());
    assert_eq!(scored.record.support, SupportMode::Membership);
    assert!(scored.record.claims.is_empty());
    assert_eq!(scored.model, None);
    assert_eq!(scored.output.answer.claims().len(), 1);
}

#[tokio::test]
async fn an_unknown_citation_still_declines_before_any_judge_call() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Decline), &judge);
    let mut input = explained(vec![hit("a", ""), hit("b", "")], &[("A.", &["a", "b"])]).await;
    input.sources.0.truncate(1);

    let error = step.verify(input).await.err().unwrap();

    assert!(matches!(
        error.kind,
        StepErrorKind::Decline(DeclineReason::Unverified(_))
    ));
    assert_eq!(record(error).unknown_citations, [ChunkId(id("b"))]);
    assert!(judge.requests().is_empty());
}

#[tokio::test]
async fn a_quoted_span_missing_from_every_cited_chunk_is_fabricated_without_a_judge_call() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Decline), &judge);
    let input = explained(
        vec![hit(
            "a",
            "Assignments live in `ROLE_USERS`. relation=supports:0.9",
        )],
        &[("Assignments live in `ROLE_USERZ`.", &["a"])],
    )
    .await;

    let record = record(step.verify(input).await.err().unwrap());

    assert_eq!(
        verdicts(&record),
        [&ClaimVerdict::Fabricated {
            span: "ROLE_USERZ".into()
        }]
    );
    assert!(judge.requests().is_empty());
}

#[tokio::test]
async fn quotes_mode_passes_spans_found_in_a_cited_chunk() {
    let step =
        Verify::<Open>::from_config(&config(SupportCheck::Quotes, PublishPolicy::Decline), None);
    let input = explained(
        vec![hit(
            "a",
            "Assignments live in\n`ROLE_USERS`, per \"role assignment\".",
        )],
        &[(
            "They are in `role_users`, the “Role  assignment” table.",
            &["a"],
        )],
    )
    .await;

    let scored = step.verify(input).await.unwrap();

    assert_eq!(scored.record.support, SupportMode::Quotes);
    assert_eq!(verdicts(&scored.record), [&ClaimVerdict::QuotesFound]);
}

#[tokio::test]
async fn a_judge_check_without_a_judge_falls_back_to_quotes() {
    let step = Verify::<Open>::from_config(&config(judged(), PublishPolicy::Decline), None);
    assert_eq!(step.support(), SupportMode::Quotes);
    assert_eq!(step.judged_by(), None);
}

#[tokio::test]
async fn a_supported_claim_is_published_with_the_judge_recorded() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Decline), &judge);
    let input = explained(
        vec![
            hit("a", "relation=says_nothing:0.9"),
            hit("b", "relation=supports:0.8"),
        ],
        &[("Roles expire.", &["a", "b"])],
    )
    .await;

    let scored = step.verify(input).await.unwrap();

    assert_eq!(scored.record.support, SupportMode::Judged);
    assert!(matches!(
        verdicts(&scored.record)[..],
        [ClaimVerdict::Supported { by }] if by.chunk.0 == id("b") && by.relation == RelationKind::Supports
    ));
    assert_eq!(judge.requests().len(), 2);
    assert_eq!(scored.model, Some(ModelRef::new("fake-fn-judge", "1")));
    assert_eq!(scored.prompt_version, Some(SUPPORT_PROMPT_VERSION));
}

#[tokio::test]
async fn support_below_the_cutoff_or_a_contradiction_fails_the_claim() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Decline), &judge);
    let input = explained(
        vec![
            hit("weak", "relation=supports:0.6"),
            hit("against", "relation=contradicts:0.9"),
        ],
        &[("Weak.", &["weak"]), ("Against.", &["against"])],
    )
    .await;

    let error = step.verify(input).await.err().unwrap();

    assert!(matches!(
        &error.kind,
        StepErrorKind::Decline(DeclineReason::Unverified(reason)) if reason.starts_with("2 claim(s)")
    ));
    let record = record(error);
    assert!(matches!(
        verdicts(&record)[..],
        [ClaimVerdict::Unsupported { relations }, ClaimVerdict::Contradicted { .. }]
            if relations.len() == 1 && relations[0].relation == RelationKind::Supports
    ));
}

#[tokio::test]
async fn strip_publishes_only_the_claims_that_passed() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Strip), &judge);
    let input = explained(
        vec![hit("good", "relation=supports:0.9"), hit("silent", "")],
        &[
            ("Kept.", &["good"]),
            ("Dropped.", &["silent"]),
            ("Uncited.", &[]),
        ],
    )
    .await;

    let scored = step.verify(input).await.unwrap();

    let published: Vec<_> = scored
        .output
        .answer
        .claims()
        .iter()
        .map(|claim| claim.value.as_str())
        .collect();
    assert_eq!(published, ["Kept."]);
    assert_eq!(scored.record.stripped, 2);
    assert_eq!(scored.record.uncited_claims, ["Uncited."]);
}

#[tokio::test]
async fn strip_declines_when_no_claim_passes() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Strip), &judge);
    let input = explained(vec![hit("silent", "")], &[("Dropped.", &["silent"])]).await;

    let error = step.verify(input).await.err().unwrap();

    assert!(matches!(
        &error.kind,
        StepErrorKind::Decline(DeclineReason::Unverified(reason)) if reason == "no claim is backed by the chunks it cites"
    ));
    assert_eq!(record(error).stripped, 1);
}

#[tokio::test]
async fn an_out_of_range_relation_declines_as_invalid_output() {
    let judge = by_text();
    let step = verify(config(judged(), PublishPolicy::Decline), &judge);
    let input = explained(vec![hit("a", "relation=supports:1.5")], &[("A.", &["a"])]).await;

    let error = step.verify(input).await.err().unwrap();

    assert!(matches!(
        error.kind,
        StepErrorKind::Decline(DeclineReason::InvalidModelOutput(_))
    ));
}

#[test]
fn quoted_spans_skip_apostrophes_and_unclosed_marks() {
    assert_eq!(
        quoted_spans("The team's `ROLE_USERS` and \"the extract\" and “Dry-Run” and `open"),
        ["ROLE_USERS", "the extract", "Dry-Run"]
    );
    assert!(quoted_spans("No quotes, it's plain.").is_empty());
    assert!(quoted_spans("Empty `` and \"  \".").is_empty());
}
