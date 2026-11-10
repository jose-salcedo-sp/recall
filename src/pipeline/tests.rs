use super::{
    trace::{StepOutcome, TraceRecord, VerifyRecord},
    *,
};
use crate::{
    access::{
        Access,
        fakes::{FailingAuthorizer, StaticAuthorizer, tester},
    },
    domain::Probability,
};

struct Fetched(u32);
struct Doubled(u32);

struct Fetch<TCategory> {
    value: u32,
    score: Probability,
    threshold: Option<Probability>,
    category: PhantomData<TCategory>,
}

impl<TCategory: Category> Step for Fetch<TCategory> {
    const NAME: &'static str = "fetch";
    type Category = TCategory;
    type Input = Asked;
    type Output = Fetched;
    type Record = VerifyRecord;

    fn threshold(&self) -> Option<Probability> {
        self.threshold
    }

    async fn run(
        &self,
        _ask: &Ask,
        _input: Asked,
    ) -> Result<Scored<Fetched, VerifyRecord>, StepError> {
        Ok(Scored::new(Fetched(self.value), empty_record()).with_score(self.score))
    }
}

struct Double;

impl Step for Double {
    const NAME: &'static str = "double";
    type Category = Open;
    type Input = Fetched;
    type Output = Doubled;
    type Record = VerifyRecord;

    async fn run(
        &self,
        _ask: &Ask,
        input: Fetched,
    ) -> Result<Scored<Doubled, VerifyRecord>, StepError> {
        Ok(Scored::new(Doubled(input.0 * 2), empty_record()))
    }
}

struct Refuse;

impl Step for Refuse {
    const NAME: &'static str = "refuse";
    type Category = Unclassified;
    type Input = Asked;
    type Output = ();
    type Record = VerifyRecord;

    async fn run(&self, _ask: &Ask, _input: Asked) -> Result<Scored<(), VerifyRecord>, StepError> {
        Err(
            StepError::decline(DeclineReason::InvalidModelOutput("not json".into()))
                .with_record(VerifyRecord {
                    uncited_claims: vec!["raw".into()],
                    ..VerifyRecord::default()
                })
                .with_model(ModelRef::new("refuser", "1"), "refuse-v1"),
        )
    }
}

fn empty_record() -> VerifyRecord {
    VerifyRecord::default()
}

fn open_run() -> Run<Open, Asked> {
    Run::new(Ask::unrestricted("does it work?")).classified()
}

fn probability(value: f32) -> Probability {
    Probability::new(value).unwrap()
}

fn fetch<TCategory>(score: f32, threshold: Option<f32>) -> Fetch<TCategory> {
    Fetch {
        value: 21,
        score: probability(score),
        threshold: threshold.map(probability),
        category: PhantomData,
    }
}

#[tokio::test]
async fn seq_drives_every_step_and_traces_each() {
    let pipeline = Seq(fetch(0.9, Some(0.5)), Double);

    let run = pipeline.drive(open_run()).await.unwrap();

    assert_eq!(run.state().0, 42);
    let steps: Vec<_> = run
        .trace()
        .entries()
        .iter()
        .map(|e| e.step.as_str())
        .collect();
    assert_eq!(steps, ["fetch", "double"]);
}

#[tokio::test]
async fn score_below_threshold_halts_with_the_trace_so_far() {
    let pipeline = Seq(fetch(0.2, Some(0.5)), Double);

    let halted = pipeline.drive(open_run()).await.err().unwrap();

    assert_eq!(halted.at, "fetch");
    assert_eq!(halted.category, Some(CategoryKind::Open));
    assert!(matches!(
        halted.halt,
        Halt::Declined {
            reason: DeclineReason::LowConfidence,
            ..
        }
    ));
    assert_eq!(halted.trace.entries().len(), 1);
}

#[tokio::test]
async fn declined_step_keeps_its_record_and_model() {
    let halted = Run::new(Ask::unrestricted("q"))
        .then(&Refuse)
        .await
        .err()
        .unwrap();

    assert_eq!(halted.category, None);
    let entry = &halted.trace.entries()[0];
    assert!(matches!(entry.outcome, StepOutcome::Declined { .. }));
    assert!(matches!(entry.record, Some(TraceRecord::Verify(_))));
    assert_eq!(entry.model, Some(ModelRef::new("refuser", "1")));
    assert_eq!(entry.prompt_version.as_deref(), Some("refuse-v1"));
}

#[tokio::test]
async fn both_traces_the_other_branch_when_one_halts() {
    let halted = Run::new(Ask::unrestricted("q"))
        .both("classify_retrieve", &Refuse, &fetch(0.9, None))
        .await
        .err()
        .unwrap();

    assert_eq!(halted.at, "refuse");
    let entries = halted.trace.entries();
    assert_eq!(entries.len(), 2);
    assert!(
        entries
            .iter()
            .all(|e| e.group.as_deref() == Some("classify_retrieve"))
    );
}

#[test]
fn trace_round_trips_through_json() {
    let entry = TraceEntry {
        step: "verify".into(),
        group: None,
        started_at: Utc::now(),
        duration_ms: 3,
        model: None,
        prompt_version: None,
        outcome: StepOutcome::Declined {
            reason: DeclineReason::Unverified("claim 2 has no citation".into()),
            score: None,
            threshold: None,
        },
        record: Some(empty_record().into()),
        access: Vec::new(),
    };
    let json = serde_json::to_string(&entry).unwrap();
    assert_eq!(serde_json::from_str::<TraceEntry>(&json).unwrap(), entry);
}

#[test]
fn stored_scores_stay_bare_numbers_and_are_checked_on_read() {
    let outcome = |score: &str| {
        format!(
            r#"{{"status": "declined", "reason": {{"kind": "low_confidence"}}, "score": {score}, "threshold": 0.5}}"#
        )
    };
    let read = serde_json::from_str::<StepOutcome>(&outcome("0.25")).unwrap();
    assert_eq!(
        read,
        StepOutcome::Declined {
            reason: DeclineReason::LowConfidence,
            score: Some(probability(0.25)),
            threshold: Some(probability(0.5)),
        }
    );
    assert_eq!(
        serde_json::to_value(&read).unwrap(),
        serde_json::from_str::<serde_json::Value>(&outcome("0.25")).unwrap()
    );
    assert!(serde_json::from_str::<StepOutcome>(&outcome("1.7")).is_err());
}

#[tokio::test]
async fn authorize_records_who_asked_and_what_they_may_read() {
    let authorizer = StaticAuthorizer::tester();

    let run = Run::authorize(Question::new("q"), tester(), &authorizer)
        .await
        .unwrap();

    assert_eq!(run.ask().principal(), &tester());
    assert_eq!(run.ask().access(), &crate::access::fakes::unrestricted());
    let entry = &run.trace().entries()[0];
    assert_eq!(entry.step, AUTHORIZE);
    let Some(TraceRecord::Authorize(record)) = &entry.record else {
        panic!("expected an authorize record, got {:?}", entry.record);
    };
    assert_eq!(record.principal, tester());
    assert_eq!(record.policy, authorizer.policy());
    assert_eq!(record.access.as_ref(), Some(run.ask().access()));
}

#[tokio::test]
async fn an_unknown_principal_runs_with_no_access() {
    let run = Run::authorize(
        Question::new("q"),
        Principal::new(crate::domain::UserId("stranger".into())),
        &StaticAuthorizer::tester(),
    )
    .await
    .unwrap();

    assert_eq!(run.ask().access(), &Access::none());
}

#[tokio::test]
async fn a_policy_that_cannot_answer_fails_the_run_before_any_step() {
    let halted = Run::authorize(Question::new("q"), tester(), &FailingAuthorizer)
        .await
        .err()
        .unwrap();

    assert_eq!(halted.at, AUTHORIZE);
    assert_eq!(halted.category, None);
    assert!(matches!(halted.halt, Halt::Failed { .. }));
    let entries = halted.trace.entries();
    assert_eq!(entries.len(), 1);
    assert!(matches!(entries[0].outcome, StepOutcome::Failed { .. }));
    let Some(TraceRecord::Authorize(record)) = &entries[0].record else {
        panic!("expected an authorize record");
    };
    assert_eq!(record.access, None);
}
