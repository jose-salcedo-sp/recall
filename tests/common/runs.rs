//! One run report per outcome, shared by the `RunStore` tests.

use drew::{
    domain::{
        AtLeastOne, CategoryKind, ChunkId, ChunkRef, ChunkVersion, Cited, Probability, Provenance,
        Question, RunId, Span,
    },
    pipeline::{
        DeclineReason,
        report::{Answer, Citation, Outcome, Rejected, RunReport, Verdict},
        trace::Trace,
    },
};
use serde_json::json;

fn citation() -> Citation {
    Citation {
        chunk: ChunkRef {
            id: ChunkId("markdown:roles.md#roles".into()),
            version: ChunkVersion("v1".into()),
        },
        title: "Roles".into(),
        url: "https://wiki.example.test/1".into(),
        excerpt: "PFCG maintains roles.".into(),
    }
}

fn trace() -> Trace {
    serde_json::from_value(json!([{
        "step": "rerank",
        "group": null,
        "started_at": "2026-10-09T12:00:01.123456789Z",
        "duration_ms": 60,
        "model": { "id": "system-one", "version": "1" },
        "prompt_version": "rerank-v1",
        "outcome": { "status": "declined", "reason": { "kind": "low_confidence" }, "score": 0.25, "threshold": 0.5 },
        "record": { "kind": "rerank", "raw": "[1]", "chunks": [
            { "chunk": { "id": "markdown:roles.md#roles", "version": "v1" }, "score": 0.25 }
        ] }
    }]))
    .expect("the fixture trace matches the trace schema")
}

/// An answered, a declined and a failed run, each with its `pipeline_runs.status`.
pub fn reports() -> Vec<(RunReport, &'static str)> {
    let report = |category, outcome| RunReport {
        id: RunId::new(),
        question: Question::new("PFCG or SU01 for roles?"),
        category,
        outcome,
        trace: trace(),
    };
    let prompt = |start, end| AtLeastOne::one(Provenance::UserPrompt(Span { start, end }));
    let answered = Outcome::Answered(Answer {
        verdict: Some(Verdict::Choice {
            selected: Cited {
                value: "PFCG".into(),
                sources: prompt(0, 4),
            },
            rejected: vec![Rejected {
                option: Cited {
                    value: "SU01".into(),
                    sources: prompt(8, 12),
                },
                reason: "edits users".into(),
            }],
        }),
        why: vec![Cited {
            value: "PFCG maintains roles.".into(),
            sources: AtLeastOne::one(Provenance::Chunk(citation().chunk.id)),
        }],
        citations: vec![citation()],
    });
    let declined = Outcome::Declined {
        at: "answer".into(),
        reason: DeclineReason::InvalidModelOutput("missing field `selected`".into()),
        score: None,
        threshold: Some(Probability::new(0.7).unwrap()),
        sources: vec![citation()],
    };
    let failed = Outcome::Failed {
        at: "classify".into(),
        error: "connection refused".into(),
    };
    vec![
        (report(Some(CategoryKind::Choice), answered), "answered"),
        (report(Some(CategoryKind::Open), declined), "declined"),
        (report(None, failed), "failed"),
    ]
}
