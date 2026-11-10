use serde_json::json;

use crate::{
    domain::{AtLeastOne, ChunkId, ChunkRef, ChunkVersion, Cited, Probability, Provenance, Span},
    pipeline::{
        DeclineReason,
        report::{Answer, Citation, Outcome, Verdict},
        trace::Trace,
    },
};

pub fn chunk_id(n: usize) -> ChunkId {
    ChunkId(format!("markdown:doc.md#s{n}"))
}

pub fn citation(n: usize) -> Citation {
    Citation {
        chunk: ChunkRef {
            id: chunk_id(n),
            version: ChunkVersion(format!("v{n}")),
        },
        title: format!("Page {n}"),
        url: format!("https://wiki.example.test/{n}"),
        excerpt: format!("Excerpt {n}."),
    }
}

pub fn cited(value: &str, chunks: &[usize]) -> Cited<String> {
    let sources = chunks
        .iter()
        .map(|&n| Provenance::Chunk(chunk_id(n)))
        .collect::<Vec<_>>();
    Cited {
        value: value.into(),
        sources: AtLeastOne::try_from(sources).expect("at least one chunk"),
    }
}

pub fn from_prompt(value: &str) -> Cited<String> {
    Cited {
        value: value.into(),
        sources: AtLeastOne::one(Provenance::UserPrompt(Span { start: 0, end: 1 })),
    }
}

pub fn answered() -> Outcome {
    Outcome::Answered(Answer {
        verdict: Some(Verdict::Boolean {
            yes: true,
            confidence: Probability::new(0.9).unwrap(),
        }),
        why: vec![cited("AGR_USERS assigns roles to users.", &[1])],
        citations: vec![citation(1)],
    })
}

pub fn declined() -> Outcome {
    Outcome::Declined {
        at: "rerank".into(),
        reason: DeclineReason::LowConfidence,
        score: Some(Probability::new(0.25).unwrap()),
        threshold: Some(Probability::new(0.5).unwrap()),
        sources: vec![citation(2), citation(1)],
    }
}

pub fn failed() -> Outcome {
    Outcome::Failed {
        at: "explain".into(),
        error: "bedrock timed out".into(),
    }
}

/// Classify and retrieve in parallel, then rerank.
pub fn trace() -> Trace {
    serde_json::from_value(json!([
        {
            "step": "classify",
            "group": "classify_retrieve",
            "started_at": "2026-10-09T12:00:00Z",
            "duration_ms": 40,
            "model": { "id": "system-one", "version": "1" },
            "prompt_version": "classify-v1",
            "outcome": { "status": "passed", "score": 0.875, "threshold": 0.5 },
            "record": { "kind": "classify", "raw": "boolean", "category": "boolean" }
        },
        {
            "step": "retrieve",
            "group": "classify_retrieve",
            "started_at": "2026-10-09T12:00:00Z",
            "duration_ms": 25,
            "model": null,
            "prompt_version": null,
            "outcome": { "status": "passed", "score": null, "threshold": null },
            "record": { "kind": "retrieve", "chunks": [
                { "chunk": { "id": "markdown:doc.md#s1", "version": "v1" }, "score": 0.03125 }
            ] }
        },
        {
            "step": "rerank",
            "group": null,
            "started_at": "2026-10-09T12:00:01Z",
            "duration_ms": 60,
            "model": { "id": "system-one", "version": "1" },
            "prompt_version": "rerank-v1",
            "outcome": { "status": "passed", "score": 0.75, "threshold": 0.5 },
            "record": { "kind": "rerank", "raw": "[1]", "chunks": [
                { "chunk": { "id": "markdown:doc.md#s1", "version": "v1" }, "score": 0.75 }
            ] }
        }
    ]))
    .expect("trace fixture deserializes")
}
