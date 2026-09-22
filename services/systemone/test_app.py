"""Request-validation and JSON-shape tests. Model is mocked — no weight download."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

import app as appmod
from app import Calibration, create_app, logit_from_prob


class FakeBackend:
    name = "mock"
    model_id = "fake-model"
    device = "cpu"

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[dict] = []
        self.choice_calls: list[dict] = []

    def warmup(self) -> None:
        return None

    def logits(self, *, question, as_of, candidates, questions, batch_size):
        self.calls.append(
            {
                "question": question,
                "as_of": as_of,
                "candidates": candidates,
                "questions": questions,
                "batch_size": batch_size,
            }
        )
        out = {}
        for qid in questions:
            if qid in self.scores:
                out[qid] = self.scores[qid]
            else:
                out[qid] = logit_from_prob(0.5)
        return out

    def state_answers(self, *, question, as_of, candidates, questions, extra=None):
        self.choice_calls.append(
            {
                "question": question,
                "as_of": as_of,
                "candidates": candidates,
                "questions": questions,
                "extra": extra,
            }
        )
        canned = getattr(self, "choices", None) or {}
        out = {}
        for qid in questions:
            out[qid] = canned.get(
                qid,
                {"choice": "atomic_lookup", "confidence": 0.9, "probabilities": {}},
            )
        return out


SAMPLE = {
    "model": "recall-systemone",
    "state": {
        "question": "When is my sister's birthday?",
        "as_of": "2026-09-21T18:00:00Z",
        "candidates": [
            {
                "id": "m0",
                "origin": "personal",
                "grantor": None,
                "text": "My sister Ana's birthday is March 14.",
            },
            {
                "id": "m1",
                "origin": "personal",
                "grantor": None,
                "text": "I bought Ana a gift card last year.",
            },
        ],
    },
    "questions": {
        "m0": {
            "type": "noul",
            "instructions": (
                "This candidate is sufficient evidence to answer the question. "
                "It is not merely on a related topic."
            ),
        },
        "m1": {
            "type": "noul",
            "instructions": (
                "This candidate is sufficient evidence to answer the question. "
                "It is not merely on a related topic."
            ),
        },
    },
}


def make_client(backend=None, calibration=None) -> TestClient:
    be = backend or FakeBackend(
        {"m0": logit_from_prob(0.94), "m1": logit_from_prob(0.06)}
    )
    cal = calibration or Calibration(1.0, False)
    application = create_app(backend=be, calibration=cal)
    return TestClient(application)


def test_mismatched_question_id_is_400():
    body = {
        "model": "recall-systemone",
        "state": SAMPLE["state"],
        "questions": {
            "nope": {"type": "noul", "instructions": "x"},
        },
    }
    with make_client() as client:
        r = client.post("/v1/systemone", json=body)
    assert r.status_code == 400
    assert "no candidate" in r.json()["detail"].lower() or "must match" in r.json()["detail"].lower()


def test_choice_without_criteria_is_400():
    body = {
        "model": "recall-systemone",
        "state": SAMPLE["state"],
        "questions": {
            "kind": {"type": "choice", "instructions": "x"},
        },
    }
    with make_client() as client:
        r = client.post("/v1/systemone", json=body)
    assert r.status_code == 400
    assert "criteria" in r.json()["detail"].lower()


def test_every_question_key_gets_an_answer_and_json_shape():
    fake = FakeBackend(
        {"m0": logit_from_prob(0.94), "m1": logit_from_prob(0.06)}
    )
    with make_client(backend=fake) as client:
        r = client.post("/v1/systemone", json=SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "recall-systemone"
    assert data["backend"] == "mock"
    assert data["calibration"] == {"temperature": 1.0, "fitted": False}
    assert set(data["answers"]) == set(SAMPLE["questions"])
    for qid, ans in data["answers"].items():
        assert set(ans) == {"noul"}
        assert isinstance(ans["noul"], float)
        assert 0.0 <= ans["noul"] <= 1.0
    assert math.isclose(data["answers"]["m0"]["noul"], 0.94, abs_tol=1e-5)
    assert math.isclose(data["answers"]["m1"]["noul"], 0.06, abs_tol=1e-5)
    assert len(fake.calls) == 1
    assert fake.calls[0]["batch_size"] == 32
    assert {c["id"] for c in fake.calls[0]["candidates"]} == {"m0", "m1"}


def test_healthz_200_when_warm():
    with make_client() as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "backend": "mock",
        "model": "fake-model",
        "device": "cpu",
        "warm": True,
    }


def test_temperature_is_applied():
    logit = logit_from_prob(0.94)
    fake = FakeBackend({"m0": logit})
    body = {
        "model": "recall-systemone",
        "state": {
            "question": "q",
            "candidates": [{"id": "m0", "origin": "personal", "grantor": None, "text": "t"}],
        },
        "questions": {"m0": {"type": "noul", "instructions": "x"}},
    }
    with make_client(backend=fake, calibration=Calibration(2.0, True, "unit")) as client:
        r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    data = r.json()
    expected = appmod.apply_temperature(logit, 2.0)
    assert math.isclose(data["answers"]["m0"]["noul"], expected, abs_tol=1e-9)
    assert data["answers"]["m0"]["noul"] < 0.94
    assert data["calibration"]["temperature"] == 2.0
    assert data["calibration"]["fitted"] is True
    assert data["calibration"]["note"] == "unit"


def test_four_nouls_per_candidate():
    fake = FakeBackend(
        {
            "m0_injection": logit_from_prob(0.1),
            "m0_contradicts": logit_from_prob(0.1),
            "m0_relevant": logit_from_prob(0.8),
            "m0_evidence": logit_from_prob(0.9),
        }
    )
    body = {
        "model": "recall-systemone",
        "state": {
            "question": "q",
            "candidates": [{"id": "m0", "origin": "personal", "grantor": None, "text": "t"}],
        },
        "questions": {
            "m0_injection": {"type": "noul", "instructions": "i", "about": "m0"},
            "m0_contradicts": {"type": "noul", "instructions": "c", "about": "m0"},
            "m0_relevant": {"type": "noul", "instructions": "r", "about": "m0"},
            "m0_evidence": {"type": "noul", "instructions": "e", "about": "m0"},
        },
    }
    with make_client(backend=fake) as client:
        r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    ans = r.json()["answers"]
    assert set(ans) == set(body["questions"])
    assert math.isclose(ans["m0_evidence"]["noul"], 0.9, abs_tol=1e-5)
    assert len(fake.calls) == 1

