"""Recall System One: batch noul scores for (question, candidate) pairs."""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("systemone")
logging.basicConfig(level=logging.INFO)

SERVICE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "recall-systemone"
LAYA_REPO = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
RERANKER_ID = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
PROB_CLIP = 1e-6


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str | None) -> float | None:
    if name is None:
        return None
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def admit_batch_size() -> int:
    n = env_int("ADMIT_BATCH_SIZE", 32)
    if n < 1:
        raise ValueError("ADMIT_BATCH_SIZE must be >= 1")
    return n


def logit_from_prob(p: float) -> float:
    p = min(max(float(p), PROB_CLIP), 1.0 - PROB_CLIP)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_temperature(logit: float, temperature: float) -> float:
    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be > 0")
    return min(max(sigmoid(logit / t), 0.0), 1.0)


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Calibration:
    def __init__(self, temperature: float, fitted: bool, note: str | None = None):
        self.temperature = float(temperature)
        self.fitted = bool(fitted)
        self.note = note

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "temperature": self.temperature,
            "fitted": self.fitted,
        }
        if self.note:
            out["note"] = self.note
        return out


def load_calibration(path: Path | None = None) -> Calibration:
    path = path or (SERVICE_DIR / "calibration.json")
    temperature = 1.0
    fitted = False
    note = None
    if path.is_file():
        data = json.loads(path.read_text())
        temperature = float(data.get("temperature", 1.0))
        fitted = bool(data.get("fitted", True))
        note = data.get("note")
    env_t = env_float("ADMIT_TEMPERATURE")
    if env_t is not None:
        temperature = env_t
        if not path.is_file():
            fitted = False
    return Calibration(temperature, fitted, note)


class Backend(Protocol):
    name: str
    model_id: str
    device: str

    def warmup(self) -> None: ...

    def logits(
        self,
        *,
        question: str,
        as_of: str | None,
        candidates: list[dict[str, Any]],
        questions: dict[str, dict[str, Any]],
        batch_size: int,
    ) -> dict[str, float]: ...


class LayaBackend:
    """Laya Agent.predict — verified API from laya 0.3.5 / convaiinnovations/laya."""

    name = "laya"
    model_id = LAYA_REPO

    def __init__(self) -> None:
        os.environ.setdefault("USE_TF", "0")
        import laya

        self.agent = laya.load(LAYA_REPO, device="cpu")
        self.device = str(getattr(self.agent, "device", "cpu"))

    def warmup(self) -> None:
        self.logits(
            question="warmup",
            as_of=None,
            candidates=[{"id": "w", "origin": "personal", "grantor": None, "text": "warmup"}],
            questions={"w": {"type": "noul", "instructions": "warmup"}},
            batch_size=admit_batch_size(),
        )

    def logits(
        self,
        *,
        question: str,
        as_of: str | None,
        candidates: list[dict[str, Any]],
        questions: dict[str, dict[str, Any]],
        batch_size: int,
    ) -> dict[str, float]:
        """Score each (question, candidate) pair against its OWN state.

        One state per candidate, one `predict` per pair. This is the cross-encoder
        shape: the question and exactly one passage are encoded jointly, which is
        what produces a relevance judgement instead of a constant.

        Two alternatives were tried and both are wrong, so they are recorded here:

        1. All candidates in one shared state, N questions with identical
           instructions. Laya's `system_one` calls `build_sequence(tok, state, q)`
           per question and batches with `collate_items`, so the state is shared and
           only `q` varies — identical instructions therefore produce identical
           sequences. It returned 0.6649 for every candidate, including a wifi
           password scored against a question about a birthday.
        2. Candidate text moved into each question's `instructions`, keeping one
           batched call. This looked excellent on a three-candidate synthetic probe
           (0.83 / 0.41 / 0.16 in 0.33s) and was badly wrong on the real corpus: for
           "When is my sister Ana's birthday?" it scored "Maya's work badge PIN is
           3301." at 0.410 and the actual Ana birthday chunk at 0.135. `instructions`
           is not where Laya expects content to be judged.

        The cost of doing it correctly is latency: roughly 15s for 32 candidates on
        CPU, versus 6s for the batched-but-wrong version. That is the single largest
        cost in the ask path and the main thing a GPU deployment buys back.
        """
        by_id = {c["id"]: c for c in candidates}
        qids = list(questions.keys())

        def score_one(qid: str) -> tuple[str, float]:
            cand = by_id[qid]
            state: dict[str, Any] = {
                "question": question,
                "candidate": cand.get("text", ""),
            }
            if cand.get("grantor"):
                state["shared_by"] = cand["grantor"]
            if as_of is not None:
                state["as_of"] = as_of

            result = self.agent.predict(state, {qid: questions[qid]})
            return qid, logit_from_prob(float(result["answers"][qid]["noul"]))

        # Torch releases the GIL during the forward pass so these overlap, but it is
        # already threading across cores internally; an unbounded pool thrashes.
        workers = max(1, min(len(qids), batch_size, (os.cpu_count() or 4) // 2))
        out: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for qid, logit in pool.map(score_one, qids):
                out[qid] = logit
        return out


class RerankerBackend:
    """BAAI/bge-reranker-base via sentence-transformers CrossEncoder."""

    name = "reranker"
    model_id = RERANKER_ID

    def __init__(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self._torch = torch
        self.model = CrossEncoder(RERANKER_ID, device="cpu")
        self.device = "cpu"

    def warmup(self) -> None:
        self.logits(
            question="warmup",
            as_of=None,
            candidates=[{"id": "w", "text": "warmup"}],
            questions={"w": {"type": "noul", "instructions": "warmup"}},
            batch_size=admit_batch_size(),
        )

    def logits(
        self,
        *,
        question: str,
        as_of: str | None,
        candidates: list[dict[str, Any]],
        questions: dict[str, dict[str, Any]],
        batch_size: int,
    ) -> dict[str, float]:
        del as_of
        by_id = {c["id"]: c for c in candidates}
        qids = list(questions.keys())
        pairs = [(question, by_id[qid]["text"]) for qid in qids]
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            activation_fn=self._torch.nn.Identity(),
        )
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [float(scores)]
        return {qid: float(s) for qid, s in zip(qids, scores)}


def load_backend(choice: str | None = None) -> Backend:
    choice = (choice or os.environ.get("SYSTEMONE_BACKEND") or "auto").strip().lower()
    if choice not in {"laya", "reranker", "auto"}:
        raise ValueError("SYSTEMONE_BACKEND must be laya, reranker, or auto")

    def laya() -> Backend:
        log.info("loading Laya backend from %s", LAYA_REPO)
        return LayaBackend()

    def reranker() -> Backend:
        log.info("loading reranker backend %s", RERANKER_ID)
        return RerankerBackend()

    if choice == "laya":
        return laya()
    if choice == "reranker":
        return reranker()
    try:
        return laya()
    except Exception:
        log.exception("Laya unavailable; falling back to reranker")
        return reranker()


class Candidate(BaseModel):
    id: str
    origin: str | None = None
    grantor: str | None = None
    text: str


class State(BaseModel):
    question: str
    as_of: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)


class TypedQuestion(BaseModel):
    type: str
    instructions: str | None = None


class SystemOneRequest(BaseModel):
    model: str = MODEL_NAME
    state: State
    questions: dict[str, TypedQuestion]


def create_app(
    *,
    backend: Backend | None = None,
    calibration: Calibration | None = None,
    skip_load: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.calibration = calibration or load_calibration()
        if backend is not None:
            app.state.backend = backend
        elif skip_load:
            app.state.backend = None
        else:
            app.state.backend = load_backend()
        if app.state.backend is not None:
            app.state.backend.warmup()
            app.state.warm = True
        else:
            app.state.warm = False
        yield

    application = FastAPI(title="Recall System One", lifespan=lifespan)

    @application.get("/healthz")
    def healthz(request: Request):
        be = getattr(request.app.state, "backend", None)
        warm = bool(getattr(request.app.state, "warm", False))
        body = {
            "ok": warm,
            "backend": getattr(be, "name", None),
            "model": getattr(be, "model_id", None),
            "device": getattr(be, "device", "cpu"),
            "warm": warm,
        }
        if not warm:
            return JSONResponse(status_code=503, content=body)
        return body

    @application.post("/v1/systemone")
    def systemone(request: Request, req: SystemOneRequest):
        be: Backend | None = getattr(request.app.state, "backend", None)
        if be is None or not getattr(request.app.state, "warm", False):
            raise HTTPException(status_code=503, detail="model not ready")

        cand_ids = [c.id for c in req.state.candidates]
        if len(cand_ids) != len(set(cand_ids)):
            raise HTTPException(status_code=400, detail="duplicate candidate ids")
        known = set(cand_ids)
        missing = [qid for qid in req.questions if qid not in known]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "questions keys must match state.candidates ids; "
                    f"no candidate for: {', '.join(missing)}"
                ),
            )
        bad_types = [
            qid for qid, q in req.questions.items() if q.type != "noul"
        ]
        if bad_types:
            raise HTTPException(
                status_code=400,
                detail="only noul is supported "
                f"(got type {req.questions[bad_types[0]].type!r} for {bad_types[0]!r})",
            )

        cal: Calibration = request.app.state.calibration
        scored = [c.model_dump() for c in req.state.candidates if c.id in req.questions]
        qmap = {
            qid: {"type": q.type, "instructions": q.instructions or ""}
            for qid, q in req.questions.items()
        }
        if not qmap:
            logits: dict[str, float] = {}
        else:
            logits = be.logits(
                question=req.state.question,
                as_of=req.state.as_of,
                candidates=scored,
                questions=qmap,
                batch_size=admit_batch_size(),
            )

        answers = {}
        for qid in req.questions:
            if qid not in logits:
                raise HTTPException(
                    status_code=500,
                    detail=f"backend returned no score for {qid!r}",
                )
            answers[qid] = {"noul": apply_temperature(logits[qid], cal.temperature)}

        return {
            "model": req.model or MODEL_NAME,
            "answers": answers,
            "backend": be.name,
            "calibration": cal.as_dict(),
        }

    return application


app = create_app()
