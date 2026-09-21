# Recall System One

HTTP service that scores a question plus a batch of candidate memory chunks and returns a calibrated **noul** per candidate: P(this chunk is *sufficient evidence* to answer the question), not merely on a related topic.

Wire protocol matches TypeSafe AI’s Jev System One API (`POST /v1/systemone`). The hosted model is Jev; this process serves the same JSON with an open backend.

## Backends

Selected with `SYSTEMONE_BACKEND` (`laya` | `reranker` | `auto`, default `auto`).

| Value | What loads |
| --- | --- |
| `laya` | [`laya` 0.3.5](https://pypi.org/project/laya/0.3.5/) — `laya.load("convaiinnovations/laya")` then `agent.predict(state, questions)`. Answers are `result["answers"][id]["noul"]`. All questions in a call are one forward pass (chunked by `ADMIT_BATCH_SIZE`). |
| `reranker` | `sentence-transformers` `CrossEncoder("BAAI/bge-reranker-base")`. `predict(pairs, batch_size=..., activation_fn=Identity)` → logits → `sigmoid(logit / T)`. |
| `auto` | Try Laya; on failure, load the reranker. `/healthz` and the response `backend` field report whichever actually loaded. |

Verified Laya API (PyPI `laya==0.3.5`, HF `convaiinnovations/laya`, GitHub `NandhaKishorM/laya`): `load(model_id, device=..., subfolder=...)` → `Agent.predict` / `system_one`.

## Run standalone

```bash
cd services/systemone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# first request / startup downloads Hugging Face weights into $HF_HOME
export USE_TF=0
uvicorn app:app --host 0.0.0.0 --port 8082
```

`GET /healthz` returns **200** only after the model is loaded **and** one warmup inference has succeeded.

### Docker

```bash
docker build -t recall-systemone .
docker run --rm -p 8082:8082 -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  recall-systemone
```

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYSTEMONE_BACKEND` | `auto` | `laya`, `reranker`, or `auto` |
| `ADMIT_BATCH_SIZE` | `32` | Candidates per model forward / CrossEncoder `batch_size` |
| `ADMIT_TEMPERATURE` | `1.0` | `T` in `noul = sigmoid(logit / T)` (overrides file value if set) |
| `LAYA_MODEL` | `convaiinnovations/laya` | Hugging Face repo for Laya |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | CrossEncoder checkpoint |
| `HF_HOME` | (unset) | Weight cache; Dockerfile sets `/root/.cache/huggingface` |
| `USE_TF` | `0` in the image | Avoid transformers/TensorFlow import deadlock (Laya docs) |

Optional `calibration.json` next to `app.py`: `{"temperature": ..., "fitted": true, "note": "..."}`. See [CALIBRATION.md](CALIBRATION.md). Until that file is a real fit, `calibration.fitted` is `false` and thresholds are not trustworthy.

## Curl example

```bash
curl -sS http://127.0.0.1:8082/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "recall-systemone",
    "state": {
      "question": "When is my sister'\''s birthday?",
      "as_of": "2026-09-21T18:00:00Z",
      "candidates": [
        {"id": "m0", "origin": "personal", "grantor": null, "text": "My sister Ana'\''s birthday is March 14."},
        {"id": "m1", "origin": "personal", "grantor": null, "text": "I bought Ana a gift card last year."}
      ]
    },
    "questions": {
      "m0": {"type": "noul", "instructions": "This candidate is sufficient evidence to answer the question. It is not merely on a related topic."},
      "m1": {"type": "noul", "instructions": "This candidate is sufficient evidence to answer the question. It is not merely on a related topic."}
    }
  }'
```

Example response:

```json
{
  "model": "recall-systemone",
  "answers": {
    "m0": {"noul": 0.94},
    "m1": {"noul": 0.06}
  },
  "backend": "laya",
  "calibration": {"temperature": 1.0, "fitted": false}
}
```

Rules: every `questions` key must be a `state.candidates` `id` (else **400**). Only `type: "noul"` is accepted (else **400**). Every question key is present in `answers`.

## Tests (no weights)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi httpx pytest
python3 -m py_compile app.py test_app.py
pytest test_app.py -q
```
