#!/usr/bin/env python3
"""Compare System One question shapes for admission, on ranking quality and cost.

The current implementation sends one question per call with one candidate in the
state, which discriminates correctly but costs one forward pass per candidate.
The protocol expects several questions batched against one shared state, so the
question is whether a batched shape can discriminate as well.

Run inside the systemone container so it talks to the loaded model directly:
  podman exec -i recall-systemone-1 python - < scripts/probe_question_shapes.py
"""
import json
import time

import laya

QUESTION = "When is my sister Ana's birthday?"
ANSWER = "My sister Ana's birthday is March 14."
CANDIDATES = [
    ANSWER,
    "Maya's birthday is December 5.",
    "My sister Ana lives in Denver on York Street.",
    "Luis's wedding anniversary is June 8.",
    "My mom's name is Carmen Salcedo.",
    "The wifi password at the cabin is redmaple88.",
    "I'm allergic to peanuts and carry an EpiPen.",
    "GitHub username is jsalcedo.",
]
SUFFICIENT = (
    "This candidate is sufficient evidence to answer the question. "
    "It is not merely on a related topic."
)
RUBRIC = [
    "unrelated to the question",
    "same topic but does not contain the answer",
    "contains the answer to the question",
]

agent = laya.load("convaiinnovations/laya", device="cpu")


def report(name, scores, elapsed, calls):
    """Rank of the true answer is the only quality measure that matters here."""
    order = sorted(range(len(CANDIDATES)), key=lambda i: -scores[i])
    rank = order.index(0) + 1
    gap = scores[0] - max(s for i, s in enumerate(scores) if i != 0)
    print(f"\n{name}")
    print(f"  answer rank {rank}/{len(CANDIDATES)}   gap {gap:+.4f}   "
          f"{elapsed:.2f}s   {calls} call(s)")
    for i in order[:4]:
        mark = "*" if i == 0 else " "
        print(f"   {mark} {scores[i]:.4f}  {CANDIDATES[i][:52]}")


# A: current shape — one state per candidate, one noul each.
t = time.time()
scores = []
for c in CANDIDATES:
    r = agent.predict({"question": QUESTION, "candidate": c},
                      {"q": {"type": "noul", "instructions": SUFFICIENT}})
    scores.append(r["answers"]["q"]["noul"])
report("A. per-candidate state, noul  (current)", scores, time.time() - t, len(CANDIDATES))

# B. shared state listing numbered candidates; one noul per candidate that
#    references its number rather than repeating its text.
state = {"question": QUESTION,
         "candidates": {f"memory_{i}": c for i, c in enumerate(CANDIDATES)}}
qs = {f"m{i}": {"type": "noul",
                "instructions": f"Considering only memory_{i}: {SUFFICIENT}"}
      for i in range(len(CANDIDATES))}
t = time.time()
r = agent.predict(state, qs)
report("B. shared state, noul by index (batched)",
       [r["answers"][f"m{i}"]["noul"] for i in range(len(CANDIDATES))],
       time.time() - t, 1)

# C. per-candidate state, score against an ordered rubric. Score also returns a
#    confidence value, which noul does not — relevant to the calibration problem.
t = time.time()
scores, confs = [], []
for c in CANDIDATES:
    r = agent.predict(
        {"question": QUESTION, "candidate": c},
        {"q": {"type": "score",
               "instructions": "How well does this candidate answer the question?",
               "criteria": RUBRIC}},
    )
    a = r["answers"]["q"]
    scores.append(a["score"] / (len(RUBRIC) - 1))
    confs.append(a.get("confidence", float("nan")))
report("C. per-candidate state, score rubric", scores, time.time() - t, len(CANDIDATES))
print(f"   confidence: answer {confs[0]:.3f}  mean others "
      f"{sum(confs[1:]) / (len(confs) - 1):.3f}")

# D. shared state, one choice over all candidates. architecture.md rejects this as
#    the only gate; measured here so the rejection rests on evidence.
t = time.time()
r = agent.predict(
    state,
    {"best": {"type": "choice",
              "instructions": "Which memory answers the question?",
              "criteria": {f"memory_{i}": c for i, c in enumerate(CANDIDATES)}}},
)
a = r["answers"]["best"]
probs = a.get("probabilities", {})
print("\nD. shared state, single choice over all candidates")
print(f"  picked {a.get('choice')}   confidence {a.get('confidence', 0):.3f}   "
      f"{time.time() - t:.2f}s   1 call")
print(f"  top probabilities: "
      f"{json.dumps(dict(sorted(probs.items(), key=lambda kv: -kv[1])[:3]))}")
