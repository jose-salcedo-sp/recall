# Calibrating System One nouls

Raw scores from both backends are **not** trustworthy probabilities.

- **Laya** (`convaiinnovations/laya`) publishes expected calibration error of **0.466 as shipped**. After fitting one temperature per (question type, option count) on held-out data, ECE drops to **0.081**. The SDK already divides logits by its own checkpoint temperatures; those are still the *shipped* values, not a fit on Recall’s admission labels.
- **BGE reranker** (`BAAI/bge-reranker-base`) emits unbounded relevance logits. They pile up near zero after a sigmoid and are not trained as P(this chunk is sufficient evidence).

This service always maps a logit to a noul with temperature scaling:

```
noul = sigmoid(logit / T)
```

`T` comes from `ADMIT_TEMPERATURE` (default `1.0`) or, if present, `calibration.json` in this directory:

```json
{"temperature": 1.7, "fitted": true, "note": "fit 2026-09-21 on holdout split"}
```

The JSON response includes `calibration.temperature` and `calibration.fitted`. **`fitted: false` means do not treat any admission threshold as calibrated.** Fit `T` on labeled data before using a cutoff in production.

## Procedure

1. **Label pairs.** Sample real `(question, candidate)` pairs from Recall retrieval. For each pair, a human (or a carefully reviewed teacher) marks `sufficient ∈ {0, 1}`: *would this chunk alone be enough evidence to answer the question?* Related-topic chunks are `0`.
2. **Hold out a split.** Fit on a training slice; never report ECE / precision-recall on the same rows used to choose `T`. Stratify by origin if personal vs granted memories behave differently.
3. **Fit T by minimizing NLL.** For each pair, take the backend logit `z` (for Laya: `logit(noul_raw)`). Predict `p = sigmoid(z / T)`. Minimize binary negative log-likelihood over `T > 0` (one-dimensional; a coarse grid then a scalar optimizer is enough). Write the result to `calibration.json` with `"fitted": true`.
4. **Report the precision/recall curve.** Sweep an admission threshold `τ` on the holdout set using the calibrated `noul`. Plot precision and recall vs `τ`, and pick the operating point the product actually wants (high precision if a false admit poisons generation). ECE after the fit is the check that `noul` matches empirical frequency — not a substitute for the PR curve.

Until that loop has been run on Recall-labeled pairs, keep `T = 1.0` and `fitted: false`, and do not ship an admission threshold as if it were a probability cutoff.
