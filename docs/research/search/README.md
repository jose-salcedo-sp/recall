# Search plan (2026-10-22)

Evidence: [lexical.md](lexical.md) (BM25 in Postgres), [grep-agentic.md](grep-agentic.md) (grep and agentic search),
[enrichment.md](enrichment.md) (query- and document-side enrichment). Baseline: eval/results/daas-2026-10-21-v4.json
(recall@50 0.784, hit@5 0.707; 8 true misses q06 q09 q24 q30 q44 q45 q50 q51; q31 is a metric artifact).

[proto/](proto/) holds the prototype scripts the notes cite. Run them from the repo root against the local DB with the
golden file as the argument (`grep_probe.py` also reads `DREW_DOCS_DIR`). The result dumps the notes cite
(results*.json, rows3.jsonl, run*.log) are not committed; the scripts regenerate them.

## Phase 0: make retrieval cheap to measure (environment)
- `drew eval retrieval`: run Retrieve only (embedder + DB, no Kev/LLM) over the golden set. Report recall@10/@50
  and a per-question diff against a baseline file. Target: all 58 questions in seconds.
- Recall counts matches only. RetrieveRecord records `context_for`, so context chunks stop crowding the top 50
  (fixes q31). Contract change: pipeline/trace.rs (additive field).
- One corpus-scope type (authoritative/evidence/skip layers) shared by ingest and any future channel, plus a
  test that nothing under DaaS/_ops/ (holds the eval answers) is ever searchable.
- This is the gate for phases 1-3: a change must raise retrieval recall without per-question regressions,
  then the full `cargo eval-misses` and the full golden run confirm the downstream effect.

## Phase 1: BM25 in plain SQL (highest measured gain, no new deps)
- Replace ts_rank with BM25 over the existing fts_prose column. Document frequencies in a term-stats table
  refreshed at the end of ingest (ts_stat). No extension: none of pg_search, pg_textsearch or VectorChord-bm25
  is available on Aurora/RDS.
- Prototype (full-text only): q50 165th -> 1st; recall@50 0.784 -> 0.862; recall@10 0.552 -> 0.724;
  regressions q17, q18. ~80 ms. Keep RRF k=60. Phrase/heading boosts showed no gain; skip.

## Phase 2: glossary/lexicon query expansion (deterministic, no model)
- Expand question terms through canon/GLOSSARY.md and customer/LEXICON.md (parsed at ingest).
- Prototype (by hand): fixed 3 of the 4 misses that have an entry (q06 17th, q24 8th, q44 11th).
- Done (`retrieval/glossary.rs`): rows parsed from the current glossary chunks on each retrieval, so no reingest and
  no second copy. A row is selected by a two-word part or alternative of its first cell, or an acronym (`NEW`,
  `SoD` from `Segregation-of-duties`) written in capitals in the question; naming the whole term selects nothing
  (that stopped `EAM` and `ErpMaestro` costing q58 and q26). At most 16 terms; the question plus terms feeds both
  BM25 and the embedding. `cargo eval-retrieval` against the BM25 baseline (recall@10 68.1%, @50 86.2%):

  | Variant | @10 | @50 | Gained | Lost |
  |---|---|---|---|---|
  | 1. any 2+ capital word, BM25 only | 66.4 | 87.1 | q44 | q26, q58 |
  | 2. same, terms as a separate BM25 list | 68.1 | 86.2 | q24 | q58 |
  | 3. capitals-majority acronyms, BM25 only, terms weigh 0.5 | 66.4 | 87.1 | q44 (one of two) | q58 |
  | 4. 3 at weight 1, whole term selects nothing | 68.1 | 87.9 | q44 | — |
  | 5. 4, also the embedded query (kept) | 70.7 | 87.9 | q24, q44 | — |
  | 6. 4, plus separate BM25 and vector lists of the expansion | 69.8 | 87.9 | q24, q44 | — |

  q06 still misses: its row adds "segregation duties two conflicting business functions", too common to lift
  `HIT-LOGIC.md#when-a-risk-becomes-a-hit`.

## Phase 3: grep channel as a third RRF list (the user's idea; keep only if it adds recall over phase 1)
- ripgrep over the ingested layers only (phase 0 scope type); 3 rarest terms + capitalised phrases; map
  file+line to the heading-section chunk; keep only current chunks so citations stay within the run.
- Prototype (grep alone): 6 of 9 targets found, 5 in top 50, q50 1st, q09 3rd. Overlaps BM25's strength,
  so measure its marginal gain after phase 1.

## Later, model-dependent (each needs its own measurement; some need reingest)
- Cross-encoder (bge-reranker-v2-m3) over a wider pool, measured with `proto/proto3.py` on the fused list
  without Kev, counting the golden questions (of 58) with a relevant chunk found:
  in the top 50: current 50, BM25 53, BM25 with 150 candidates and the cross-encoder 56;
  in the top 5: current 37, BM25 43, cross-encoder 45 (loses q13, q17, q42, q59, q60).
  About 9.2 s per question on a local machine (see enrichment.md section 4), far above Kev; only worth revisiting on a GPU or hosted.
- Contextual prefixes per chunk (Anthropic Contextual Retrieval): better vector rank for 5 of 8 misses, worse
  for 3. Reingest.
- One chunk per table row: q50 vector rank 322 -> 127. Reingest.
- LLM query rewrite as an extra list: 51/58 vs 50/58, one qwen call per question.

## Rejected
- HyDE (no gain), multi-query (49/58, worse), BM25 extensions (not on Aurora), LLM-driven search loop
  (latency, nondeterminism, citation scope; coding agents' gains came from lexical search, which phases 1-3 cover).
