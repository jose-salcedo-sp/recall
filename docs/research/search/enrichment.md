# Query- and document-side enrichment for Drew retrieval (2026-10-22)

Question: which enrichments have measured gains for a small (1,386 current chunks, ~510k tokens) domain corpus
queried with colloquial, Slack-style questions, and what would each cost Drew with local models (bge-m3, qwen3:14b,
Kev)?

Baseline: v4 eval, recall@50 = 50/58; misses q06 q09 q24 q30 q44 q45 q50 q51
(`eval/results/daas-2026-10-21-v4.json`). The prototype reproduces 50/58 (results2.json,
`original`).

## Ranked recommendations

| # | Change | Prototype result (8 misses / all 58) | Ingest cost | Query cost | Deterministic |
|---|---|---|---|---|---|
| 1 | **IDF-weighted full text (BM25)** in place of `ts_rank`, fused with the vector list as today | **53/58** (+q24, +q50, +q51, no losses); fused hit@5 37 → 43 (proto3) | None, or a stats table | One SQL query | Yes |
| 2 | **Cross-encoder (bge-reranker-v2-m3) over the top 150 fused (BM25 + vector), cut to 50, before Kev** | **56/58** (+q06, q24, q44, q45, q50, q51, no losses); fused hit@5 37 → 45 (+13 −5). Only q09, q30 still missed | None | 150 pairs through a 0.6B model: median 8.9 s/question measured (fp32, MPS, batch 16, max 512 tokens) | Yes |
| 3 | **Glossary/lexicon query expansion** from `canon/GLOSSARY.md` and `customer/LEXICON.md`, plus a short acronym list | Fixed 3 of the 4 misses that have an entry (q06, q24, q44); q09 not. Hand-applied, not run on all 58 | Parse 2 files at ingest | One lookup, no model | Yes |
| 4 | **LLM query rewrite added as an extra RRF list** (original lists kept) | 51/58: gains q24, q51; loses q26 (not re-measured on top of BM25) | None | 1 qwen3:14b call (can run beside `Classify`) | Only with temp 0 + seed, and the rewrite stored in the trace |
| 5 | **Contextual Retrieval** (LLM context prefix on each chunk, embedded and in full text) | Vector rank better for 5/8 misses (q09 347→46), worse for 3/8 | ~1,400 qwen3:14b calls, each with the whole document in the prompt; cache by chunk version | None | Yes after ingest (stored) |
| 6 | **Table-row chunks** for register tables (`RETIRED.md` etc.) | q50 vector rank 322→127 (one row embedded) | More chunks; no model | None | Yes |
| — | Not recommended now: HyDE, multi-query RAG-Fusion, step-back, late chunking, doc2query, bge-m3 sparse/ColBERT | HyDE 50/58 (±0); multi-query 49/58 (+2 −3) | | | |

Heading-path prefixes are **already done**: `embedding_input` in `src/store/chunks.rs` embeds
`title \n heading path \n\n text`, and `fts_prose` weights the heading A. Not a lever.

Order: #1 is deterministic, needs no model, and has no losses on all 58. #2 adds the most recall but is a new model
service and ~9 s/question as measured. #3 covers the vocabulary gaps neither fixes, but its gain is only measured
by hand on 4 questions. The section numbers below predate this ranking: §1 is the glossary (rank 3), §2 BM25
(rank 1), §3 rewrite (rank 4), §4 cross-encoder (rank 2).

### 1. Glossary/lexicon expansion

The misses are vocabulary gaps that the corpus's own registers already cover:

| Question | Query words | Register row that bridges it | Target chunk |
|---|---|---|---|
| q24 | "known issue we're tracking" | LEXICON: `BUG-nn` → "an issue we're tracking — see the disclosure rules"; *Phrases to avoid*: "Known issue" → "an issue we're aware of and are looking into" | `DISCLOSURE-TIERS.md#decisions-on-record` (quotes that phrase) |
| q44 | "NEW screen" | LEXICON: "Legacy / NEW generation" ↔ "the current screens and the earlier screens" | `change-impact.md#3-which-generations-does-it-affect` |
| q06 | "SoD" | GLOSSARY *Rulebook vocabulary*: "Segregation-of-duties risk: two conflicting business functions" (needs an `SoD` acronym entry; the glossary never spells "SoD") | `HIT-LOGIC.md#when-a-risk-becomes-a-hit` |
| q09 | "report-embed layer" | GLOSSARY: "RMS … Owns the report-embed / QIX request path" | `REPORTS.md#known-defects` (still missed: FTS 193→85) |
| q30, q45, q50, q51 | "sandbox", "too big / out of memory", "can't find it", "anyone else working" | No entry | — |

Measured (results.json, hand-applied rows above; fused rank, ≤50 is a hit): q06 >50 → 17, q24 >50 → 8,
q44 >50 → 11, q09 still >50.

Code: a deterministic expander (new `retrieval/expand.rs`) that reads the two tables at ingest (they are already
chunks), maps "Say instead"/"Word" phrases to the internal term, and adds the expansion as a **separate** full-text
and vector list in RRF, so the original question's lists are unchanged. Record the matched rows in the `Retrieve`
trace record. Caveat: the prototype matched rows by hand; an automatic phrase matcher needs its own regression run on
all 58, and the registers need a few colloquial entries (SoD, sandbox → demo file, "can't find" → retired). This is
also what Anthropic suggests for the contextualizer prompt: "including a glossary of key terms that might only be
defined in other documents" ([Anthropic][cr]).

### 2. BM25 / IDF full text

Postgres's ranking functions "do not use any global information" ([PostgreSQL docs][pgts]), so `ts_rank` has no
IDF; in q50 the only chunk containing `financi` (df = 1) ranked 165/1,220. Anthropic's first conclusion is that
"Embeddings+BM25 is better than embeddings on their own" ([Anthropic][cr]). Covered in detail in `lexical.md`;
change is the ORDER BY in `HybridRetriever::full_text_candidates`.

Measured here, fused (proto3.py, rows3.jsonl): BM25 (k1 = 1.2, b = 0.75, over `fts_prose` via `unnest(tsvector)`,
identifier matches still ordered first exactly as in `hybrid.rs`) fused with the unchanged vector list gives
recall@50 53/58 against 50/58 for the faithful copy of the current SQL, gaining q24, q50, q51 and losing none.
q50's chunk goes from outside the full-text top 150 to 1st. Fused hit@5 rises 37 → 43 (+8 −2: loses q17, q35).
`lexical.md` saw q17 and q18 regress on full text alone; fused, neither loses its recall@50 hit.

### 3. LLM rewrite as an extra list

Neutral prompt, all 58 (results2.json): rewrite *alone* drops to 42/58 (loses 10); rewrite **added** to the
original lists gives 51/58 (+q24, +q51, −q26). A first run whose prompt listed corpus topics fixed q50 and q44 as
well (results.json), but only because the prompt leaked "tenant setup" (q50's heading) and "legacy vs current
generations" — evidence that a corpus-aware prompt (e.g. the glossary's section names) helps, not that the model
understood the question. Published: Rewrite-Retrieve-Read reports gains with a trained rewriter
([Ma et al. 2023][rrr]); in Wang et al.'s RAG best-practices study, LLM query rewriting *lowered* TREC DL19
nDCG@10 from 70.20 to 67.89 (LLM-Embedder retriever, R@50 49.06 → 51.45) ([Wang et al. 2024][bp], Table 6), which
matches what we see: keep the original query.

Code: `Retrieve` gets the rewrite; `Retriever::retrieve(&str)` would need an additive multi-query method (contract
change in `src/retrieval/mod.rs`), and the raw rewrite goes in the trace (design §5.6). Latency: one qwen3:14b
call (seconds locally), parallel with `Classify`.

### 4. Cross-encoder before Kev

Measured (proto3.py, rows3.jsonl; the first run died after 43 questions and was resumed from q45, run3.log and
run3b.log): candidates are the top 150 of RRF(vector top 150, BM25 top 150); bge-reranker-v2-m3 scores each
(question, `title \n heading \n\n text`) pair at max length 512, and the top 50 by score are kept. Recall@50:
**56/58** (BM25-only 53, current 50), no losses. Miss ranks after reranking: q50 1st, q51 2nd, q24 4th, q45 14th,
q44 25th, q06 49th; q09 and q30 are not in the 150-candidate pool at all, so no reranker can reach them. Hit@5 of the
reranked order is 45/58, against 37 for the current fused order and 43 for BM25 fused (+13 −5: loses q13, q17,
q42, q59, q60). That is a proxy for the `Rerank` step, not Kev's hit@5 (v4: 0.707, with neighbour context and
Kev). Latency: median 8.9 s per question for 150 pairs, fp32 on MPS through sentence-transformers. fp16, a 100-pair
pool, or a persistent server would cut that; not measured.

Reason it works: 6 of the 8 misses sit at rank 45–125 in at
least one list (q06 vec 80, q24 73/78, q30 vec 121, q44 96/100, q45 109/111, q51 45/61), so recall@150 is higher
than recall@50, and a reranker is what turns a wider pool into a better top 50. Anthropic retrieved 150 and
reranked to 20 (Cohere Rerank), going from a 2.9% to 1.9% top-20 failure rate ([Anthropic][cr]). BGE recommends
"hybrid retrieval + re-ranking" ([BGE-M3 README][m3readme]).

Why before Kev rather than instead of Kev's prompt: design §7 milestone 5 records that ordering by Kev's
probability alone scored hit@5 36% vs 53% for retrieval order, because the probabilities sit close together.
Zhuang et al. found yes/no relevance prompting (RG-YN) averages 0.4770 nDCG@10 on 8 BEIR sets against 0.4992
for 3-level labels and 0.5022 for a 0–4 scale (FLAN PaLM2 S reranking BM25 top 100) ([Zhuang et al. 2023][yn],
Table 2), so a binary prompt is the weakest pointwise LLM format. Supervised rerankers are strong for their size:
monoT5 (220M) reranking BM25 top-1000 on MS MARCO dev gives MRR@10 31.78 vs RankLLaMA-7B 32.35, at 4.5 s vs
82.4 s per query ([Wang et al. 2024][bp], Table 9).

Expected cost: bge-reranker-v2-m3 is 0.6B params, max length 512 in the model card example
([model card][bgerr]). Bedrock alternatives: Cohere Rerank 3.5 (`cohere.rerank-v3-5:0`) or Amazon Rerank 1.0
([AWS docs][bedrockrr]). Deterministic. Code: a `Reranker` client trait (`#[async_trait]`, fake + wiremock), a
pre-rank step between `Retrieve` and `Rerank`, `HybridConfig.fused_limit` raised to ~150 with the cross-encoder
cutting to 50; recall@50 would then be measured after the cross-encoder. The model card publishes its benchmark
numbers only as images, so no quotable figure for this exact model.

### 5. Contextual Retrieval

Published ([Anthropic][cr]): prepend 50–100 tokens of LLM-written, chunk-specific context before embedding and
before BM25 indexing. Averaged over codebases, fiction, ArXiv and science papers with Gemini Text 004 embeddings,
top-20 retrieval failure (1 − recall@20) fell 5.7% → 3.7% with contextual embeddings (−35%), → 2.9% with
contextual BM25 too (−49%), → 1.9% with reranking (−67%). They also report "adding generic document summaries to
chunks (we experimented and saw very limited gains)" and that HyDE and summary-based indexing performed poorly.

Prototype (results2.json, `document_side`; Anthropic's prompt verbatim, qwen3:14b, whole document in context;
only the gold chunk re-embedded, so ranks are optimistic for gold and ignore other chunks gaining context;
vector side only, no contextual BM25): q09 347→46, q30 121→72, q50 322→225, q24 78→60, q06 80→72; worse:
q44 96→137, q45 109→132, q51 61→75. Mixed, and the biggest win (q09) came from the context naming
"Qlik … reporting", which the glossary expansion could also supply at query time.

Cost: ~1,400 qwen3:14b calls with up to 8k-token documents (hours locally); the context changes whenever any
section of its document changes, so a document edit re-contextualizes and re-embeds all its chunks. Stored once,
so retrieval stays deterministic. Code: a nullable `context` column (new migration; contract), included in
`embedding_input` and in new `fts_*` generated columns (generated columns cannot be altered in place, so a new
migration recreates them), a contextualizer at ingest keyed by (chunk version, document version, prompt version,
model). Ingested text must be delimited in that prompt (AGENTS.md).

### 6. Table-row chunks

q50's answer is one row of a five-row table. Embedding that row alone with its breadcrumb and header row moves its
vector rank from 322 to 127 (results2.json): better, not enough by itself; with BM25 length normalization a short
row chunk would also rank higher on full text. No primary source measures row-level chunking for RAG retrieval;
Table Meets LLM shows LLM table understanding is sensitive to serialization format (GPT-3.5/4)
([Sui et al. 2024][tables]) but measures QA, not retrieval. Code: `src/ingest/markdown.rs` emits one extra chunk per
row for pipe tables under a heading (row + header + heading path), keeping the section chunk; neighbour expansion
already brings back the section.

## Evidence for the techniques not recommended now

**HyDE.** Original paper: with unsupervised Contriever, TREC DL19 nDCG@10 44.5 → 61.3 (InstructGPT); BM25 50.6
([Gao et al. 2022][hyde], Table 1). With a *fine-tuned* encoder the gain shrinks or reverses: Contriever-FT 62.1
→ 60.2 with Flan-T5-11B, 61.4 with Cohere-52B, 67.4 with GPT-175B (Table 4). bge-m3 is a fine-tuned encoder and
qwen3:14b is near the Flan-T5 scale, so little gain is expected. Prototype: 50/58, no change; on misses it helped
q44 and q50 on the vector side but hallucinated ("FIA … has been rebranded"). Wang et al. found HyDE + hybrid the
best retrieval setting on DL19 (R@50 55.38 vs 49.06) with LLM-Embedder ([Wang et al. 2024][bp], Table 6); that is
web-domain text the generator knows, unlike DaaS internals.

**Multi-query + RRF (RAG-Fusion).** The only paper is a qualitative Infineon case study with manual answer grading
and no retrieval metric; it notes answers "strayed off topic when the generated queries' relevance … is
insufficient" ([Rackauckas 2024][fusion]). Prototype: 49/58 (+q50, +q51, −q16, −q31, −q40).

**Step-back prompting.** Measured on reasoning QA, not retrieval (MMLU Physics +7%, TimeQA +27% for PaLM-2L)
([Zheng et al. 2023][stepback]). Prototype on misses: hit q44 and q51 only (results.json).

**Query2doc / LLM expansion for BM25.** Large BM25 gains on MS MARCO (DL19 nDCG@10 51.2 → 66.2 with
text-davinci-003 pseudo-docs), small for strong dense retrievers (E5-base+KD DL19 74.3 → 74.9)
([Wang et al. 2023][q2d], Table 1). Drew's lexical side would gain, but generated text invents identifiers, which
the identifier-exact full-text side would then match.

**Late chunking.** Averaged over 3 Jina/Nomic models and 4 BeIR sets, +1.5 to +1.9 nDCG@10 points over naive
chunking ([Günther et al. 2024][late]); gains concentrate on small fixed-size chunks, and for some long-document
tasks naive chunking with large chunks did better. Drew's chunks are whole heading sections (mean ~370 tokens)
already prefixed with their heading path, and late chunking needs token-level embeddings, which Ollama's
`/api/embed` does not return ([Ollama API][ollama]). Skip.

**doc2query / generated questions per chunk.** docT5query lifts BM25 MS MARCO dev MRR@10 18.4 → 27.7
([Wang et al. 2023][q2d], Table 1, citing Nogueira & Lin; method in [Nogueira et al. 2019][d2q]), but it needs a
trained query generator, and Doc2Query-- shows generated queries hallucinate, filtering them improves effectiveness
by up to 16% and cuts index size 33% ([Gospodinov et al. 2023][d2qmm]). With qwen3:14b it costs the same ingest
time as Contextual Retrieval; try only if #5 shows gains, and generate Slack-style questions.

**Small-to-big / parent documents.** Drew already does this: hits pull in neighbouring sections
(`expand` in `src/retrieval/hybrid.rs`). Finer retrieval units can help: propositions beat passages by +12.0
Recall@5 averaged over five open-domain QA sets for unsupervised SimCSE ([Chen et al. 2023][densex]). The table-row
chunk (#6) is the cheap version of this for DaaS.

**bge-m3 sparse and multi-vector.** BGE-M3 paper, MIRACL dev nDCG@10 averaged over 18 languages: dense 69.2,
sparse 53.9, multi-vector 70.5, dense+sparse 70.4, all three 71.5; English: 56.9 / 43.8 / 59.3 / 58.8 / 59.6
([Chen et al. 2024][m3], Table 1). M3 sparse beats BM25 on MIRACL (53.9 vs 38.5 with a Lucene analyzer) but not on
long documents (MLDR 62.2 vs 64.1), and the authors note "BM25 remains a highly competitive baseline" (Appendix,
Table 11). Gains over dense are +1 to +2 points, multi-vector is used only to rerank the top 200, and Ollama serves
only the dense vector ([Ollama API][ollama]); sparse would need a Python FlagEmbedding sidecar and pgvector
`sparsevec` (HNSW indexes up to 1,000 non-zero elements) ([pgvector][pgv]). BM25 (#2) gets most of the lexical
benefit with no new service.

## Method and caveats

- Prototype scripts and outputs: `proto/proto.py` (8 misses, 7 variants; results.json, run.log) and
  `proto/proto2.py` (all 58 with a neutral prompt plus document-side tests; results2.json, run2.log), and
  `proto/proto3.py` (all 58, no LLM: faithful current full-text SQL vs BM25, and bge-reranker-v2-m3 over the top
  150; rows3.jsonl, run3.log + run3b.log).
  Read-only against the local DB (exact scan, `enable_indexscan = off`); Ollama bge-m3 and qwen3:14b at
  temperature 0, seed 7.
- recall@50 is "a gold chunk in the fused top 50 before neighbour expansion", RRF k = 60 over each list's top 50,
  matching `HybridConfig` defaults. Full text in the prototype is the prose side only (no identifier ordering);
  it still reproduces 50/58.
- One run per variant; qwen3 output can change with Ollama version or hardware. 58 questions: a ±1 question
  difference is noise.
- The glossary rows were applied by hand; Contextual Retrieval re-embedded only gold chunks.

[cr]: https://www.anthropic.com/news/contextual-retrieval
[pgts]: https://www.postgresql.org/docs/current/textsearch-controls.html
[m3]: https://arxiv.org/abs/2402.03216
[m3readme]: https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3
[hyde]: https://arxiv.org/abs/2212.10496
[bp]: https://arxiv.org/abs/2407.01219
[rrr]: https://arxiv.org/abs/2305.14283
[yn]: https://arxiv.org/abs/2310.14122
[bgerr]: https://huggingface.co/BAAI/bge-reranker-v2-m3
[bedrockrr]: https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-supported.html
[late]: https://arxiv.org/abs/2409.04701
[q2d]: https://arxiv.org/abs/2303.07678
[d2q]: https://arxiv.org/abs/1904.08375
[d2qmm]: https://arxiv.org/abs/2301.03266
[densex]: https://arxiv.org/abs/2312.06648
[fusion]: https://arxiv.org/abs/2402.03367
[stepback]: https://arxiv.org/abs/2310.06117
[tables]: https://arxiv.org/abs/2305.13062
[ollama]: https://github.com/ollama/ollama/blob/main/docs/api.md
[pgv]: https://github.com/pgvector/pgvector
