# IDF-aware lexical ranking for the Postgres hybrid retriever (2026-10-22)

Question: what is the best way to get BM25-quality lexical ranking (plus phrase/proximity boosting) in
`src/retrieval/hybrid.rs`, given the store is `pgvector/pgvector:pg17` now and Amazon Aurora PostgreSQL later?

## Recommendation

**Compute BM25 in plain SQL over the existing `fts_prose` tsvector. Do not adopt a BM25 extension.**

- **Aurora rules out every BM25 extension.** None of pg_search (ParadeDB), pg_textsearch (Tiger Data) or
  vchord_bm25 (TensorChord) is on AWS's supported-extension lists for Aurora PostgreSQL 17.11 / 18.6 or RDS for
  PostgreSQL. RUM is not listed either ([Aurora list][aurora-ext], [RDS list][rds-ext]). `pg_tle`, which Aurora
  does list, only installs extensions written in trusted languages such as PL/pgSQL or JavaScript. It cannot load a
  C or Rust index access method ([pg_tle README][pgtle]). The only documented way to use ParadeDB next to RDS or
  Aurora is a self-managed ParadeDB logical-replication subscriber ([ParadeDB docs][pdb-logrep]), which is a second
  database to run, secure and keep in sync.
- **Plain-SQL BM25 fixes q50, and the cost is negligible at this corpus size.** On the local database (1,386 current
  chunks), BM25 (Lucene's IDF, k1=1.2, b=0.75) ranks `markdown:canon/RETIRED.md#tenant-setup-modules-and-systems`
  **1st of 1,220 candidates for q50, up from 165th** with today's `ts_rank`. Across all 58 golden questions,
  full-text-only recall@50 goes from **0.784 to 0.862** and recall@10 from **0.552 to 0.724**. The query takes about
  80 ms with the statistics precomputed; recomputing the document frequencies with `ts_stat` takes about 34 ms
  (numbers under "Prototype").
- **Shape:** keep the current candidate filter (`fts_prose @@ OR-query OR fts_tokens @@ identifiers`) and keep
  identifier matches first. Replace `ts_rank(...)` as the second sort key with
  `Σ idf(t)·tf·(k1+1)/(tf + k1·(1−b+b·len/avglen))`, where:
  - `tf` = `array_length(positions)` from `unnest(fts_prose)`;
  - `idf` = `ln(1 + (N − df + 0.5)/(df + 0.5))`, with `df` from `ts_stat` ([PG docs][pg-tsstat]);
  - `len` = the summed position counts of the chunk.
  Two ways to supply `df` and `len`:
  1. *No migration:* compute `ts_stat('SELECT fts_prose FROM chunks WHERE superseded_at IS NULL')` per query. This is
     O(corpus), and was 34 ms here.
  2. *Scales:* a new migration adding a `chunk_terms(lexeme, ndoc)` table plus an `fts_len` column, rebuilt in the
     ingest transaction. This is a contract change (`migrations/`, `src/store`).

  Start with 1. Move to 2 if the corpus grows by about two orders of magnitude. Both work unchanged on Docker
  pgvector and on Aurora, because they use only core text search.
- **Leave phrase boosts and heading weighting out for now.** In the prototype, neither a capitalised-phrase boost
  (`phraseto_tsquery`, 19 of 58 questions have such a phrase) nor counting heading (weight A) matches twice changed
  recall@50 over plain BM25. Heading weighting cost one question (0.862 → 0.853); the phrase boost gained 0.017 at
  recall@10. The phrase tooling is all core Postgres (`phraseto_tsquery`, `<->`), so it can be added later without an
  extension.
- **Keep RRF with k=60.** This is the paper's value and the default in Elastic, OpenSearch, Vespa and ParadeDB's docs
  (details below). Revisit weights only after BM25 lands, with `drew eval`.
- **Caveat for the RLS phase.** Corpus statistics computed over all rows leak term frequencies of rows a user cannot
  read. pg_textsearch documents this for its own index, and so does Elastic ([pg_textsearch README][pgts]). If
  `ts_stat` runs inside the user's RLS session, `df` is computed over only the visible rows. That avoids the leak,
  but scores then differ per user. A stats table bypasses RLS unless it gets its own policy. Decide this in the
  security phase.

If Aurora were not the target, the best extension would be **pg_textsearch**: Postgres license, it reuses Postgres
text-search configurations (so `english` stays), and it is generally available on Google Cloud SQL, in preview on
AlloyDB and Azure HorizonDB, and in Timescale's HA image. Its ranking is bag-of-words: phrases work only as a
Boolean filter.

## Options compared

| Option | License | Index | Phrase / proximity | On Aurora / RDS | In a pgvector Docker image |
|---|---|---|---|---|---|
| Plain-SQL BM25 over `tsvector` | core Postgres | existing GIN | `phraseto_tsquery`, `<N>` (core) | yes | yes (current image) |
| pg_textsearch v1.5.1 (Tiger Data) | PostgreSQL | `USING bm25`, `<@>` | Boolean filter only, rechecked on heap; ranking ignores positions | no | `timescale/timescaledb-ha` (pg17+) |
| pg_search v0.26.2 (ParadeDB) | AGPL-3.0 (Enterprise commercial) | `USING paradedb` (Tantivy) | `###` phrase with slop, `##` proximity | no; logical-replica subscriber only | `paradedb/paradedb` (bundles pgvector) |
| VectorChord-bm25 0.3.0 | AGPLv3 or ELv2 | `bm25vector` + `USING bm25`, `<&>` | none (stores term frequencies, no positions) | no | `tensorchord/vchord-suite` |
| `ts_rank_cd` (cover density) | core | GIN | proximity-aware, no IDF | yes | yes |

### Maturity and details

- **pg_textsearch**
  - Repository created July 2025; v1.0 GA in April 2026 ([PG news][pgts-news]); v1.5.1 released 2026-10-02;
    about 4.0k stars (GitHub API, 2026-10-22).
  - Supports PostgreSQL 17 and 18 and needs `shared_preload_libraries`. Index options are `k1`/`b` (default 1.2/0.75),
    with partial and expression indexes. Top-k uses Block-Max WAND ([README][pgts]).
  - v1.0 stored no positions and had no phrase queries ([Tiger blog][pgts-blog]). v1.5.0 added "Boolean and phrase
    queries" through `@@ tsquery` with heap recheck. BM25 scoring plus a Boolean filter "is not yet optimized as a
    single index scan" ([README][pgts]).
  - Managed availability: Cloud SQL GA ([GCP][cloudsql-pgts]), AlloyDB ([GCP][alloydb-bm25]), Azure HorizonDB
    ([Microsoft][horizondb]). Azure Flexible Server: "not currently on our roadmap" ([Microsoft Q&A][azure-qa]).
  - Docker: Timescale's HA image installs it and adds it to `shared_preload_libraries` for pg17+
    ([Dockerfile][tsha-docker]).
- **pg_search (ParadeDB)**
  - Repository created 2023; v0.26.2 released 2026-10-16; about 9.4k stars.
  - Uses Tantivy through pgrx. Since 0.25.0 the access method is `paradedb`, with `bm25` kept as an alias
    ([docs][pdb-index]).
  - Phrase `###` with slop ([docs][pdb-phrase]); proximity `##` ([docs][pdb-prox]).
  - Community indexes are not physically replicated, so they exist only on the primary. HA and read replicas need
    Enterprise ([docs][pdb-ent]).
  - Neon retired pg_search on 2026-09-21 ([Neon][neon-pgsearch]).
  - The `paradedb/paradedb` image installs `postgresql-17-pgvector` ([Dockerfile][pdb-docker]).
- **VectorChord-bm25**
  - The repository now lives under `supervc-stack`; last release 0.3.0 (2025-12-15); about 380 stars.
  - `bm25vector` stores token ids and frequencies (no positions). The index returns at most `bm25_catalog.bm25_limit`
    rows. Tested "only against English" ([README][vcbm25]).
  - Tokenizers come from the separate `pg_tokenizer` extension. The image is `tensorchord/vchord-suite`, which also
    installs pgvector and vchord ([Dockerfile][vc-docker]).
- **RUM** (Postgres Pro): a GIN variant that stores positions so it can order by `<=>` and run phrase search faster.
  Its ranking is `ts_rank`-style, not BM25 ([README][rum]). Not on Aurora.

### Published benchmarks (vendor-run; treat as indicative)

- **Tiger vs ParadeDB v0.21.6** (Tiger's numbers) on MS-MARCO v2, 138M passages, warm cache, c6i.4xlarge:
  - pg_textsearch was 2.4–6.5× faster on 2–4-term queries and had 8.7× higher concurrent throughput.
  - ParadeDB built indexes about 2× faster.
  - The two were not checked for ranking equivalence ([Tiger blog][pgts-blog]).
- **VectorChord-bm25** nDCG@10 vs Elasticsearch: trec-covid 67.67 vs 68.80, webis-touche2020 31.0 vs 34.70. This table
  is commented out of the current README ([README source][vcbm25]).
- None of this matters at 1,386 chunks: plain SQL already answers in about 80 ms here.

## Plain SQL without an extension: evidence

- **`ts_rank` and `ts_rank_cd` have no IDF.** The Postgres docs say "the ranking functions do not use any global
  information" ([PG docs][pg-rank]). `ts_rank_cd` is cover density (Clarke, Cormack and Tudhope 1999): it rewards
  proximity but still has no IDF.
- **`ts_stat`** returns `word`, `ndoc` and `nentry` for each lexeme over any tsvector query ([PG docs][pg-tsstat]).
  That is exactly `df`.
- **Position limits:** at most 256 positions per lexeme, and positions are capped at 16,383 ([PG docs][pg-limits]).
  So `tf` from positions saturates for very long chunks. BM25's tf saturation (k1=1.2) makes this harmless for
  section chunks.
- **BM25 reference:** Lucene's `BM25Similarity` defaults to k1=1.2, b=0.75 and
  `idf = log(1 + (docCount − docFreq + 0.5)/(docFreq + 0.5))` ([Lucene source][lucene-bm25]). The prototype uses
  this formula.
- **Who does this in production:** I found no first-party write-up of plain-SQL BM25 in production.
  - Supabase's hybrid-search guide uses `ts_rank_cd` with `websearch_to_tsquery` (AND semantics, no IDF)
    ([Supabase][supabase]).
  - `plpgsql_bm25` keeps auxiliary tables and cannot track document changes ([source][plpgsql-bm25]).
  - ParadeDB's own explainer calls plain-SQL BM25 "convoluted and very slow" ([ParadeDB learn][pdb-learn]). That is a
    vendor claim about large corpora, and the measurement here contradicts it at this scale.

## Fusion in well-known hybrid systems

| System | Fusion | Default constant / weights |
|---|---|---|
| RRF paper (Cormack, Clarke, Büttcher, SIGIR 2009) | Σ 1/(k+r) | k=60, "near-optimal, but … not critical" ([paper][rrf]) |
| Elasticsearch `rrf` retriever | RRF, optional per-retriever `weight` | `rank_constant` 60; `rank_window_size` = `size` ([Elastic][es-rrf]) |
| OpenSearch | `score-ranker-processor` (RRF, 2.19+) or `normalization-processor` (min_max / l2 / z_score + mean, 2.10+) | RRF `rank_constant` 60, weights sum to 1 ([OpenSearch][os-rrf]); min_max default ([OpenSearch][os-norm]) |
| Vespa | global-phase `reciprocal_rank(x, k)` / `reciprocal_rank_fusion(a,b)`, or `normalize_linear` | k=60 ([Vespa][vespa-phased]) |
| Weaviate | `relativeScoreFusion` (default since 1.24) or `rankedFusion` | `alpha` 0.75 (vector-leaning) ([Weaviate][weaviate]) |
| Supabase guide | weighted RRF in SQL | `rrf_k` 50, weights 1/1 ([Supabase][supabase]) |
| ParadeDB docs | weighted RRF in SQL | k=60, example weights 1.0 text / 0.7 vector ([ParadeDB][pdb-rrf]) |
| Cloud SQL pg_textsearch guide | RRF in SQL | k=60 ([GCP][cloudsql-pgts]) |

Published fusion numbers:

- **OpenSearch on six BEIR sets:** RRF averaged 3.86% lower nDCG@10 than score normalization. For example, FiQA was
  0.2474 vs 0.2693 and Quora 0.796 vs 0.8452 ([OpenSearch blog][os-rrf-blog]). z-score was +2.08% over min-max on
  five sets ([OpenSearch blog][os-z]).
- **Vespa's NFCorpus tutorial:**

  | Profile | nDCG@10 |
  |---|---|
  | BM25 | 0.3210 |
  | dense | 0.3077 |
  | RRF | 0.3233 |
  | atan-normalised BM25 + cosine | 0.3410 |
  | linear-normalised | 0.3423 |

  ([Vespa tutorial][vespa-hybrid]).

Takeaway: RRF k=60 is the safe default everywhere. Score normalization can win by a few percent, but it needs score
distributions you can trust, and `ts_rank` scores are not that. With BM25 in place, a linear or z-score fusion is
worth one eval run later.

## Prototype (read-only queries; temp tables only, local DB, 2026-10-22)

Setup:

- Corpus: 1,386 current chunks. The q50 OR-query matches 1,220 of them.
- Lexemes in q50 by document frequency: `financi` 1, `impact` 16, `broken` 24, `find` 153, `say` 232, `use` 439,
  `custom` 501, `analysi` 503, `arm` 556, `run` 651.
- Reproduce the current ranking with `ts_rank(fts_prose, q, 1)`: the target ranks 165 / 1220, matching the earlier
  diagnosis.

q50 rank of `markdown:canon/RETIRED.md#tenant-setup-modules-and-systems`:

| Ranking | Rank |
|---|---|
| current `ts_rank` | 165 |
| `ts_rank_cd` (norm 1) | 112 |
| `websearch_to_tsquery` (AND) | not matched |
| **BM25** | **1** (score 17.13; next 13.04) |
| BM25 with heading tf doubled | 1 |
| BM25 + "Financial Impact Analysis" phrase bonus | 1 |

Full-text side only (no vector, no fusion), 58 golden questions, identifier matches kept first in every variant:

| Ranking | recall@50 | recall@10 | MRR |
|---|---|---|---|
| `ts_rank` (today) | 0.784 | 0.552 | 0.407 |
| `ts_rank_cd` | 0.621 | — | — |
| **BM25** | **0.862** | **0.724** | **0.493** |
| BM25, heading tf doubled | 0.853 | — | — |
| BM25 + capitalised-phrase bonus | 0.862 | — | — |
| BM25 + heading + phrase | 0.853 | 0.741 | 0.485 |
| BM25 with identifiers as a soft +3 per match | 0.871 | — | — |

- **Of the eight misses,** BM25 moves q50 (165 → 1) and q24 (73 → 4) into the top 50, lifts q51 (45 → 28; it was
  already inside the full-text top 50, so its miss is downstream of fusion) and q09 (193 → 69). q44 gains one of
  its two chunks (100 → 28). q06, q30 and q45 stay out.
- **Into the top 50:** ten relevant chunks move in or out of the top 50, eight in and two out:

  | Moved in | Moved out |
  |---|---|
  | q13, q24, q27 (272 → 1), q37, q43, q44, q50, q59 | q17 (7 → 63), q18 (30 → 101) |

- **Latency:** BM25 for q50 took 82 ms using indexed temp tables for `df` and chunk length, and `ts_stat` over the
  corpus took 34 ms (Docker on an arm64 Mac).
- **Reproduce:** run `python3 docs/research/search/proto/bm25_gen.py | docker compose exec -T db psql -U drew -d drew -q`
  from the repo root. `proto/bm25_eval.sql` holds the SQL.

Production-shaped query (the temp tables stand in for a stats table):

```sql
WITH q AS (SELECT tsvector_to_array(to_tsvector('english', $1)) AS terms),
k AS (SELECT count(*)::float n, avg(len) avgdl FROM doc_len),
idf AS (SELECT word, ln(1 + (k.n - ndoc + 0.5)/(ndoc + 0.5)) idf FROM term_df, q, k WHERE word = ANY(q.terms))
SELECT c.id, sum(idf.idf * u.tf * 2.2 / (u.tf + 1.2*(0.25 + 0.75*d.len/k.avgdl))) AS bm25
FROM chunks c JOIN doc_len d ON d.id = c.id CROSS JOIN q CROSS JOIN k
     CROSS JOIN LATERAL (SELECT lexeme, coalesce(array_length(positions,1),1) tf FROM unnest(c.fts_prose)) u
     JOIN idf ON idf.word = u.lexeme
WHERE c.superseded_at IS NULL AND c.fts_prose @@ array_to_string(q.terms, ' | ')::tsquery
GROUP BY c.id ORDER BY bm25 DESC LIMIT 50;
```

Caveats:

- 58 questions is a small set.
- These are full-text-side numbers. Fused recall@50 needs a `drew eval` run with Bedrock embeddings.
- `to_tsvector` drops stop words, so the query lexemes differ slightly from `prose_terms`. The production version
  should derive both from one tokenisation.

[aurora-ext]: https://docs.aws.amazon.com/AmazonRDS/latest/AuroraPostgreSQLReleaseNotes/AuroraPostgreSQL.Extensions.html
[rds-ext]: https://docs.aws.amazon.com/AmazonRDS/latest/PostgreSQLReleaseNotes/postgresql-extensions.html
[pgtle]: https://github.com/aws/pg_tle/blob/main/README.md
[pdb-logrep]: https://www.paradedb.com/docs/operate/deploy/logical-replication/getting-started
[pdb-index]: https://www.paradedb.com/docs/reference/indexing/create-index
[pdb-phrase]: https://www.paradedb.com/docs/reference/full-text/phrase
[pdb-prox]: https://www.paradedb.com/docs/reference/full-text/proximity
[pdb-ent]: https://www.paradedb.com/docs/operate/deploy/enterprise
[pdb-rrf]: https://www.paradedb.com/docs/reference/hybrid/rrf
[pdb-docker]: https://github.com/paradedb/paradedb/blob/main/docker/Dockerfile.paradedb-17
[pdb-learn]: https://www.paradedb.com/learn/search-in-postgresql/bm25
[pgts]: https://github.com/timescale/pg_textsearch/blob/main/README.md
[pgts-blog]: https://www.tigerdata.com/blog/pg-textsearch-bm25-full-text-search-postgres
[pgts-news]: https://www.postgresql.org/about/news/pg_textsearch-v10-3264/
[tsha-docker]: https://github.com/timescale/timescaledb-docker-ha/blob/master/Dockerfile
[cloudsql-pgts]: https://docs.cloud.google.com/sql/docs/postgres/pg-textsearch
[alloydb-bm25]: https://docs.cloud.google.com/alloydb/docs/ai/create-bm25-index
[horizondb]: https://learn.microsoft.com/en-us/azure/horizondb/ai/full-text-search
[azure-qa]: https://learn.microsoft.com/en-us/answers/questions/5760513/pg-textsearch-extension-support-on-azure-database
[neon-pgsearch]: https://neon.com/docs/extensions/pg_search
[vcbm25]: https://github.com/tensorchord/VectorChord-bm25/blob/main/README.md
[vc-docker]: https://github.com/tensorchord/VectorChord-images/blob/main/vchord-suite/Dockerfile
[rum]: https://github.com/postgrespro/rum
[pg-rank]: https://www.postgresql.org/docs/17/textsearch-controls.html
[pg-tsstat]: https://www.postgresql.org/docs/17/textsearch-features.html
[pg-limits]: https://www.postgresql.org/docs/17/textsearch-limitations.html
[lucene-bm25]: https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java
[plpgsql-bm25]: https://github.com/jankovicsandras/plpgsql_bm25
[supabase]: https://supabase.com/docs/guides/ai/hybrid-search
[rrf]: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
[es-rrf]: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever
[os-rrf]: https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/
[os-norm]: https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/
[os-rrf-blog]: https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
[os-z]: https://opensearch.org/blog/introducing-the-z-score-normalization-technique-for-hybrid-search/
[vespa-phased]: https://docs.vespa.ai/en/ranking/phased-ranking.html
[vespa-hybrid]: https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html
[weaviate]: https://docs.weaviate.io/weaviate/concepts/search/hybrid-search
