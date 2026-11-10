"""All 58 questions, deterministic variants (no LLM):
- current full text (hybrid.rs SQL, identifiers first, then ts_rank) vs BM25 over fts_prose (identifiers still first);
- bge-reranker-v2-m3 cross-encoder over the top 150 fused candidates, keeping 50.
recall@50 = any gold chunk in the fused top 50 (before neighbour expansion); hit@5 likewise."""
import json, re, sys, time
import psycopg
from proto import DB, MISSES, embed, rank, rrf

CURRENT = """SELECT c.id FROM chunks c,
  (SELECT to_tsquery('english', %(prose)s) AS prose, %(tokens)s::tsquery AS tokens) q
  WHERE c.superseded_at IS NULL AND (c.fts_prose @@ q.prose OR c.fts_tokens @@ q.tokens)
  ORDER BY length(c.fts_tokens) - length(ts_delete(c.fts_tokens, %(ids)s::text[])) DESC,
           coalesce(ts_rank(c.fts_prose, q.prose, 1), 0) DESC, c.id LIMIT %(n)s"""

BM25 = """WITH q AS (SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %(text)s))),
docs AS (SELECT c.id, u.lexeme, coalesce(array_length(u.positions, 1), 1) AS tf
         FROM chunks c, unnest(c.fts_prose) u WHERE c.superseded_at IS NULL),
dl AS (SELECT id, sum(tf) AS len FROM docs GROUP BY id),
stats AS (SELECT count(*) AS n, avg(len) AS avgdl FROM dl),
df AS (SELECT lexeme, count(*) AS df FROM docs WHERE lexeme IN (SELECT lexeme FROM q) GROUP BY lexeme),
bm AS (SELECT d.id, sum(ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
                 * d.tf * 2.2 / (d.tf + 1.2 * (0.25 + 0.75 * dl.len / s.avgdl))) AS score
       FROM docs d JOIN df USING (lexeme) JOIN dl USING (id) CROSS JOIN stats s GROUP BY d.id)
SELECT c.id FROM chunks c LEFT JOIN bm USING (id)
WHERE c.superseded_at IS NULL AND (bm.score IS NOT NULL OR c.fts_tokens @@ %(tokens)s::tsquery)
ORDER BY length(c.fts_tokens) - length(ts_delete(c.fts_tokens, %(ids)s::text[])) DESC,
         coalesce(bm.score, 0) DESC, c.id LIMIT %(n)s"""


def params(q, n):
    prose = [w for w in re.split(r"[^\w]+", q) if w and w.isalnum()]
    ids = sorted({t.strip("_").lower() for t in re.split(r"[^A-Za-z0-9_]+", q)
                  if t.strip("_") and re.search(r"[A-Za-z]", t) and re.search(r"[_0-9]", t.strip("_"))})
    return {"prose": " | ".join(prose) or None, "text": q,
            "tokens": " | ".join(f"'{t}'" for t in ids) or None, "ids": ids, "n": n}


def main():
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="mps")
    golden = [json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2]) if len(sys.argv) > 2 else 0:]
    totals, rows = {}, []
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET enable_indexscan = off")
        cur.execute("""SELECT c.id, d.title || E'\n' || coalesce(c.heading, '') || E'\n\n' || c.text
                       FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.superseded_at IS NULL""")
        texts = dict(cur.fetchall())
        for g in golden:
            q, gold = g["question"], g["relevant_chunks"]
            (eq,) = embed([q])
            cur.execute("SELECT id FROM chunks WHERE superseded_at IS NULL ORDER BY embedding <=> %s::vector, id LIMIT 150", (str(eq),))
            vec = [r[0] for r in cur.fetchall()]
            cur.execute(CURRENT, params(q, 150)); fts = [r[0] for r in cur.fetchall()]
            cur.execute(BM25, params(q, 150)); bm = [r[0] for r in cur.fetchall()]
            base = rrf([vec, fts], top=50)[:50]
            bm25 = rrf([vec, bm], top=50)[:50]
            t0 = time.time()
            cands = rrf([vec, bm], top=150)[:150]
            scores = ce.predict([(q, texts[c]) for c in cands], batch_size=16)
            reranked = [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]
            dt = time.time() - t0
            res = {
                "base_r50": rank(base, gold) is not None, "bm25_r50": rank(bm25, gold) is not None,
                "bm25_150_rerank_r50": (rank(reranked, gold) or 999) <= 50,
                "base_hit5": (rank(base, gold) or 999) <= 5, "bm25_hit5": (rank(bm25, gold) or 999) <= 5,
                "rerank_hit5": (rank(reranked, gold) or 999) <= 5,
            }
            for k, v in res.items():
                totals[k] = totals.get(k, 0) + v
            row = {"id": g["id"], "rank_base": rank(base, gold), "rank_bm25_fts": rank(bm, gold),
                   "rank_cur_fts": rank(fts, gold), "rank_rerank": rank(reranked, gold),
                   "rerank_seconds": round(dt, 2), **res}
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("TOTALS", totals, "of", len(golden))
    json.dump({"rows": rows, "totals": totals}, open("docs/research/search/proto/results3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
