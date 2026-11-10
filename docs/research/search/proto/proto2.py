"""All 58 questions: recall@50 (fused, before neighbour expansion) for query-side variants with a neutral prompt.
Then, for the 8 misses: document-side contextual prefix and q50 table-row chunk, ranked against unchanged chunks."""
import json, re, sys
import psycopg
from proto import DB, MISSES, embed, chat, vec_list, fts_list, rank, rrf

HINT = "The documentation is internal support documentation for SailPoint Access Risk Management (ARM) on SAP."


def rewrite(q):
    return chat(f"{HINT}\nRewrite this support question as one search query of 10-25 keywords in the vocabulary "
                f"the documentation would use (formal terms, synonyms). Output only the query.\n\nQuestion: {q}")


def hyde(q):
    return chat(f"{HINT}\nWrite a short passage (3-5 sentences) from that documentation that would answer the "
                f"question. Output only the passage.\n\nQuestion: {q}")


def multi(q):
    out = chat(f"{HINT}\nWrite 3 different search queries that would find documentation answering this question, "
               f"one per line, no numbering.\n\nQuestion: {q}")
    return [l.strip(" -0123456789.\"") for l in out.splitlines() if l.strip()][:3]


def hit(lists, gold):
    r = rank(rrf(lists), gold)
    return r is not None and r <= 50


def query_side(cur, golden):
    out, totals = [], {}
    for g in golden:
        q, gold = g["question"], g["relevant_chunks"]
        rw, hy, mq = rewrite(q), hyde(q), multi(q)
        eq, erw, ehy = embed([q, rw, hy])
        emq = embed(mq) if mq else []
        base = [vec_list(cur, eq, 50), fts_list(cur, q)[:50]]
        rwl = [vec_list(cur, erw, 50), fts_list(cur, rw)[:50]]
        mql = [vec_list(cur, e, 50) for e in emq] + [fts_list(cur, m)[:50] for m in mq]
        res = {
            "original": hit(base, gold),
            "rewrite_only": hit(rwl, gold),
            "orig+rewrite_rrf": hit(base + rwl, gold),
            "orig+hyde_vec_rrf": hit(base + [vec_list(cur, ehy, 50)], gold),
            "orig+multi_rrf": hit(base + mql, gold),
        }
        for k, v in res.items():
            totals[k] = totals.get(k, 0) + v
        out.append({"id": g["id"], "rewrite": rw, "multi": mq, **res})
        print(g["id"], res, flush=True)
    return out, totals


def document_side(cur, golden):
    out = []
    for qid in MISSES:
        g = golden[qid]
        q, gold = g["question"], g["relevant_chunks"][0]
        doc = gold.split("#")[0]
        cur.execute("""SELECT d.title, c.heading, c.text, c.ordinal FROM chunks c JOIN documents d ON d.id = c.document_id
                       WHERE c.superseded_at IS NULL AND c.id = %s""", (gold,))
        title, heading, text, _ = cur.fetchone()
        cur.execute("""SELECT c.text FROM chunks c WHERE c.superseded_at IS NULL AND c.document_id =
                       (SELECT document_id FROM chunks WHERE id = %s AND superseded_at IS NULL) ORDER BY ordinal""", (gold,))
        whole = "\n\n".join(r[0] for r in cur.fetchall())[:24000]
        ctx = chat(f"<document>\n{whole}\n</document>\nHere is the chunk we want to situate within the whole document\n"
                   f"<chunk>\n{text}\n</chunk>\nPlease give a short succinct context to situate this chunk within the "
                   f"overall document for the purposes of improving search retrieval of the chunk. Answer only with "
                   f"the succinct context and nothing else.")
        base_in = f"{title}\n{heading}\n\n{text}"
        eq, eb, ec = embed([q, base_in, f"{title}\n{heading}\n\n{ctx}\n\n{text}"])
        cur.execute("SELECT count(*) FROM chunks WHERE superseded_at IS NULL AND id <> %s AND (embedding <=> %s::vector) < %s",
                    (gold, str(eq), cos_dist(eq, eb)))
        r_base = cur.fetchone()[0] + 1
        cur.execute("SELECT count(*) FROM chunks WHERE superseded_at IS NULL AND id <> %s AND (embedding <=> %s::vector) < %s",
                    (gold, str(eq), cos_dist(eq, ec)))
        r_ctx = cur.fetchone()[0] + 1
        row = {"id": qid, "context": ctx, "vec_rank_recomputed": r_base, "vec_rank_contextual": r_ctx}
        print(json.dumps(row), flush=True)
        out.append(row)
    # q50 as one table row with its section breadcrumb and the table header.
    g = golden["daas-q50"]
    row_text = ("Retired and retiring features\nRetired and retiring features > Tenant setup, modules and systems\n\n"
                "| Feature | Status | Since | Tell the customer | Record |\n"
                "| Financial Impact Analysis (FIA) | Retired | Code 2023-10; tenant-module option 2025-06 | "
                "The feature no longer exists; nothing replaces it | `ARM-7700`, `ARM-31269`, database `ARM-29` |")
    eq, er = embed([g["question"], row_text])
    cur.execute("SELECT count(*) FROM chunks WHERE superseded_at IS NULL AND (embedding <=> %s::vector) < %s",
                (str(eq), cos_dist(eq, er)))
    out.append({"id": "daas-q50", "table_row_vec_rank": cur.fetchone()[0] + 1})
    print(out[-1], flush=True)
    return out


def cos_dist(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    golden = [json.loads(l) for l in open(sys.argv[1])]
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET enable_indexscan = off")
        doc = document_side(cur, {g["id"]: g for g in golden})
        qs, totals = query_side(cur, golden)
    print("TOTALS", totals)
    json.dump({"document_side": doc, "query_side": qs, "totals": totals}, open("docs/research/search/proto/results2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
