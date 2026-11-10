"""Read-only retrieval prototype: rank of gold chunks under query/document enrichments.

Ranks are 1-based among the 1,386 current chunks. Full text replicates the prose side of
src/retrieval/hybrid.rs (OR of words, ts_rank normalization 1). Fused = RRF(k=60) over the
top-50 of each list, as HybridConfig defaults do.
"""
import json, re, sys, urllib.request
import psycopg

DB = "postgres://drew:drew@localhost:5433/drew"
OLLAMA = "http://localhost:11434"
MISSES = ["daas-q06", "daas-q09", "daas-q24", "daas-q30", "daas-q44", "daas-q45", "daas-q50", "daas-q51"]

# Hand-applied lookups, each from one row of canon/GLOSSARY.md or customer/LEXICON.md (row cited in enrichment.md).
GLOSSARY = {
    "daas-q06": "Segregation-of-duties risk: two conflicting business functions",
    "daas-q09": "RMS Report Management Service owns the report-embed QIX request path",
    "daas-q24": "BUG-nn an issue we're tracking disclosure rules; known issue: an issue we're aware of and are looking into",
    "daas-q30": "",
    "daas-q44": "Legacy / NEW generation: the current screens and the earlier screens",
    "daas-q45": "",
    "daas-q50": "",
    "daas-q51": "",
}


def post(path, body):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def embed(texts):
    return post("/api/embed", {"model": "bge-m3", "input": texts})["embeddings"]


def chat(prompt):
    out = post("/api/chat", {
        "model": "qwen3:14b", "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 7, "num_predict": 300, "num_ctx": 16384},
        "messages": [{"role": "user", "content": prompt}],
    })["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


CORPUS_HINT = ("The corpus is internal support documentation for SailPoint Access Risk Management (ARM) on SAP: "
               "rulebooks, risk analysis, hits, mitigating controls, reports (Qlik), legacy vs current generations, "
               "tenant setup, customer communication rules.")


def rewrite(q):
    return chat(f"{CORPUS_HINT}\nRewrite this support question as one search query of 10-25 keywords in the "
                f"vocabulary the documentation would use (formal terms, synonyms, likely section headings). "
                f"Output only the query.\n\nQuestion: {q}")


def hyde(q):
    return chat(f"{CORPUS_HINT}\nWrite a short passage (3-5 sentences) from that documentation that would answer "
                f"the question. Output only the passage.\n\nQuestion: {q}")


def multi(q):
    out = chat(f"{CORPUS_HINT}\nWrite 3 different search queries that would find documentation answering this "
               f"question, one per line, no numbering.\n\nQuestion: {q}")
    return [line.strip(" -0123456789.") for line in out.splitlines() if line.strip()][:3]


def stepback(q):
    return chat(f"{CORPUS_HINT}\nWhat is the more general, underlying question behind this one? "
                f"Output only that question.\n\nQuestion: {q}")


def vec_list(cur, v, n=None):
    cur.execute("SELECT id FROM chunks WHERE superseded_at IS NULL ORDER BY embedding <=> %s::vector, id"
                + (f" LIMIT {n}" if n else ""), (str(v),))
    return [r[0] for r in cur.fetchall()]


def fts_list(cur, q):
    words = [w for w in re.split(r"[^\w]+", q) if w and w.isalnum()]
    if not words:
        return []
    cur.execute("""SELECT c.id FROM chunks c, (SELECT to_tsquery('english', %s) AS q) t
                   WHERE c.superseded_at IS NULL AND c.fts_prose @@ t.q
                   ORDER BY ts_rank(c.fts_prose, t.q, 1) DESC, c.id""", (" | ".join(words),))
    return [r[0] for r in cur.fetchall()]


def rank(lst, gold):
    rs = [lst.index(g) + 1 for g in gold if g in lst]
    return min(rs) if rs else None


def rrf(lists, k=60, top=50):
    s = {}
    for lst in lists:
        for i, cid in enumerate(lst[:top]):
            s[cid] = s.get(cid, 0) + 1 / (k + i + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))]


def evaluate(cur, gold, vec_queries, fts_queries):
    vls = [vec_list(cur, v) for v in vec_queries]
    fls = [fts_list(cur, q) for q in fts_queries]
    return {
        "vec": min((rank(l, gold) or 9999) for l in vls),
        "fts": min((rank(l, gold) or 9999) for l in fls) if fls else None,
        "fused": rank(rrf(vls + fls), gold) or ">50x",
    }


def main():
    golden = {j["id"]: j for j in map(json.loads, open(sys.argv[1]))}
    rows = []
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET enable_indexscan = off")  # exact scan: rank over every chunk, not HNSW's approximation
        for qid in MISSES:
            g = golden[qid]
            q, gold = g["question"], g["relevant_chunks"]
            rw, hy, mq, sb = rewrite(q), hyde(q), multi(q), stepback(q)
            gl = f"{q} {GLOSSARY[qid]}".strip()
            e = dict(zip(["q", "gl", "rw", "hy", "qhy", "sb"], embed([q, gl, rw, hy, f"{q}\n{hy}", sb])))
            me = embed(mq)
            row = {"id": qid, "q": q, "rewrite": rw, "hyde": hy, "multi": mq, "stepback": sb, "variants": {
                "original": evaluate(cur, gold, [e["q"]], [q]),
                "glossary": evaluate(cur, gold, [e["gl"]], [gl]) if GLOSSARY[qid] else None,
                "llm_rewrite": evaluate(cur, gold, [e["rw"]], [rw]),
                "hyde_vec_only": evaluate(cur, gold, [e["hy"]], [q]),
                "q+hyde": evaluate(cur, gold, [e["qhy"]], [q]),
                "multi_query_rrf": evaluate(cur, gold, [e["q"]] + me, [q] + mq),
                "stepback+orig_rrf": evaluate(cur, gold, [e["q"], e["sb"]], [q, sb]),
            }}
            rows.append(row)
            print(json.dumps({"id": qid, **{k: v for k, v in row["variants"].items()}}), flush=True)
    json.dump(rows, open("docs/research/search/proto/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
