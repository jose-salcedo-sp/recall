#!/usr/bin/env python3
"""Answer-chunk rank probe.

Judges retrieve (and optionally admit) by where the chunk that actually answers
the question landed — not by the noul histogram.

Usage:
    scripts/rank_probe.py              retrieve ranks only (embed + SQL)
    scripts/rank_probe.py --admit      also run /v1/ask/sync for noul rank
    scripts/rank_probe.py --json       machine-readable table

Requires: stack up, seeded corpus, psql, embedder credentials in the env.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

BASE = os.environ.get("RECALL_BASE", "http://localhost:8000")
BRAIN = "00000000-0000-0000-0000-000000000001"
K = int(os.environ.get("RETRIEVE_K", "32"))
EMBEDDER = os.environ.get("EMBEDDER_URL", "https://ai-gateway.vercel.sh").rstrip("/")
MODEL = os.environ.get("NEXUS_EMBEDDING_MODEL", "openai/text-embedding-3-small")
API_KEY = os.environ.get("EMBEDDER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
TOKEN = os.environ.get("RECALL_SERVICE_TOKEN", "")

# Frozen set: Finding 4 (first four) plus real asks against the seed corpus.
# value is a statement substring identifying the answer chunk, or None if the
# corpus cannot answer it.
PROBES: list[tuple[str, str | None]] = [
    ("When is my sister Ana's birthday?", "Ana's birthday is March 14"),
    ("What am I allergic to?", "allergic to penicillin"),
    ("What is the wifi password at the cabin?", "wifi password at the cabin is redmaple"),
    ("What is the atomic mass of ruthenium?", None),
    ("When is Luis's birthday?", "Luis's birthday is July 2"),
    ("Where does Ana live?", "Ana lives in Denver"),
    ("What is my mom's name?", "Carmen Salcedo"),
    ("What is the home wifi password?", "Winterthorn-5G with password winterthorn"),
    ("How old is Maple?", "Maple is a six-year-old"),
    ("Who is my manager?", "manager is Elena Voss"),
    ("When did I start this job?", "started this job on 2022-09-12"),
    ("What is the cabin lockbox code?", "cabin lockbox code is 4218"),
    ("When is Maya's birthday?", "Maya's birthday is December 5"),
    ("What is Maya's work badge PIN?", "badge PIN is 3301"),
    ("What is Maya allergic to?", "allergic to shellfish"),
    ("Where is Priya's spare key?", "under the blue planter"),
    ("When is Luis's anniversary?", "anniversary is June 8"),
    ("What is my car's license plate?", "plate KRN-441"),
    ("Who is my dentist?", "dentist is Dr. Okonkwo"),
    ("Where did we stay in Lisbon?", "Hotel do Chiado"),
    ("What is my employee ID?", "Employee ID is R-10428"),
    ("When is the Austin wedding?", "wedding in Austin on 2026-10-10"),
    ("What is my blood type?", "blood type is O positive"),
    ("How do I take my coffee?", "coffee with oat milk"),
]


def psql(sql: str) -> str:
    proc = subprocess.run(
        [
            "psql",
            "-h",
            "localhost",
            "-p",
            "5433",
            "-U",
            "recall",
            "-d",
            "recall",
            "-t",
            "-A",
            "-F",
            "|",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        env={"PGPASSWORD": "recall", "PATH": os.environ.get("PATH", "/usr/bin:/bin:/opt/homebrew/bin")},
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout)
    return proc.stdout


def embed(text: str) -> list[float]:
    payload = json.dumps({"input": [text], "model": MODEL})
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-m",
            "60",
            f"{EMBEDDER}/v1/embeddings",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: Bearer {API_KEY}",
            "-d",
            payload,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    return data["data"][0]["embedding"]


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def search(question: str, embedding: list[float], k: int, rrf: bool) -> list[dict]:
    lit = vec_literal(embedding).replace("'", "''")
    q = question.replace("'", "''")
    if rrf:
        sql = f"""
        WITH p AS (
          SELECT memory_id, memory_statement,
                 row_number() OVER (ORDER BY score DESC) AS rank
          FROM hybrid_search_brain(
            '{BRAIN}'::uuid, '{q}', '{lit}'::vector, now(), {k})
        ),
        g AS (
          SELECT memory_id, memory_statement, origin,
                 row_number() OVER (ORDER BY score DESC) AS rank
          FROM search_mounted_for_brain(
            '{BRAIN}'::uuid, '{q}', '{lit}'::vector, now(), {k})
        ),
        fused AS (
          SELECT coalesce(p.memory_id, g.memory_id) AS id,
                 coalesce(p.memory_statement, g.memory_statement) AS statement,
                 coalesce(g.origin, 'personal') AS origin,
                 coalesce(1.0/(60+p.rank),0) + coalesce(1.0/(60+g.rank),0) AS rrf
          FROM p
          FULL OUTER JOIN g ON p.memory_id = g.memory_id
        )
        SELECT id, statement, origin, rrf FROM fused ORDER BY rrf DESC LIMIT {k};
        """
    else:
        # Current retrieve.rs: personal then granted, truncate k.
        sql = f"""
        WITH p AS (
          SELECT memory_id AS id, memory_statement AS statement,
                 'personal'::text AS origin, score
          FROM hybrid_search_brain(
            '{BRAIN}'::uuid, '{q}', '{lit}'::vector, now(), {k})
        ),
        g AS (
          SELECT memory_id AS id, memory_statement AS statement, origin, score
          FROM search_mounted_for_brain(
            '{BRAIN}'::uuid, '{q}', '{lit}'::vector, now(), {k})
        )
        SELECT id, statement, origin, score FROM (
          SELECT *, 0 AS src FROM p
          UNION ALL
          SELECT *, 1 AS src FROM g
        ) x
        ORDER BY src, score DESC
        LIMIT {k};
        """
    rows = []
    for line in psql(sql).strip().split("\n"):
        if "|" not in line:
            continue
        id_, stmt, origin, rrf = line.split("|", 3)
        rows.append(
            {"id": id_, "statement": stmt, "origin": origin, "rrf": float(rrf)}
        )
    return rows


def post_sync(question: str) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/ask/sync",
        json.dumps({"question": question, "brain_id": BRAIN}).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


def audit(ask_id: str) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/asks/{ask_id}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


def statements(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    q = "select id::text, statement from chunks where id in (%s);" % ",".join(
        "'%s'" % i for i in ids
    )
    out = {}
    for line in psql(q).strip().split("\n"):
        if "|" in line:
            i, s = line.split("|", 1)
            out[i] = s
    return out


def find_rank(rows: list[dict], expect: str | None, stmt_key: str = "statement") -> int | None:
    if expect is None:
        return None
    for i, r in enumerate(rows, 1):
        if expect in r.get(stmt_key, ""):
            return i
    return None


def main() -> int:
    admit = "--admit" in sys.argv
    as_json = "--json" in sys.argv
    rrf = "--rrf" in sys.argv
    if not API_KEY:
        print("FATAL: OPENAI_API_KEY / EMBEDDER_API_KEY required", file=sys.stderr)
        return 2

    table = []
    empty = 0
    answerable = 0
    in_k = 0
    fails = 0

    for question, expect in PROBES:
        vec = embed(question)
        retrieved = search(question, vec, K, rrf)
        retrieve_rank = find_rank(retrieved, expect)
        row = {
            "question": question,
            "expect": expect,
            "retrieve_rank": retrieve_rank,
            "retrieve_k": K,
            "noul_rank": None,
            "empty": None,
            "admitted": None,
        }
        if expect is None:
            pass
        else:
            answerable += 1
            if retrieve_rank is not None:
                in_k += 1
            else:
                fails += 1

        if admit:
            r = post_sync(question)
            a = audit(r["ask_id"])
            names = statements([c["id"] for c in a["candidates"]])
            ranked = sorted(a["candidates"], key=lambda c: -(c.get("noul") or c.get("evidence") or 0))
            for c in ranked:
                c["statement"] = names.get(c["id"], "")
            row["noul_rank"] = find_rank(ranked, expect)
            row["empty"] = r.get("empty")
            row["admitted"] = not r.get("empty")
            if expect is None:
                empty += 1 if r.get("empty") else 0
                if not r.get("empty"):
                    fails += 1
            elif row["noul_rank"] != 1:
                fails += 1

        table.append(row)
        if not as_json:
            rr = row["retrieve_rank"]
            nr = row["noul_rank"]
            mark = " " if expect is None else ("*" if rr == 1 else " ")
            extra = ""
            if admit:
                extra = f"  noul@{nr}  empty={row['empty']}"
            print(f"{mark} retrieve@{rr}  {question}{extra}")

    summary = {
        "k": K,
        "probes": len(PROBES),
        "answerable": answerable,
        "answer_chunk_in_k": in_k,
        "empty_admit_on_holes": empty if admit else None,
        "fails": fails,
        "rows": table,
    }
    if as_json:
        json.dump(summary, sys.stdout, indent=2)
        print()
    else:
        print(
            f"\nanswer-chunk in retrieve@{K}: {in_k}/{answerable}"
            + (f"   empty-admit holes: {empty}" if admit else "")
        )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
