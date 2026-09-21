#!/usr/bin/env python3
"""Show what admission actually scored, joined to chunk statements.

Sorted noul values alone are not enough to judge the classifier: they look
reasonable even when the ranking is anti-correlated with relevance. This prints
the statement next to each score, and reports where the chunk that genuinely
answers the question landed.

Usage:  scripts/rank_probe.py            (runs the built-in probe set)
Requires the stack up and the recall binary already running on :8000.
"""
import json
import os
import subprocess
import sys
import urllib.request

BASE = os.environ.get("RECALL_BASE", "http://localhost:8000")
BRAIN = "00000000-0000-0000-0000-000000000001"

# question -> substring identifying the chunk that should answer it, or None if
# the corpus genuinely cannot answer it.
PROBES = {
    "When is my sister Ana's birthday?": "Ana's birthday",
    "What am I allergic to?": "allergic to",
    "What is the wifi password at the cabin?": "cabin is redmaple",
    "What is the atomic mass of ruthenium?": None,
}


def post(path, body):
    req = urllib.request.Request(
        f"{BASE}{path}", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("RECALL_SERVICE_TOKEN","")}
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


def statements(ids):
    q = "select id,statement from chunks where id in (%s);" % ",".join(
        "'%s'" % i for i in ids
    )
    out = subprocess.run(
        ["psql", "-h", "localhost", "-p", "5433", "-U", "recall", "-d", "recall",
         "-t", "-A", "-F", "|", "-c", q],
        capture_output=True, text=True,
        env={"PGPASSWORD": "recall", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    ).stdout
    return dict(l.split("|", 1) for l in out.strip().split("\n") if "|" in l)


def main():
    fails = 0
    for question, expect in PROBES.items():
        r = post("/v1/ask/sync", {"question": question, "brain_id": BRAIN})
        audit = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/asks/{r['ask_id']}", headers={"Authorization": "Bearer " + os.environ.get("RECALL_SERVICE_TOKEN","")}), timeout=60))
        cands = audit["candidates"]
        names = statements([c["id"] for c in cands])
        ranked = sorted(cands, key=lambda c: -(c["noul"] or 0))

        print(f"\n{question}")
        for i, c in enumerate(ranked[:5], 1):
            mark = "*" if expect and expect in names.get(c["id"], "") else " "
            print(f"  {mark}{i:>2}  {c['noul']:.4f}  {names.get(c['id'], '?')[:58]}")

        if expect is None:
            ok = r["empty"]
            print(f"   unanswerable -> empty admit: {ok}")
        else:
            pos = next(
                (i for i, c in enumerate(ranked, 1) if expect in names.get(c["id"], "")),
                None,
            )
            ok = pos == 1
            print(f"   answer chunk rank: {pos}   admitted: {not r['empty']}")
        if not ok:
            fails += 1

    print(f"\n{len(PROBES) - fails}/{len(PROBES)} probes correct")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
