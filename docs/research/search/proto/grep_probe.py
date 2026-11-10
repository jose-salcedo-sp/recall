"""Naive rare-term grep probe over the DaaS corpus for Drew's missed golden questions.

Term rule (fixed before looking at results):
  1. Tokenise the question into [A-Za-z0-9_]+ words; lowercase; drop stopwords and words < 4 chars.
  2. Document frequency df(t) = number of ingested chunks whose text matches /\bt/i (prefix match, so
     'financial' also hits 'financials'), computed with ripgrep line hits mapped to chunks.
  3. Keep the K rarest terms with df >= 1 (K = 3). Also add each run of >= 2 Capitalised words as a phrase.
  4. ripgrep each selected term/phrase (case-insensitive, \b prefix) over the ingested layers, map file:line to
     the heading-section chunk id exactly as src/ingest/markdown.rs does, and score each chunk by the sum of
     idf = ln(N / df) over distinct matched terms (ties by chunk id).
Reports the rank of each target chunk in the grep list.
"""
import json, math, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ["DREW_DOCS_DIR"])
LAYERS = ["canon", "playbooks", "customer", "knowledge", "runbooks"]
SKIP = ["runbooks/tools"]
GOLDEN = Path("eval/golden/daas.jsonl")
MISSED = ["daas-q06", "daas-q09", "daas-q24", "daas-q30", "daas-q44", "daas-q45", "daas-q50", "daas-q51"]
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3

STOP = set("""a about above after again all also an and any are aren as at be because been before being below
between both but by can can't cannot could did do does doing don down during each few for from further had has
have having he her here hers him his how i if in into is isn it its itself just me more most must my no nor not
now of off on once only or other our out over own same she should so some such than that the their them then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours anyone anything someone something tell said says used still else working thing things
start enough doing make made want need able""".split())


def slug(h):
    s = ""
    for c in h.lower():
        if c.isalnum():
            s += c
        elif s and not s.endswith("-"):
            s += "-"
    s = s.rstrip("-")
    return s or "section"


def atx(line):
    t = line.strip()
    n = len(t) - len(t.lstrip("#"))
    if n == 0 or n > 6:
        return None
    text = t[n:].strip()
    return (n, text) if text else None


def line_to_chunk(path, rel):
    """Map each 1-based line number to its chunk locator, mirroring markdown.rs (front matter, fences, unique)."""
    lines = path.read_text(errors="replace").split("\n")
    # front matter: markdown.rs strips a leading '---' block
    start = 0
    stripped = "\n".join(lines).lstrip()
    if stripped.startswith("---"):
        lead = len(lines) - len(stripped.split("\n"))
        for i in range(lead + 1, len(lines)):
            if lines[i].startswith("---"):
                start = i + 1
                break
    sections, cur, path_h, fence = [], [], [], None
    for i in range(start, len(lines)):
        ln, t = lines[i], lines[i].strip()
        if fence:
            if t.startswith(fence):
                fence = None
            cur.append((i + 1, ln))
            continue
        if t.startswith("```") or t.startswith("~~~"):
            fence = t[:3]
            cur.append((i + 1, ln))
            continue
        h = atx(ln)
        if h:
            sections.append((list(path_h), cur))
            cur = []
            while len(path_h) >= h[0]:
                path_h.pop()
            path_h.append(h[1])
            cur.append((i + 1, ln))  # heading line belongs to its own section for grep purposes
            continue
        cur.append((i + 1, ln))
    sections.append((list(path_h), cur))
    used, out = {"intro"}, {}
    for ph, body in sections:
        text = "\n".join(l for n, l in body if not (ph and n == body[0][0] and atx(l)))
        if not text.strip():
            continue
        if not ph:
            loc = "intro"
        else:
            base, loc, n = slug(ph[-1]), slug(ph[-1]), 2
            while loc in used:
                loc = f"{base}-{n}"
                n += 1
            used.add(loc)
        cid = f"markdown:{rel}#{loc}"
        for n, _ in body:
            out[n] = cid
    return out


def corpus():
    files = []
    for layer in LAYERS:
        for p in sorted((ROOT / layer).rglob("*.md")):
            rel = p.relative_to(ROOT).as_posix()
            if any(rel.startswith(s + "/") for s in SKIP):
                continue
            files.append((p, rel))
    return files


FILES = corpus()
MAPS = {rel: line_to_chunk(p, rel) for p, rel in FILES}
ALL_CHUNKS = {c for m in MAPS.values() for c in m.values()}
N = len(ALL_CHUNKS)


def rg_chunks(pattern):
    args = ["rg", "-n", "-i", "--no-heading", "-e", pattern, "--"] + [rel for _, rel in FILES]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    hits = set()
    for line in r.stdout.splitlines():
        rel, n, _ = line.split(":", 2)
        c = MAPS[rel].get(int(n))
        if c:
            hits.add(c)
    return hits


def terms(q):
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", q)]
    words = [w for w in dict.fromkeys(words) if len(w) >= 4 and w not in STOP]
    phrases = [m for m in re.findall(r"((?:[A-Z][a-z]+\s+){1,}[A-Z][a-z]+)", q)]
    phrases = [p for p in dict.fromkeys(p.strip() for p in phrases)]
    return words, phrases


def main():
    gold = {j["id"]: j for j in map(json.loads, GOLDEN.read_text().splitlines())}
    print(f"files={len(FILES)} chunks={N} K={K}")
    summary = []
    for qid in MISSED:
        g = gold[qid]
        words, phrases = terms(g["question"])
        df = {w: rg_chunks(r"\b" + re.escape(w)) for w in words}
        rare = sorted([w for w in words if df[w]], key=lambda w: (len(df[w]), w))[:K]
        selected = {w: df[w] for w in rare}
        for p in phrases:
            hits = rg_chunks(r"\b" + r"\s+".join(map(re.escape, p.split())))
            if hits:
                selected[p] = hits
        score = {}
        for t, hits in selected.items():
            idf = math.log(N / len(hits))
            for c in hits:
                score[c] = score.get(c, 0) + idf
        ranked = sorted(score, key=lambda c: (-score[c], c))
        ranks = {t: (ranked.index(t) + 1 if t in score else None) for t in g["relevant_chunks"]}
        summary.append((qid, ranks, len(ranked)))
        print(f"\n{qid}: {g['question']}")
        print("  selected:", {t: len(h) for t, h in selected.items()})
        print(f"  grep list size: {len(ranked)}")
        for t, r in ranks.items():
            print(f"  target {t}: rank {r}")
    hit = lambda k: sum(1 for _, rs, _ in summary for r in rs.values() if r and r <= k)
    total = sum(len(rs) for _, rs, _ in summary)
    print(f"\nTOTAL targets={total} found_any={hit(10**9)} top50={hit(50)} top10={hit(10)} top5={hit(5)} top1={hit(1)}")
    full = sum(1 for _, rs, _ in summary if all(r and r <= 50 for r in rs.values()))
    print(f"questions with all targets in top50: {full}/{len(summary)}")


main()
