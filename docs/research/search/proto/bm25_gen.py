import json, re
rows=[]
lit=lambda s: "'"+s.replace("'","''")+"'"
arr=lambda xs: "ARRAY["+",".join(lit(x) for x in xs)+"]::text[]"
for line in open('eval/golden/daas.jsonl'):
    if not line.strip(): continue
    d=json.loads(line); q=d['question']
    prose=[w for w in re.split(r'[^0-9A-Za-z\u00C0-\uFFFF]', q) if w]
    ids=sorted({t.lower() for t in (x.strip('_') for x in re.split(r'[^A-Za-z0-9_]', q)) if len(t)<64 and re.search('[A-Za-z]',t) and re.search(r'[_0-9]',t)})
    # runs of >=2 capitalised words, ignoring the sentence-initial word
    words=re.findall(r"[A-Za-z][A-Za-z0-9'-]*", q)
    phrases=[]; run=[]
    for i,w in enumerate(words):
        if w[0].isupper() and i>0: run.append(w)
        else:
            if len(run)>=2: phrases.append(' '.join(run))
            run=[]
    if len(run)>=2: phrases.append(' '.join(run))
    rows.append(f"({lit(d['id'])},{lit(q)},{lit(' | '.join(prose))},{lit(' | '.join(chr(39)+i+chr(39) for i in ids)) if ids else 'NULL'},{arr(ids)},{arr(d.get('relevant_chunks',[]))},{arr(phrases)})")
print("CREATE TEMP TABLE g(qid text, q text, prose text, tok text, ids text[], rel text[], phrases text[]);")
print("INSERT INTO g VALUES\n"+",\n".join(rows)+";")
print("SELECT qid, phrases FROM g WHERE cardinality(phrases)>0;")
print(open(__file__.replace('bm25_gen.py','bm25_eval.sql')).read())
