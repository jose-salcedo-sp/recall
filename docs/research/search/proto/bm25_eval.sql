CREATE TEMP TABLE st AS SELECT word, ndoc FROM ts_stat('SELECT fts_prose FROM chunks WHERE superseded_at IS NULL');
CREATE UNIQUE INDEX ON st(word);
CREATE TEMP TABLE dl AS SELECT c.id, sum(coalesce(array_length(u.positions,1),1))::float dl FROM chunks c, unnest(c.fts_prose) u WHERE superseded_at IS NULL GROUP BY c.id;
CREATE TEMP TABLE cand AS
SELECT g.qid, c.id,
  length(c.fts_tokens) - length(ts_delete(c.fts_tokens, g.ids)) AS idm,
  coalesce(ts_rank(c.fts_prose, to_tsquery('english', g.prose), 1), 0) AS tsr,
  b.bm25, b.bm25f,
  (SELECT coalesce(sum((SELECT sum(ln(1 + (n.n - st.ndoc + 0.5)/(st.ndoc + 0.5)))
                        FROM unnest(tsvector_to_array(to_tsvector('english', p))) l JOIN st ON st.word = l)), 0)
     FROM unnest(g.phrases) p WHERE c.fts_prose @@ phraseto_tsquery('english', p)) AS phrase_bonus
FROM g JOIN chunks c ON c.superseded_at IS NULL
  AND (c.fts_prose @@ to_tsquery('english', g.prose) OR c.fts_tokens @@ g.tok::tsquery)
JOIN dl ON dl.id = c.id
CROSS JOIN (SELECT count(*)::float n FROM chunks WHERE superseded_at IS NULL) n
CROSS JOIN (SELECT avg(dl) avgdl FROM dl) a
CROSS JOIN LATERAL (
  SELECT coalesce(sum(idf * tf*2.2/(tf + 1.2*(0.25+0.75*dl.dl/a.avgdl))),0) bm25,
         coalesce(sum(idf * tfa*2.2/(tfa + 1.2*(0.25+0.75*dl.dl/a.avgdl))),0) bm25f
  FROM (SELECT ln(1 + (n.n - st.ndoc + 0.5)/(st.ndoc + 0.5)) idf, t.tf,
               t.tf + CASE WHEN 'A' = ANY(t.weights) THEN 1 ELSE 0 END AS tfa
        FROM (SELECT u.lexeme, coalesce(array_length(u.positions,1),1) tf, u.weights FROM unnest(c.fts_prose) u
              WHERE u.lexeme = ANY(tsvector_to_array(to_tsvector('english', g.q)))) t
        JOIN st ON st.word = t.lexeme) s) b;
CREATE TEMP TABLE ranked AS SELECT qid, id,
  row_number() OVER (PARTITION BY qid ORDER BY idm DESC, tsr DESC, id) r_cur,
  row_number() OVER (PARTITION BY qid ORDER BY idm DESC, bm25 DESC, id) r_bm,
  row_number() OVER (PARTITION BY qid ORDER BY idm DESC, bm25f DESC, id) r_bmf,
  row_number() OVER (PARTITION BY qid ORDER BY idm DESC, bm25 + phrase_bonus DESC, id) r_bmp,
  row_number() OVER (PARTITION BY qid ORDER BY idm DESC, bm25f + phrase_bonus DESC, id) r_bmfp
FROM cand;
WITH per AS (
 SELECT g.qid, cardinality(g.rel) nrel,
   count(*) FILTER (WHERE r.r_cur<=50) cur, count(*) FILTER (WHERE r.r_bm<=50) bm,
   count(*) FILTER (WHERE r.r_bmf<=50) bmf, count(*) FILTER (WHERE r.r_bmp<=50) bmp, count(*) FILTER (WHERE r.r_bmfp<=50) bmfp,
   count(*) FILTER (WHERE r.r_cur<=10) cur10, count(*) FILTER (WHERE r.r_bm<=10) bm10, count(*) FILTER (WHERE r.r_bmfp<=10) bmfp10,
   max(1.0/r.r_cur) mrr_cur, max(1.0/r.r_bm) mrr_bm, max(1.0/r.r_bmfp) mrr_bmfp
 FROM g LEFT JOIN ranked r ON r.qid=g.qid AND r.id = ANY(g.rel)
 WHERE cardinality(g.rel)>0 GROUP BY g.qid, g.rel)
SELECT count(*) q,
 round(avg(cur::float/nrel)::numeric,3) r50_tsrank, round(avg(bm::float/nrel)::numeric,3) r50_bm25,
 round(avg(bmf::float/nrel)::numeric,3) r50_bm25f, round(avg(bmp::float/nrel)::numeric,3) r50_bm25_phr, round(avg(bmfp::float/nrel)::numeric,3) r50_bm25f_phr,
 round(avg(cur10::float/nrel)::numeric,3) r10_tsrank, round(avg(bm10::float/nrel)::numeric,3) r10_bm25, round(avg(bmfp10::float/nrel)::numeric,3) r10_bm25f_phr,
 round(avg(coalesce(mrr_cur,0))::numeric,3) mrr_tsrank, round(avg(coalesce(mrr_bm,0))::numeric,3) mrr_bm25, round(avg(coalesce(mrr_bmfp,0))::numeric,3) mrr_bm25f_phr
FROM per;
SELECT g.qid, r.r_cur, r.r_bm, r.r_bmf, r.r_bmp, r.r_bmfp FROM g JOIN ranked r ON r.qid=g.qid AND r.id = ANY(g.rel)
WHERE g.qid IN ('daas-q50','daas-q17','daas-q18','daas-q09','daas-q24') ORDER BY 1;
