-- Hybrid retrieve: vector KNN + English FTS, fused with Reciprocal Rank Fusion.
--
-- Called from Rust via sqlx with positional parameters, in this order:
--   $1  vector(768)   query embedding
--   $2  text          query text (fed to websearch_to_tsquery('english', $2))
--   $3  uuid          brain_id
--   $4  timestamptz   as_of; NULL is treated as now()
--   $5  int           k, number of rows to return (each list is limited to k*4)
--
-- Empty or stopword-only $2 yields an empty tsquery. FTS is skipped in that
-- case (numnode = 0) so the statement still returns vector hits instead of erroring.

WITH vec AS (
    SELECT
        id,
        statement,
        text,
        origin,
        grantor_name,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
    FROM chunks
    WHERE brain_id = $3
      AND state = 'active'
      AND sensitivity <> 'secret'
      AND (valid_from IS NULL OR valid_from <= coalesce($4, now()))
      AND (valid_to IS NULL OR valid_to > coalesce($4, now()))
    ORDER BY embedding <=> $1
    LIMIT ($5 * 4)
),
fts AS (
    SELECT
        id,
        statement,
        text,
        origin,
        grantor_name,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, tsq) DESC) AS rank
    FROM chunks
    CROSS JOIN LATERAL (
        SELECT websearch_to_tsquery('english', coalesce($2, '')) AS tsq
    ) q
    WHERE brain_id = $3
      AND state = 'active'
      AND sensitivity <> 'secret'
      AND (valid_from IS NULL OR valid_from <= coalesce($4, now()))
      AND (valid_to IS NULL OR valid_to > coalesce($4, now()))
      AND numnode(q.tsq) > 0
      AND tsv @@ q.tsq
    ORDER BY ts_rank_cd(tsv, q.tsq) DESC
    LIMIT ($5 * 4)
)
SELECT
    coalesce(vec.id, fts.id) AS id,
    coalesce(vec.statement, fts.statement) AS statement,
    coalesce(vec.text, fts.text) AS text,
    coalesce(vec.origin, fts.origin) AS origin,
    coalesce(vec.grantor_name, fts.grantor_name) AS grantor_name,
    vec.rank::int AS vec_rank,
    fts.rank::int AS fts_rank,
    (
        coalesce(1.0 / (60 + vec.rank), 0)
        + coalesce(1.0 / (60 + fts.rank), 0)
    )::float8 AS rrf_score
FROM vec
FULL OUTER JOIN fts ON vec.id = fts.id
ORDER BY rrf_score DESC
LIMIT $5;
