-- Local-only stand-ins for the two Nexus search functions.
--
-- Recall's retrieve stage calls `hybrid_search_brain` and `search_mounted_for_brain`
-- because that is the only surface the `recall_search` role is granted on Nexus.
-- Without these, the local Compose stack could not exercise the retrieve path at all
-- and `scripts/e2e.sh` would only ever test the Nexus deployment by proxy.
--
-- These are NOT the Nexus definitions and must never be applied there: this whole
-- migration set runs only when INDEX_WRITES_ENABLED=true, which is off against Nexus.
-- Grant semantics here are a crude local imitation (origin = 'granted' rows in the
-- local corpus); the real rules live in the Nexus function and are not reproduced.
--
-- The embedding parameter is vector(768) to match the local nomic embedder. Nexus
-- declares vector(1536) for text-embedding-3-small. Same call site, different
-- declared type, which is why sql/search_*.sql bind the vector positionally.

CREATE OR REPLACE FUNCTION hybrid_search_brain(
    p_brain_id        uuid,
    p_query_text      text,
    p_query_embedding vector(768),
    p_as_of           timestamptz,
    p_k               int
)
RETURNS TABLE (
    id               uuid,
    memory_statement text,
    text             text,
    grantor_name     text,
    source           text,
    occurred_at      timestamptz,
    score            float8
)
LANGUAGE sql STABLE AS $$
    WITH ts AS (SELECT coalesce(p_as_of, now()) AS at),
    tsq AS (SELECT websearch_to_tsquery('english', coalesce(p_query_text, '')) AS q),
    vec AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> p_query_embedding) AS rank
        FROM chunks c, ts
        WHERE c.brain_id = p_brain_id AND c.origin = 'personal'
          AND c.state = 'active' AND c.sensitivity <> 'secret'
          AND (c.valid_from IS NULL OR c.valid_from <= ts.at)
          AND (c.valid_to   IS NULL OR c.valid_to   >  ts.at)
        ORDER BY c.embedding <=> p_query_embedding
        LIMIT p_k * 4
    ),
    fts AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, tsq.q) DESC) AS rank
        FROM chunks c, ts, tsq
        WHERE c.brain_id = p_brain_id AND c.origin = 'personal'
          AND c.state = 'active' AND c.sensitivity <> 'secret'
          AND (c.valid_from IS NULL OR c.valid_from <= ts.at)
          AND (c.valid_to   IS NULL OR c.valid_to   >  ts.at)
          AND numnode(tsq.q) > 0 AND c.tsv @@ tsq.q
        ORDER BY ts_rank_cd(c.tsv, tsq.q) DESC
        LIMIT p_k * 4
    )
    SELECT c.id, c.statement, c.text, NULL::text, NULL::text, NULL::timestamptz,
           coalesce(1.0 / (60 + v.rank), 0) + coalesce(1.0 / (60 + f.rank), 0)
    FROM chunks c
    LEFT JOIN vec v ON v.id = c.id
    LEFT JOIN fts f ON f.id = c.id
    WHERE v.id IS NOT NULL OR f.id IS NOT NULL
    ORDER BY 7 DESC
    LIMIT p_k;
$$;

CREATE OR REPLACE FUNCTION search_mounted_for_brain(
    p_requester       uuid,
    p_query_text      text,
    p_query_embedding vector(768),
    p_as_of           timestamptz,
    p_k               int
)
RETURNS TABLE (
    id               uuid,
    memory_statement text,
    text             text,
    grantor_name     text,
    source           text,
    occurred_at      timestamptz,
    score            float8
)
LANGUAGE sql STABLE AS $$
    WITH ts AS (SELECT coalesce(p_as_of, now()) AS at)
    SELECT c.id, c.statement, c.text, c.grantor_name, NULL::text, NULL::timestamptz,
           1.0 - (c.embedding <=> p_query_embedding)
    FROM chunks c, ts
    WHERE c.brain_id = p_requester AND c.origin = 'granted'
      AND c.state = 'active' AND c.sensitivity <> 'secret'
      AND (c.valid_from IS NULL OR c.valid_from <= ts.at)
      AND (c.valid_to   IS NULL OR c.valid_to   >  ts.at)
    ORDER BY c.embedding <=> p_query_embedding
    LIMIT p_k;
$$;
