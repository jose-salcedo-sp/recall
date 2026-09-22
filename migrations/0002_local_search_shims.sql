-- Local-only stand-ins for the two Nexus search functions.
--
-- Recall's retrieve stage calls `hybrid_search_brain` and `search_mounted_for_brain`
-- because that is the only surface the `recall_search` role is granted on Nexus.
-- Without these, the local Compose stack could not exercise the retrieve path and
-- `scripts/e2e.sh` would test nothing of it.
--
-- These are NOT the Nexus definitions and must never be applied there: this whole
-- migration set runs only when INDEX_WRITES_ENABLED=true, which is off against Nexus.
-- Grant semantics here are a crude local imitation (origin = 'granted' rows in the
-- local corpus); the real rules live in the Nexus function and are not reproduced.
--
-- Signatures and column names mirror the deployed Nexus ones exactly, verified with
-- scripts/introspect_nexus.sh, so sql/search_*.sql is correct against both. Two
-- deliberate difference: columns the real functions expose but the local corpus has
-- no data for are returned as NULL. The vector(1536) type and every column name match
-- Nexus exactly, so sql/search_*.sql is correct against both with no dimension caveat.

DROP FUNCTION IF EXISTS hybrid_search_brain(uuid, text, vector, timestamptz, int);
DROP FUNCTION IF EXISTS search_mounted_for_brain(uuid, text, vector, timestamptz, int);

CREATE FUNCTION hybrid_search_brain(
    p_brain_id      uuid,
    query_text      text,
    query_embedding vector(1536),
    as_of           timestamptz DEFAULT now(),
    match_limit     int         DEFAULT 12
)
RETURNS TABLE (
    id               uuid,
    text             text,
    segment_ref      text,
    score            double precision,
    memory_id        uuid,
    memory_statement text,
    valid_from       timestamptz,
    valid_to         timestamptz
)
LANGUAGE sql STABLE AS $$
    WITH ts AS (SELECT coalesce(as_of, now()) AS at),
    tsq AS (SELECT websearch_to_tsquery('english', coalesce(query_text, '')) AS q),
    vec AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rank
        FROM chunks c, ts
        WHERE c.brain_id = p_brain_id AND c.origin = 'personal'
          AND c.state = 'active' AND c.sensitivity <> 'secret'
          AND (c.valid_from IS NULL OR c.valid_from <= ts.at)
          AND (c.valid_to   IS NULL OR c.valid_to   >  ts.at)
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_limit * 4
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
        LIMIT match_limit * 4
    )
    SELECT c.id, c.text, NULL::text, 
           coalesce(1.0 / (60 + v.rank), 0) + coalesce(1.0 / (60 + f.rank), 0),
           c.id, c.statement, c.valid_from, c.valid_to
    FROM chunks c
    LEFT JOIN vec v ON v.id = c.id
    LEFT JOIN fts f ON f.id = c.id
    WHERE v.id IS NOT NULL OR f.id IS NOT NULL
    ORDER BY 4 DESC
    LIMIT match_limit;
$$;

-- Same hybrid as personal. Rank fusion lives in Recall; this must not be a
-- vector-only list or a granted lexical hit is dropped before RRF sees it.
CREATE FUNCTION search_mounted_for_brain(
    p_requester     uuid,
    query_text      text,
    query_embedding vector(1536),
    as_of           timestamptz DEFAULT now(),
    match_limit     int         DEFAULT 12
)
RETURNS TABLE (
    id                uuid,
    text              text,
    segment_ref       text,
    score             double precision,
    memory_id         uuid,
    memory_statement  text,
    valid_from        timestamptz,
    valid_to          timestamptz,
    origin            text,
    grantor_brain_id  uuid,
    grantor_name      text,
    grantor_face_seed text,
    org_id            uuid,
    source_id         uuid,
    source_channel    text,
    source_name       text,
    occurred_at       timestamptz,
    sensitivity       text
)
LANGUAGE sql STABLE AS $$
    WITH ts AS (SELECT coalesce(as_of, now()) AS at),
    tsq AS (SELECT websearch_to_tsquery('english', coalesce(query_text, '')) AS q),
    vec AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rank
        FROM chunks c, ts
        WHERE c.brain_id = p_requester AND c.origin = 'granted'
          AND c.state = 'active' AND c.sensitivity <> 'secret'
          AND (c.valid_from IS NULL OR c.valid_from <= ts.at)
          AND (c.valid_to   IS NULL OR c.valid_to   >  ts.at)
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_limit * 4
    ),
    fts AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, tsq.q) DESC) AS rank
        FROM chunks c, ts, tsq
        WHERE c.brain_id = p_requester AND c.origin = 'granted'
          AND c.state = 'active' AND c.sensitivity <> 'secret'
          AND (c.valid_from IS NULL OR c.valid_from <= ts.at)
          AND (c.valid_to   IS NULL OR c.valid_to   >  ts.at)
          AND numnode(tsq.q) > 0 AND c.tsv @@ tsq.q
        ORDER BY ts_rank_cd(c.tsv, tsq.q) DESC
        LIMIT match_limit * 4
    )
    SELECT c.id, c.text, NULL::text,
           coalesce(1.0 / (60 + v.rank), 0) + coalesce(1.0 / (60 + f.rank), 0),
           c.id, c.statement, c.valid_from, c.valid_to,
           'granted'::text, c.grantor_brain_id, c.grantor_name, NULL::text,
           NULL::uuid, NULL::uuid, NULL::text, NULL::text,
           NULL::timestamptz, c.sensitivity
    FROM chunks c
    LEFT JOIN vec v ON v.id = c.id
    LEFT JOIN fts f ON f.id = c.id
    WHERE v.id IS NOT NULL OR f.id IS NOT NULL
    ORDER BY 4 DESC
    LIMIT match_limit;
$$;
