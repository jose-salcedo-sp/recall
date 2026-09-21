-- The brain's own memories. Executed as role `recall_search`, which may call this
-- function and read no tables directly.
--
-- Deployed signature (verified with scripts/introspect_nexus.sh, 2026-09-21):
--   hybrid_search_brain(
--     p_brain_id uuid,
--     query_text text,
--     query_embedding vector,
--     as_of timestamptz DEFAULT now(),
--     match_limit integer DEFAULT 12
--   ) RETURNS TABLE(
--     id uuid, text text, segment_ref text, score double precision,
--     memory_id uuid, memory_statement text,
--     valid_from timestamptz, valid_to timestamptz
--   )
--
-- Note this result set has no origin/grantor/source columns; those exist only on
-- search_mounted_for_brain. Everything here is origin = personal by construction.
--
-- Positional binds:
--   $1 uuid  brain_id   $2 text query   $3 vector(1536) embedding
--   $4 timestamptz as_of (nullable)     $5 int k
--
-- as_of is coalesced here rather than passed through as NULL: the parameter
-- defaults to now(), but binding an explicit NULL would override that default with
-- NULL rather than fall back to it.

SELECT *
FROM hybrid_search_brain(
    p_brain_id      => $1,
    query_text      => $2,
    query_embedding => $3,
    as_of           => coalesce($4::timestamptz, now()),
    match_limit     => $5
);
