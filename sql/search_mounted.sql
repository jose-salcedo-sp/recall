-- Memories granted to this brain by others. Grant semantics live entirely inside
-- this function; Recall must not filter or re-derive them.
--
-- Deployed signature (verified with scripts/introspect_nexus.sh, 2026-09-21):
--   search_mounted_for_brain(
--     p_requester uuid,
--     query_text text,
--     query_embedding vector,
--     as_of timestamptz DEFAULT now(),
--     match_limit integer DEFAULT 12
--   ) RETURNS TABLE(
--     id uuid, text text, segment_ref text, score double precision,
--     memory_id uuid, memory_statement text,
--     valid_from timestamptz, valid_to timestamptz,
--     origin text, grantor_brain_id uuid, grantor_name text, grantor_face_seed text,
--     org_id uuid, source_id uuid, source_channel text, source_name text,
--     occurred_at timestamptz, sensitivity text
--   )
--
-- Positional binds:
--   $1 uuid  requester  $2 text query   $3 vector(1536) embedding
--   $4 timestamptz as_of (nullable)     $5 int k
--
-- as_of is coalesced here for the same reason as the personal search: binding an
-- explicit NULL would override the now() default rather than fall back to it.

SELECT *
FROM search_mounted_for_brain(
    p_requester     => $1,
    query_text      => $2,
    query_embedding => $3,
    as_of           => coalesce($4::timestamptz, now()),
    match_limit     => $5
);
