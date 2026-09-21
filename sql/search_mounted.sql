-- Memories granted to this brain by others. Grant semantics live entirely inside
-- this function; Recall must not filter or re-derive them.
--
-- Positional binds, in this order:
--   $1  uuid          requester brain_id
--   $2  text          query text
--   $3  vector(1536)  query embedding
--   $4  timestamptz   as_of, nullable
--   $5  int           k
--
-- Parameter names must match the deployed signature exactly; run
-- `scripts/introspect_nexus.sh` to print it.

SELECT *
FROM search_mounted_for_brain(
    p_requester       => $1,
    p_query_text      => $2,
    p_query_embedding => $3,
    p_as_of           => $4,
    p_k               => $5
);
