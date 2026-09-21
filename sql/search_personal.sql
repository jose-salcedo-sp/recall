-- The brain's own memories. Executed as role `recall_search`, which may call this
-- function and read no tables directly.
--
-- Positional binds, in this order:
--   $1  uuid          brain_id
--   $2  text          query text
--   $3  vector(1536)  query embedding (must be the same model Nexus indexed with)
--   $4  timestamptz   as_of, nullable
--   $5  int           k
--
-- Named notation is used so this stays correct if the function's parameter order
-- changes. Parameter names must match the deployed signature exactly; run
-- `scripts/introspect_nexus.sh` to print it.

SELECT *
FROM hybrid_search_brain(
    p_brain_id        => $1,
    p_query_text      => $2,
    p_query_embedding => $3,
    p_as_of           => $4,
    p_k               => $5
);
