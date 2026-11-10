-- Prose full text: English stemming over the heading (weight A) and body (weight B).
ALTER TABLE chunks ADD COLUMN fts_prose tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(heading, '')), 'A')
    || setweight(to_tsvector('english', text), 'B')
) STORED;

-- Exact tokens: every lowercased run of [a-z0-9_], unsplit and unstemmed, so SAP identifiers survive intact
-- (the default parser splits S_TCODE at '_' and drops the stop word 's'; English stems AGR_USERS to agr + user).
-- Underscores at token edges are separators, so markdown `_AGR_USERS_` indexes as agr_users.
-- Runs of 64+ characters are dropped: they are never identifiers and tsvector rejects lexemes over 2046 bytes.
-- `retrieval::hybrid::identifier_terms` tokenizes queries the same way.
ALTER TABLE chunks ADD COLUMN fts_tokens tsvector GENERATED ALWAYS AS (
    array_to_tsvector(array_remove(regexp_split_to_array(
        regexp_replace(lower(coalesce(heading, '') || ' ' || text), '[a-z0-9_]{64,}', ' ', 'g'),
        '_*[^a-z0-9_]+_*|^_+|_+$'
    ), ''))
) STORED;

CREATE INDEX chunks_current_fts_prose ON chunks USING gin (fts_prose) WHERE superseded_at IS NULL;
CREATE INDEX chunks_current_fts_tokens ON chunks USING gin (fts_tokens) WHERE superseded_at IS NULL;
CREATE INDEX chunks_current_embedding ON chunks USING hnsw (embedding vector_cosine_ops) WHERE superseded_at IS NULL;

-- Lets the retriever read the first and last current embedding model in two index probes.
CREATE INDEX chunks_current_embedding_model ON chunks (embedding_model, embedding_model_version)
    WHERE superseded_at IS NULL;
