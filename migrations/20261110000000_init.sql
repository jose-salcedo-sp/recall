CREATE EXTENSION IF NOT EXISTS vector;

-- One source unit, as a `Source` connector returns it.
CREATE TABLE documents (
    id                TEXT PRIMARY KEY,  -- {source}:{external_id}
    source            TEXT NOT NULL CHECK (source IN ('markdown')),
    external_id       TEXT NOT NULL,
    title             TEXT NOT NULL,
    url               TEXT NOT NULL,
    source_updated_at TIMESTAMPTZ NOT NULL,
    last_ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, external_id)
);

-- Immutable, content-addressed chunk versions. Re-ingestion inserts a new version and supersedes the old one.
CREATE TABLE chunks (
    id                      TEXT NOT NULL,  -- {document_id}#{locator}
    version                 TEXT NOT NULL,  -- hex SHA-256, see ChunkVersion::compute
    document_id             TEXT NOT NULL REFERENCES documents (id),
    ordinal                 INT NOT NULL,   -- position within the document
    heading                 TEXT,
    url                     TEXT NOT NULL,
    text                    TEXT NOT NULL,
    authoritative           BOOLEAN NOT NULL,
    acl                     TEXT[] NOT NULL,
    tenant                  TEXT,
    sensitivity             TEXT NOT NULL CHECK (sensitivity IN ('public', 'internal', 'confidential')),
    embedding               vector(1024) NOT NULL,
    embedding_model         TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_at           TIMESTAMPTZ,
    PRIMARY KEY (id, version)
);

CREATE UNIQUE INDEX chunks_one_current_version ON chunks (id) WHERE superseded_at IS NULL;
CREATE INDEX chunks_current_by_document ON chunks (document_id, ordinal) WHERE superseded_at IS NULL;

-- Every run: answered, declined, and failed. `trace` is the audit record: each step's model, prompt version and
-- score, every admission screen and every claim verdict.
CREATE TABLE pipeline_runs (
    id         UUID PRIMARY KEY,
    user_id    TEXT NOT NULL,
    question   TEXT NOT NULL,
    category   TEXT CHECK (category IN ('boolean', 'choice', 'ranking', 'open')),
    status     TEXT NOT NULL CHECK (status IN ('answered', 'declined', 'failed')),
    outcome    JSONB NOT NULL,  -- pipeline::report::Outcome
    trace      JSONB NOT NULL,  -- pipeline::trace::Trace
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX pipeline_runs_by_user ON pipeline_runs (user_id, created_at DESC);

-- Prose full text: English stemming over the heading (weight A) and body (weight B).
ALTER TABLE chunks ADD COLUMN fts_prose tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(heading, '')), 'A')
    || setweight(to_tsvector('english', text), 'B')
) STORED;

-- Exact tokens: every lowercased run of [a-z0-9_], unsplit and unstemmed, so identifiers survive intact (the
-- default parser splits A_OPCODE at '_' and drops the stop word 'a'; English stems ROLE_USERS to role + user).
-- Underscores at token edges are separators, so markdown `_ROLE_USERS_` indexes as role_users. Runs of 64+
-- characters are dropped: they are never identifiers and tsvector rejects lexemes over 2046 bytes.
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

-- BM25 corpus statistics over current chunks' fts_prose, as of the last refresh. `recall ingest` refreshes them
-- after every run; retrieval reads them in its own snapshot, so a query sees one consistent refresh.

-- Document frequency: how many current chunks contain each lexeme.
CREATE TABLE fts_prose_terms (
    lexeme TEXT PRIMARY KEY,
    chunks INT NOT NULL CHECK (chunks > 0)
);

-- One row. A chunk's length is its lexeme occurrences: the summed position counts of its fts_prose.
CREATE TABLE fts_prose_corpus (
    singleton    BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
    chunks       BIGINT NOT NULL CHECK (chunks >= 0),
    avg_length   DOUBLE PRECISION CHECK (avg_length >= 0),  -- NULL when there are no current chunks
    refreshed_at TIMESTAMPTZ NOT NULL
);

-- The lock serializes concurrent refreshes without blocking readers, who see the old rows until commit.
CREATE FUNCTION refresh_fts_prose_stats() RETURNS void LANGUAGE sql AS $$
    LOCK TABLE fts_prose_terms, fts_prose_corpus IN SHARE ROW EXCLUSIVE MODE;
    DELETE FROM fts_prose_terms;
    INSERT INTO fts_prose_terms (lexeme, chunks)
        SELECT word, ndoc FROM ts_stat('SELECT fts_prose FROM chunks WHERE superseded_at IS NULL');
    INSERT INTO fts_prose_corpus (chunks, avg_length, refreshed_at)
        SELECT count(*), avg(l.length), now()
        FROM chunks c
        CROSS JOIN LATERAL (
            SELECT coalesce(sum(coalesce(array_length(positions, 1), 1)), 0) AS length FROM unnest(c.fts_prose)
        ) l
        WHERE c.superseded_at IS NULL
    ON CONFLICT (singleton) DO UPDATE SET
        chunks = EXCLUDED.chunks, avg_length = EXCLUDED.avg_length, refreshed_at = EXCLUDED.refreshed_at;
$$;

SELECT refresh_fts_prose_stats();
