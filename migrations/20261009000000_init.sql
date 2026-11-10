CREATE EXTENSION IF NOT EXISTS vector;

-- One source unit: a Confluence page, a Jira issue, or a Slack thread.
CREATE TABLE documents (
    id                TEXT PRIMARY KEY,  -- {source}:{external_id}
    source            TEXT NOT NULL CHECK (source IN ('confluence', 'jira', 'slack')),
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
    customer                TEXT,
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

CREATE TABLE ingest_cursors (
    source      TEXT PRIMARY KEY CHECK (source IN ('confluence', 'jira', 'slack')),
    last_run_at TIMESTAMPTZ NOT NULL
);

-- Every run: answered, declined, and failed.
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
