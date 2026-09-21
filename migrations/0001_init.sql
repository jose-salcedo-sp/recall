-- sqlx migration: initial Recall schema (pgvector + FTS + ask audit).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE chunks (
    id uuid PRIMARY KEY,
    brain_id uuid NOT NULL,
    text text NOT NULL,
    statement text NOT NULL,
    embedding vector(1536) NOT NULL,
    origin text NOT NULL CHECK (origin IN ('personal', 'granted')),
    grantor_brain_id uuid NULL,
    grantor_name text NULL,
    sensitivity text NOT NULL DEFAULT 'normal' CHECK (sensitivity IN ('normal', 'sensitive')),
    valid_from timestamptz NULL,
    valid_to timestamptz NULL,
    state text NOT NULL DEFAULT 'active' CHECK (state = 'active'),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(statement, '') || ' ' || coalesce(text, ''))
    ) STORED
);

CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_brain_id_idx ON chunks (brain_id);

CREATE TABLE asks (
    ask_id uuid PRIMARY KEY,
    brain_id uuid NOT NULL,
    trace_id uuid NULL,
    question text NOT NULL,
    as_of timestamptz NULL,
    candidates jsonb NOT NULL DEFAULT '[]',
    admitted_ids jsonb NOT NULL DEFAULT '[]',
    stages jsonb NOT NULL DEFAULT '[]',
    empty boolean NOT NULL DEFAULT false,
    answer text NULL,
    error text NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
