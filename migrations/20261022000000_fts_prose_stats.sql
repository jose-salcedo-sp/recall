-- BM25 corpus statistics over current chunks' fts_prose, as of the last refresh. `drew ingest` refreshes them after
-- every run; retrieval reads them in its own snapshot, so a query sees one consistent refresh.

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
