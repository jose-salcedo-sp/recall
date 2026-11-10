-- Legacy connector chunks are no longer retrieved; traces can still reference them by id.
UPDATE chunks SET superseded_at = now() WHERE superseded_at IS NULL;

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_check;
ALTER TABLE documents ADD CONSTRAINT documents_source_check
    CHECK (source IN ('confluence', 'jira', 'slack', 'markdown'));

DROP TABLE IF EXISTS ingest_cursors;
