-- Markdown is the only source. Chunks reference documents without ON DELETE CASCADE, so they go first;
-- pipeline_runs traces keep the deleted chunk ids as plain JSON.
DELETE FROM chunks
WHERE document_id IN (SELECT id FROM documents WHERE source <> 'markdown');
DELETE FROM documents WHERE source <> 'markdown';

ALTER TABLE documents DROP CONSTRAINT documents_source_check;
ALTER TABLE documents ADD CONSTRAINT documents_source_check CHECK (source IN ('markdown'));
