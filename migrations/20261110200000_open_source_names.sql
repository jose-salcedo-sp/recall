-- Source names are open: each `Source` connector names itself (`domain::SourceName`), including connectors outside
-- this crate, so the column checks the name's format instead of listing the connectors. Every stored name still
-- parses back into a `SourceName`; existing 'markdown' rows pass.
ALTER TABLE documents DROP CONSTRAINT documents_source_check;
ALTER TABLE documents ADD CONSTRAINT documents_source_name CHECK (source ~ '^[a-z0-9_-]+$');
