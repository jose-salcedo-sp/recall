-- recall is single-tenant: chunks no longer carry a tenant, and grants no longer scope by one. A current chunk with
-- a tenant was readable only through a grant that admitted it; dropping the column would make it readable by every
-- grant that clears its sensitivity, so the migration refuses instead. Re-ingest those sources without tenants first.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM chunks WHERE tenant IS NOT NULL AND superseded_at IS NULL) THEN
        RAISE EXCEPTION 'current chunks carry a tenant; re-ingest their sources without tenants before migrating';
    END IF;
END
$$;

ALTER TABLE chunks DROP COLUMN tenant;
