-- ACL entries are tagged (`access::AclEntry`): `user:<id>` names one user, `role:<name>` every holder of a role.
-- Untagged entries matched a user id or a role name alike, so a user whose id was `hr` read chunks ACL'd to role
-- `hr`. An untagged entry that is a valid role name becomes `role:`, anything else `user:`: each tagged entry admits
-- a subset of whom the untagged one did, so the rewrite never widens access. Chunk versions are left as they were;
-- the next ingestion re-versions the chunks whose labels now hash differently.
UPDATE chunks
SET acl = ARRAY(
    SELECT CASE WHEN entry ~ '^[a-z0-9_.:-]+$' THEN 'role:' || entry ELSE 'user:' || entry END
    FROM unnest(acl) WITH ORDINALITY AS t(entry, position)
    ORDER BY position
)
WHERE cardinality(acl) > 0;

-- The same patterns as `UserId::new` and `Role::new`.
CREATE FUNCTION acl_entries_are_tagged(acl TEXT[]) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(bool_and(entry ~ '^(user:[^[:cntrl:]]+|role:[a-z0-9_.:-]+)$'), true) FROM unnest(acl) AS entry
$$;

ALTER TABLE chunks ADD CONSTRAINT chunks_acl_tagged CHECK (acl_entries_are_tagged(acl));

-- Stored traces record each run's chunk scope (`access::ChunkScope`), whose members were the asker's id and role
-- names, untagged. The entry equal to the run's user becomes `user:`, the rest `role:`. A run whose user id was
-- also one of its role names keeps the user entry only.
UPDATE pipeline_runs r
SET trace = (
    SELECT jsonb_agg(
        CASE
            WHEN jsonb_typeof(e #> '{record,access,chunks,members}') = 'array' THEN jsonb_set(
                e,
                '{record,access,chunks,members}',
                coalesce((
                    SELECT jsonb_agg(DISTINCT CASE WHEN m = r.user_id THEN 'user:' || m ELSE 'role:' || m END)
                    FROM jsonb_array_elements_text(e #> '{record,access,chunks,members}') AS m
                ), '[]'::jsonb)
            )
            ELSE e
        END
        ORDER BY i
    )
    FROM jsonb_array_elements(r.trace) WITH ORDINALITY AS x(e, i)
)
WHERE jsonb_typeof(trace) = 'array' AND jsonb_array_length(trace) > 0;
