-- `status` copied the `status` tag of `outcome`, written separately by the run store, so the two could disagree.
-- The column is now derived from `outcome`, with the same values and check.
ALTER TABLE pipeline_runs DROP COLUMN status;
ALTER TABLE pipeline_runs ADD COLUMN status TEXT NOT NULL
    GENERATED ALWAYS AS (outcome ->> 'status') STORED
    CHECK (status IN ('answered', 'declined', 'failed'));
