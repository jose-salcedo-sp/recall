-- sqlx migration: persist kind, filter routes, verify verdicts.

ALTER TABLE asks ADD COLUMN IF NOT EXISTS kind text;
ALTER TABLE asks ADD COLUMN IF NOT EXISTS kind_confidence double precision;
ALTER TABLE asks ADD COLUMN IF NOT EXISTS verdicts jsonb NOT NULL DEFAULT '[]';
