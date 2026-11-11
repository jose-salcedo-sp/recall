-- An `explain` record carries `contradicts_verdict` only when it explained a judge's verdict
-- (`pipeline::trace::ExplainRecord::Verdict`). Records written before that carried it on the open path too, always
-- false. A run took the verdict path exactly when its trace has an `answer` entry; the flag is removed from every
-- other run's explain records.
UPDATE pipeline_runs r
SET trace = (
    SELECT jsonb_agg(
        CASE
            WHEN e #>> '{record,kind}' = 'explain' THEN e #- '{record,contradicts_verdict}'
            ELSE e
        END
        ORDER BY i
    )
    FROM jsonb_array_elements(r.trace) WITH ORDINALITY AS x(e, i)
)
WHERE jsonb_typeof(trace) = 'array'
  AND jsonb_array_length(trace) > 0
  AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements(r.trace) AS a(e) WHERE a.e ->> 'step' = 'answer');
