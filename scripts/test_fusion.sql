BEGIN;

INSERT INTO chunks (
    id, brain_id, text, statement, embedding, origin, grantor_name, sensitivity, state
) VALUES (
    '00000000-0000-4000-8000-00000000f001',
    '00000000-0000-0000-0000-000000000001',
    'The zirconium-quilt passphrase is quilted-zirc-9.',
    'The zirconium-quilt passphrase is quilted-zirc-9.',
    array_fill(0::real, ARRAY[1536])::vector,
    'granted',
    'Probe',
    'normal',
    'active'
) ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    statement = EXCLUDED.statement,
    origin = EXCLUDED.origin;

DO $$
DECLARE n int;
BEGIN
    SELECT count(*) INTO n
    FROM search_mounted_for_brain(
        '00000000-0000-0000-0000-000000000001'::uuid,
        'zirconium-quilt passphrase',
        array_fill(0::real, ARRAY[1536])::vector,
        now(),
        64
    )
    WHERE memory_statement ILIKE '%quilted-zirc-9%';
    IF n < 1 THEN
        RAISE EXCEPTION 'granted lexical hit dropped (got %)', n;
    END IF;
END $$;

DELETE FROM chunks WHERE id = '00000000-0000-4000-8000-00000000f001';
COMMIT;
