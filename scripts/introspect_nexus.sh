#!/usr/bin/env bash
# Print the real signatures and return columns of the two Nexus search functions,
# and confirm what `recall_search` is actually allowed to do.
#
# The parameter names in sql/search_personal.sql and sql/search_mounted.sql use
# named notation, so they must match the deployed signature exactly. Run this after
# any Nexus migration and correct those files if the output disagrees.
#
#   DATABASE_URL='postgresql://recall_search:...@...pooler.supabase.com:5432/postgres?sslmode=require' \
#     scripts/introspect_nexus.sh
set -uo pipefail

: "${DATABASE_URL:?set DATABASE_URL to the session pooler URL for role recall_search}"

echo "== connection identity =="
psql "$DATABASE_URL" -X -c "select current_user, current_database(), version();" 2>&1 | head -6

echo
echo "== function signatures =="
psql "$DATABASE_URL" -X -c "
select p.proname,
       pg_get_function_arguments(p.oid) as arguments,
       pg_get_function_result(p.oid)    as returns
from pg_proc p
join pg_namespace n on n.oid = p.pronamespace
where p.proname in ('hybrid_search_brain','search_mounted_for_brain')
order by p.proname;" 2>&1

echo
echo "== execute privileges for current_user =="
psql "$DATABASE_URL" -X -c "
select p.proname,
       has_function_privilege(current_user, p.oid, 'EXECUTE') as can_execute
from pg_proc p
where p.proname in ('hybrid_search_brain','search_mounted_for_brain')
order by p.proname;" 2>&1

echo
echo "== table reads should be denied =="
psql "$DATABASE_URL" -X -c "select count(*) from public.memories;" 2>&1 | head -3
echo "  (an error above is the expected and correct result)"

echo
echo "== embedding dimension in use =="
psql "$DATABASE_URL" -X -c "
select a.atttypmod as declared_dim
from pg_attribute a
join pg_class c on c.oid = a.attrelid
where a.attname = 'embedding' and c.relkind = 'r'
limit 5;" 2>&1 | head -8
echo "  (may be denied under recall_search; confirm 1536 with Nexus if so)"
