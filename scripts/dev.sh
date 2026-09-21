#!/usr/bin/env bash
# Run the service on the host against either corpus.
#
#   scripts/dev.sh local           # local Compose corpus (postgres on :5433)
#   scripts/dev.sh nexus           # Nexus Supabase, read-only
#
# .env holds Compose-internal hostnames (systemone, generator) and the Nexus
# DATABASE_URL, neither of which is right for a host-side run, so the overrides
# below exist rather than a second env file to keep in sync.
#
# Logs are JSON on stdout and tee'd to /tmp/recall.jsonl. In a second terminal:
#   recall-dash /tmp/recall.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-local}"
set -a; . ./.env; set +a

export SYSTEM_ONE_URL=http://localhost:8082
export GENERATOR_URL=http://localhost:8080
# Loopback by default: Recall authenticates callers but has no user-level authz,
# so anyone who can reach it can name any brain_id. Override only when the caller
# is off-host (a container, another machine):
#   BIND_ADDR=0.0.0.0:8000 scripts/dev.sh nexus
export BIND_ADDR="${BIND_ADDR:-127.0.0.1:${RECALL_PORT:-8000}}"
export LOG_FORMAT=json
export EMBEDDING_DIM=1536
export ASK_TIMEOUT_MS=300000
export ADMIT_TIMEOUT_MS=280000

case "$MODE" in
  local)
    # Recall owns this database, so migrations, index writes and ask persistence
    # are all on. The local Postgres has no TLS.
    export DATABASE_URL='postgresql://recall:recall@localhost:5433/recall?sslmode=disable'
    export ALLOW_INSECURE_DB=true INDEX_WRITES_ENABLED=true PERSIST_ASKS=true
    ;;
  nexus)
    # DATABASE_URL comes from .env. Everything that writes stays off: recall_search
    # has no table privileges and Recall must not create any.
    export INDEX_WRITES_ENABLED=false PERSIST_ASKS=false
    ;;
  *) echo "usage: $0 [local|nexus]" >&2; exit 2 ;;
esac

[ -n "${RECALL_SERVICE_TOKEN:-}" ] || { echo "RECALL_SERVICE_TOKEN missing from .env" >&2; exit 2; }
[ -n "${OPENAI_API_KEY:-}${EMBEDDER_API_KEY:-}" ] || { echo "no embedder key in .env" >&2; exit 2; }

echo "recall [$MODE] on $BIND_ADDR — logs also at /tmp/recall.jsonl" >&2
exec ./target/debug/recall 2>&1 | tee /tmp/recall.jsonl
