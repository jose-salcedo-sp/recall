#!/usr/bin/env bash
# End-to-end check against the local stack.
#
# Starts the recall binary, exercises the ask paths, and asserts the behaviour the
# product depends on: a real question gets cited from the corpus, an unanswerable one
# returns an empty admit with no generator call, and the audit record keeps the
# rejected candidates that calibration will later be fitted on.
set -uo pipefail
cd "$(dirname "$0")/.."

BRAIN=00000000-0000-0000-0000-000000000001
BASE=http://localhost:8000
pass=0; fail=0
check() { if [ "$1" = "1" ]; then echo "  PASS  $2"; pass=$((pass+1)); else echo "  FAIL  $2"; fail=$((fail+1)); fi; }

export DATABASE_URL=${DATABASE_URL:-postgresql://recall:recall@localhost:5433/recall}
export EMBEDDER_URL=${EMBEDDER_URL:-http://localhost:8081}
export SYSTEM_ONE_URL=${SYSTEM_ONE_URL:-http://localhost:8082}
export GENERATOR_URL=${GENERATOR_URL:-http://localhost:8080}
export BIND_ADDR=${BIND_ADDR:-127.0.0.1:8000}
export ADMIT_THRESHOLD=${ADMIT_THRESHOLD:-0.15}
export ASK_TIMEOUT_MS=${ASK_TIMEOUT_MS:-180000}
export ADMIT_TIMEOUT_MS=${ADMIT_TIMEOUT_MS:-150000}
export RUST_LOG=${RUST_LOG:-info,recall=debug}

pkill -f "target/debug/recall" 2>/dev/null; sleep 1
./target/debug/recall > /tmp/recall-e2e.log 2>&1 &
SVC=$!
trap 'kill $SVC 2>/dev/null' EXIT

for _ in $(seq 1 30); do
  curl -sf -m 2 "$BASE/healthz" >/dev/null 2>&1 && break
  sleep 1
done

echo "== readyz =="
READY=$(curl -s -m 60 "$BASE/readyz")
echo "$READY" | head -c 300; echo
[ -n "$READY" ] && echo "$READY" | grep -q '"ok":true' && check 1 "all dependencies ready" || check 0 "all dependencies ready"

ask() {
  curl -s -m 300 "$BASE/v1/ask/sync" -H 'Content-Type: application/json' \
    -d "{\"question\":$1,\"brain_id\":\"$BRAIN\"}"
}

echo
echo "== answerable question =="
R1=$(ask '"When is my sister Ana'"'"'s birthday?"')
echo "$R1" | head -c 600; echo
echo "$R1" | grep -q '"empty":false' && check 1 "answered rather than refused" || check 0 "answered rather than refused"
echo "$R1" | grep -qi 'march' && check 1 "answer contains the fact from the corpus" || check 0 "answer contains the fact from the corpus"
echo "$R1" | grep -q '"citations":\[{' && check 1 "returned at least one citation" || check 0 "returned at least one citation"

echo
echo "== unanswerable question (empty admit) =="
R2=$(ask '"What is the atomic mass of ruthenium?"')
echo "$R2" | head -c 400; echo
echo "$R2" | grep -q '"empty":true' && check 1 "empty admit on an unanswerable question" || check 0 "empty admit on an unanswerable question"
echo "$R2" | grep -q '"citations":\[\]' && check 1 "no citations when nothing is admitted" || check 0 "no citations when nothing is admitted"
echo "$R2" | grep -q 'I do not have that' && check 1 "fixed refusal string, not generated prose" || check 0 "fixed refusal string, not generated prose"
grep -q 'empty admission; skipping generator' /tmp/recall-e2e.log \
  && check 1 "generator was never called for the empty admit" \
  || check 0 "generator was never called for the empty admit"

echo
echo "== audit record =="
AID=$(echo "$R1" | sed -n 's/.*"ask_id":"\([^"]*\)".*/\1/p')
AUD=$(curl -s -m 30 "$BASE/v1/asks/$AID")
echo "$AUD" | head -c 400; echo
echo "$AUD" | grep -q '"admitted":false' && check 1 "rejected candidates kept for calibration" || check 0 "rejected candidates kept for calibration"
echo "$AUD" | grep -q '"stage":"admit"' && check 1 "per-stage timings recorded" || check 0 "per-stage timings recorded"

echo
echo "== index invariants =="
bad() { curl -s -o /dev/null -w '%{http_code}' -m 20 -X PUT "$BASE/v1/index/chunks/$1" \
  -H 'Content-Type: application/json' -d "$2"; }
C=$(bad 11111111-1111-1111-1111-111111111111 "{\"brain_id\":\"$BRAIN\",\"text\":\"x\",\"statement\":\"x\",\"origin\":\"personal\",\"sensitivity\":\"secret\"}")
[ "$C" = "400" ] && check 1 "rejects sensitivity=secret ($C)" || check 0 "rejects sensitivity=secret (got $C)"
C=$(bad 11111111-1111-1111-1111-111111111112 "{\"brain_id\":\"$BRAIN\",\"text\":\"x\",\"statement\":\"x\",\"origin\":\"personal\",\"state\":\"pending\"}")
[ "$C" = "400" ] && check 1 "rejects state=pending ($C)" || check 0 "rejects state=pending (got $C)"

echo
echo "== SSE stream =="
SSE=$(curl -s -N -m 300 -D /tmp/sse-headers.txt "$BASE/v1/ask" -H 'Content-Type: application/json' \
  -d "{\"question\":\"What am I allergic to?\",\"brain_id\":\"$BRAIN\"}")
echo "$SSE" | head -c 500; echo
grep -qi 'x-accel-buffering: no' /tmp/sse-headers.txt && check 1 "X-Accel-Buffering:no set (Cloudflare will not buffer)" || check 0 "X-Accel-Buffering:no set"
echo "$SSE" | grep -q 'event: stage' && check 1 "stage progress events emitted" || check 0 "stage progress events emitted"
echo "$SSE" | grep -q 'event: admitted' && check 1 "admitted event precedes tokens" || check 0 "admitted event precedes tokens"
echo "$SSE" | grep -q 'event: token' && check 1 "tokens streamed" || check 0 "tokens streamed"
echo "$SSE" | grep -q 'event: done' && check 1 "done event closes the stream" || check 0 "done event closes the stream"

echo
echo "================================"
echo "  passed: $pass   failed: $fail"
echo "================================"
[ "$fail" -eq 0 ]
