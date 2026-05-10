#!/usr/bin/env bash
# smoke-test.sh — verify a running ds4-vllm instance produces correct output
#
# Usage:
#   scripts/smoke-test.sh [--port 8000] [--model dsv4] [--host localhost]
#                         [--strict] [--verbose] [--no-color]
#
# Exit codes:
#   0  — first-token check PASSED (response begins with " We", matching the
#        ds4 reference for "The capital of France is")
#   1  — health check failed or response did not parse
#   2  — first-token check FAILED (server reachable but output is wrong)
#
# Background:
#   The canonical first-token check uses a 5-token prompt that the ds4 C+Metal
#   reference produces deterministic output for. With temperature=0, vLLM
#   should emit token 2581 (' We') first. If that fails, the most likely
#   cause is a missing  VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0  on SM121
#   (the default Triton compressed-decode kernel emits one correct token then
#   degenerates on consumer Blackwell).
#
#   In --strict mode we also check that the first 5 tokens match
#   " We are asked" exactly — this catches subtler regressions that produce
#   plausible-but-wrong text (e.g. " We have a question").

set -euo pipefail

PORT=8000
HOST=localhost
MODEL=dsv4
STRICT=0
VERBOSE=0
NO_COLOR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     PORT="$2"; shift 2 ;;
    --host)     HOST="$2"; shift 2 ;;
    --model)    MODEL="$2"; shift 2 ;;
    --strict)   STRICT=1; shift ;;
    --verbose)  VERBOSE=1; shift ;;
    --no-color) NO_COLOR=1; shift ;;
    -h|--help)
      awk '/^#!/{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "$NO_COLOR" -eq 1 ]] || ! [[ -t 1 ]]; then
  C_OK=""; C_ERR=""; C_INFO=""; C_DIM=""; C_END=""
else
  C_OK=$'\033[1;32m'; C_ERR=$'\033[1;31m'
  C_INFO=$'\033[1;36m'; C_DIM=$'\033[2m'; C_END=$'\033[0m'
fi

say()   { printf '%s[smoke]%s %s\n'    "$C_INFO" "$C_END" "$*"; }
ok()    { printf '%s[smoke] ✓%s %s\n'  "$C_OK"   "$C_END" "$*"; }
fail()  { printf '%s[smoke] ✗%s %s\n'  "$C_ERR"  "$C_END" "$*" >&2; }
debug() { [[ "$VERBOSE" -eq 1 ]] && printf '%s    %s%s\n' "$C_DIM" "$*" "$C_END" >&2 || true; }

BASE="http://$HOST:$PORT"
HEALTH_URL="$BASE/health"
COMPLETIONS_URL="$BASE/v1/completions"

# --- 1. health check ------------------------------------------------------
say "checking $HEALTH_URL"
if ! curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
  fail "vLLM health endpoint unreachable at $HEALTH_URL"
  fail "is the container running?  docker ps --filter name=vllm-ds4"
  exit 1
fi
ok "health endpoint reachable"

# --- 2. first-token check -------------------------------------------------
say "POST $COMPLETIONS_URL  prompt='The capital of France is'  max_tokens=5  temp=0"
REQ_BODY="$(cat <<JSON
{"model":"$MODEL","prompt":"The capital of France is","max_tokens":5,"temperature":0}
JSON
)"

RESP="$(curl -fsS --max-time 60 -H 'Content-Type: application/json' -d "$REQ_BODY" "$COMPLETIONS_URL" 2>&1 || true)"
debug "raw response: $RESP"

TEXT="$(printf '%s' "$RESP" | python3 -c '
import sys, json
try:
    j = json.loads(sys.stdin.read())
    print(j["choices"][0]["text"], end="")
except Exception as e:
    print(f"<<parse-error: {e}>>", end="")
' 2>/dev/null)"

if [[ "$TEXT" == "<<parse-error:"* || -z "$TEXT" ]]; then
  fail "could not parse completion response"
  fail "raw: ${RESP:0:300}"
  exit 1
fi

debug "completion text: |$TEXT|"

# --- 3. validate ----------------------------------------------------------
# The ds4 reference produces exactly: " We are asked: \""  for the canonical prompt.
# In --strict mode we check the full 5-token text. Otherwise we only require
# the first token (" We").
EXPECTED_PREFIX=' We'
EXPECTED_FULL=' We are asked'

if [[ "$STRICT" -eq 1 ]]; then
  if [[ "$TEXT" == "$EXPECTED_FULL"* ]]; then
    ok "STRICT PASS — response='$TEXT'"
    exit 0
  else
    fail "STRICT FAIL — expected text starting with: $EXPECTED_FULL"
    fail "got: '$TEXT'"
    exit 2
  fi
fi

if [[ "$TEXT" == "$EXPECTED_PREFIX"* ]]; then
  ok "PASS — first-token check  response='$TEXT'"
  if [[ "$TEXT" == "$EXPECTED_FULL"* ]]; then
    ok "       (matches ds4 reference exactly through first 5 tokens)"
  else
    debug "first token matches; later tokens diverged from reference (often fine)"
  fi
  exit 0
fi

fail "FAIL — response does NOT start with '$EXPECTED_PREFIX'"
fail "got: '$TEXT'"
fail ""
fail "expected ' We are asked: \"' (token 2581 first). The most likely causes are:"
fail "  1. VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 not set in the container"
fail "     check:  docker inspect vllm-ds4 | grep VLLM_TRITON"
fail "  2. wrong model loaded (--model arg here is '$MODEL'; verify with /v1/models)"
fail "  3. checkpoint corrupted; verify SHA256SUMS in the model dir"
exit 2
