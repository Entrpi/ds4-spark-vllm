#!/usr/bin/env bash
# smoke-test.sh — verify a running ds4-vllm instance produces correct output
#
# Usage:
#   scripts/smoke-test.sh [--port 8000] [--model dsv4] [--host localhost]
#                         [--strict] [--verbose] [--no-color]
#
# Exit codes:
#   0  — semantic check PASSED (response contains "Paris", the correct
#        capital of France)
#   1  — health check failed or response did not parse
#   2  — semantic check FAILED (server reachable but output is gibberish
#        or factually wrong)
#
# Background:
#   The default check sends "The capital of France is" with temperature=0
#   and max_tokens=20, then verifies the response contains "Paris" (case-
#   insensitive). This catches outright gibberish AND factual regressions
#   without depending on tokenizer/BOS-token differences between vLLM's
#   /v1/completions API and the ds4 C+Metal reference.
#
#   --strict mode additionally verifies the first 5 tokens match
#   " We are asked" exactly — the ds4-reference output under its specific
#   BOS-token setup. Useful for regression-testing against the bring-up
#   validation, but expect FAIL via the OpenAI completions API path because
#   that path doesn't apply the same BOS prefix (the model goes into
#   direct-answer mode " Paris." instead of CoT-preamble mode).
#
#   If the default check fails, the most likely cause is a missing
#   VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 on SM121 (the default Triton
#   compressed-decode kernel emits one correct token then degenerates on
#   consumer Blackwell — you'd see one valid word followed by garbage).

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

# --- 2. semantic check ----------------------------------------------------
# max_tokens=20 gives the model room to emit "Paris" whether it goes into
# direct-answer mode (" Paris.") or CoT-preamble mode (" We are asked: ...").
say "POST $COMPLETIONS_URL  prompt='The capital of France is'  max_tokens=20  temp=0"
REQ_BODY="$(cat <<JSON
{"model":"$MODEL","prompt":"The capital of France is","max_tokens":20,"temperature":0}
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
# Default check: response must contain "Paris" (case-insensitive). Catches
# gibberish AND factually-wrong output. Robust against tokenizer/BOS
# differences — works whether the model goes into direct-answer mode
# (" Paris.") or CoT-preamble mode (" We are asked: \"...Paris...\"").
#
# --strict additionally verifies the exact ds4-reference 5-token prefix
# (" We are asked"). Note this typically FAILS via the OpenAI completions
# API path because vLLM doesn't apply the same BOS prefix the C+Metal
# reference uses — the model emits direct answers there. --strict is for
# regression-testing against the bring-up validation context specifically.
SEMANTIC_TOKEN='Paris'
DS4_REF_PREFIX=' We are asked'

semantic_pass=0
strict_pass=0

if [[ "${TEXT,,}" == *"${SEMANTIC_TOKEN,,}"* ]]; then
  semantic_pass=1
fi
if [[ "$TEXT" == "$DS4_REF_PREFIX"* ]]; then
  strict_pass=1
fi

if [[ "$STRICT" -eq 1 ]]; then
  if [[ "$strict_pass" -eq 1 ]]; then
    ok "STRICT PASS — response='$TEXT'"
    ok "       (matches ds4 C+Metal reference exactly through first 5 tokens)"
    exit 0
  else
    fail "STRICT FAIL — expected text starting with: '$DS4_REF_PREFIX'"
    fail "got: '$TEXT'"
    if [[ "$semantic_pass" -eq 1 ]]; then
      fail ""
      fail "NOTE: the response IS semantically correct (contains 'Paris'). The"
      fail "model is working — it just isn't producing the ds4-reference's CoT"
      fail "preamble, which depends on a specific BOS-token setup that vLLM's"
      fail "OpenAI /v1/completions API doesn't apply. To validate parity at the"
      fail "tokenizer/BOS layer, use the chat-completions endpoint with a"
      fail "matching system prompt, or run without --strict for a robust check."
    fi
    exit 2
  fi
fi

if [[ "$semantic_pass" -eq 1 ]]; then
  ok "PASS — response contains '$SEMANTIC_TOKEN' (factually correct)"
  ok "       text: '$TEXT'"
  if [[ "$strict_pass" -eq 1 ]]; then
    ok "       (also matches ds4 reference's CoT-preamble prefix exactly)"
  fi
  exit 0
fi

fail "FAIL — response does NOT contain '$SEMANTIC_TOKEN'"
fail "got: '$TEXT'"
fail ""
fail "the model should emit 'Paris' (the capital of France) within 20 tokens"
fail "for this prompt. likely causes:"
fail "  1. VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 not set in the container"
fail "     check:  docker inspect vllm-ds4 | grep VLLM_TRITON"
fail "     symptom: one valid word, then garbage (decode kernel produces wrong"
fail "     output for layers with compress_ratio≥4 on SM121)"
fail "  2. wrong model loaded (--model arg here is '$MODEL'; verify with /v1/models)"
fail "  3. checkpoint corrupted; verify SHA256SUMS in the model dir"
exit 2
