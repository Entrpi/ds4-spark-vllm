#!/usr/bin/env bash
# Publish a converted DSv4-Flash 2-bit checkpoint to Hugging Face.
#
# Idempotent. Safe to re-run: hf upload-large-folder diffs against the hub
# and only uploads what's missing or changed.
#
# Usage:
#   scripts/upload-checkpoint.sh [--repo USER/NAME] [--model-dir PATH]
#                                [--meta-dir PATH]   [--private] [--dry-run]
#
# Defaults:
#   --repo       Entrpi/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8
#   --model-dir  /home/ent/models/deepseek-v4-flash-ds4-q2
#   --meta-dir   <repo>/local/distribution
#   --private    (off — public by default)
#   --dry-run    (off — actually uploads)
#
# Requirements (verified at startup):
#   * `hf` CLI from huggingface_hub >= 1.7      (`hf --version`)
#   * `hf_xet` installed for fast Xet uploads   (`pip install -U "huggingface_hub[cli,hf_xet]"`)
#   * Write token already saved                 (`hf auth login`)
#
# What this script does:
#   1. Sanity-check the model dir + metadata dir.
#   2. Generate SHA256SUMS over every file we will upload (slow: hashes
#      ~85 GiB at ~500 MB/s = ~3 min on Spark NVMe).
#   3. Stage README.md, LICENSE, and SHA256SUMS into a single small dir.
#   4. Create the HF repo if missing.
#   5. `hf upload-large-folder` the model dir (excluding *.bak).
#   6. `hf upload-large-folder` the metadata staging dir.
#   7. Print the final HF URL.

set -euo pipefail

# --- defaults --------------------------------------------------------------
HF_REPO="${HF_REPO:-Entrpi/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8}"
MODEL_DIR="${MODEL_DIR:-/home/ent/models/deepseek-v4-flash-ds4-q2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
META_DIR="${META_DIR:-$REPO_ROOT/local/distribution}"
PRIVATE=0
DRY_RUN=0
NUM_WORKERS="${NUM_WORKERS:-8}"

usage() {
  sed -n 's/^# \{0,1\}//p' "$0" | sed -n '2,/^$/p'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)      HF_REPO="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --meta-dir)  META_DIR="$2"; shift 2 ;;
    --private)   PRIVATE=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --workers)   NUM_WORKERS="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

# --- 1. preflight ----------------------------------------------------------
say() { printf '\033[1;36m[upload-ckpt]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[upload-ckpt ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# Try to put hf on PATH if it lives in the user's local bin.
if ! command -v hf >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/hf" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
command -v hf >/dev/null 2>&1 || err "hf CLI not found. Run: pip install --user -U 'huggingface_hub[cli,hf_xet]'"

HF_VERSION="$(hf --version 2>&1 || true)"
say "hf CLI: $HF_VERSION"
hf auth whoami >/dev/null 2>&1 || err "Not logged in. Run: hf auth login"
WHOAMI="$(hf auth whoami 2>&1 | head -1)"
say "authenticated as: $WHOAMI"

[[ -d "$MODEL_DIR" ]] || err "model dir not found: $MODEL_DIR"
[[ -d "$META_DIR"  ]] || err "metadata dir not found: $META_DIR (expected README.md and LICENSE here)"
[[ -f "$META_DIR/README.md" ]] || err "missing $META_DIR/README.md"
[[ -f "$META_DIR/LICENSE"   ]] || err "missing $META_DIR/LICENSE"

# Check the model dir has the expected pieces.
required_in_model_dir=( config.json tokenizer.json tokenizer_config.json
                        generation_config.json model.safetensors.index.json )
for f in "${required_in_model_dir[@]}"; do
  [[ -f "$MODEL_DIR/$f" ]] || err "missing required file: $MODEL_DIR/$f"
done
n_shards="$(ls -1 "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')"
[[ "$n_shards" -ge 1 ]] || err "no safetensors shards found in $MODEL_DIR"
say "model dir: $MODEL_DIR ($n_shards shards)"

# Warn if there are stray files we'd rather not ship.
strays="$(ls -1 "$MODEL_DIR" | grep -E '\.(bak|swp|tmp|log)$' || true)"
if [[ -n "$strays" ]]; then
  say "skipping stray files (won't be uploaded):"
  while IFS= read -r f; do say "  - $f"; done <<< "$strays"
fi

# --- 2. generate SHA256SUMS ------------------------------------------------
say "computing SHA256SUMS over $MODEL_DIR + metadata (this takes a few min)..."
SHA_FILE="$META_DIR/SHA256SUMS"

# Capture all files we will upload (model + metadata), hashed under the
# repo-relative path they will end up at.
(
  cd "$MODEL_DIR"
  # exclude bak/swp/tmp/log; sort for reproducibility
  find . -maxdepth 1 -type f \
    \! -name '*.bak' \! -name '*.swp' \! -name '*.tmp' \! -name '*.log' \
    -printf '%P\n' | sort | xargs -I{} sha256sum -b "{}"
) > "$SHA_FILE.tmp"

(
  cd "$META_DIR"
  for f in README.md LICENSE; do
    sha256sum -b "$f"
  done
) >> "$SHA_FILE.tmp"

mv "$SHA_FILE.tmp" "$SHA_FILE"
say "wrote $SHA_FILE ($(wc -l < "$SHA_FILE") entries)"

# --- 3. create repo (idempotent) -------------------------------------------
PRIVACY_FLAG=()
if [[ "$PRIVATE" -eq 1 ]]; then
  PRIVACY_FLAG=(--private)
  say "creating PRIVATE repo $HF_REPO (if missing)..."
else
  say "creating PUBLIC repo $HF_REPO (if missing)..."
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "[dry-run] would: hf repo create $HF_REPO --repo-type model ${PRIVACY_FLAG[*]} --exist-ok"
else
  hf repo create "$HF_REPO" --repo-type model "${PRIVACY_FLAG[@]}" --exist-ok
fi

# --- 4. upload model dir (large) -------------------------------------------
say "uploading model dir → $HF_REPO  (workers=$NUM_WORKERS, exclude=*.bak)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  say "[dry-run] would: hf upload-large-folder $HF_REPO $MODEL_DIR --repo-type model --num-workers $NUM_WORKERS --exclude '*.bak'"
else
  hf upload-large-folder \
    "$HF_REPO" \
    "$MODEL_DIR" \
    --repo-type model \
    --num-workers "$NUM_WORKERS" \
    --exclude '*.bak' \
    --exclude '*.swp' \
    --exclude '*.tmp' \
    --exclude '*.log'
fi

# --- 5. upload metadata dir -------------------------------------------------
say "uploading metadata (README.md, LICENSE, SHA256SUMS) → $HF_REPO"
if [[ "$DRY_RUN" -eq 1 ]]; then
  say "[dry-run] would: hf upload-large-folder $HF_REPO $META_DIR --repo-type model"
else
  # Only upload the three files we curate; if META_DIR has extras, skip them.
  TMP_META="$(mktemp -d)"
  trap 'rm -rf "$TMP_META"' EXIT
  cp "$META_DIR/README.md"   "$TMP_META/README.md"
  cp "$META_DIR/LICENSE"     "$TMP_META/LICENSE"
  cp "$META_DIR/SHA256SUMS"  "$TMP_META/SHA256SUMS"
  hf upload-large-folder \
    "$HF_REPO" \
    "$TMP_META" \
    --repo-type model \
    --num-workers 2
fi

# --- 6. done ----------------------------------------------------------------
say "✓ upload complete"
say "https://huggingface.co/$HF_REPO"
