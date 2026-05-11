#!/usr/bin/env bash
# install.sh — DSv4-Flash 2-bit hybrid for vLLM on DGX Spark (GB10 / SM121)
#
#   curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-spark-vllm/main/install.sh | bash
#   curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-spark-vllm/main/install.sh | bash -s -- --help
#
# What this script does (all steps idempotent — safe to re-run):
#
#   1. Verifies the host is a DGX Spark (or other GB10/SM121 system) with
#      ≥120 GiB RAM and a working Docker. Refuses with a clear error and
#      override flag (--force) on anything else.
#   2. Installs the `hf` CLI from huggingface_hub if missing.
#   3. Authenticates with HuggingFace using --hf-token / $HF_TOKEN, an
#      existing `hf auth` session, or an interactive prompt.
#   4. Downloads the quantized checkpoint (~85 GiB across 17 shards) from
#      bleysg/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target via Xet.
#   5. Verifies SHA256SUMS against on-disk shards.
#   6. Pulls the lmxxf/vllm-deepseek-v4-dgx-spark base image.
#   7. Clones (or pulls) the Entrpi/ds4-spark-vllm overlay into ~/ds4-spark-vllm.
#   8. Starts the vLLM serve container with all correctness-critical flags:
#        - --quantization deepseek_v4_hybrid_iq2  (registered by overlay plugin)
#        - VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 (SM121 decode-kernel workaround)
#        - --kv-cache-dtype fp8, --attention-backend FLASHINFER, --enforce-eager
#          (CUDA graph perf-on path tracked separately; capture fails silently
#          on the current image, see install.sh ENFORCE_EAGER notes)
#   9. Polls /health, then runs a single-token smoke test against the
#      reference prompt ("The capital of France is" → expect text starting
#      with " We"). Prints PASS or FAIL with the actual response.
#  10. Optionally installs a systemd unit so the container restarts on boot.
#
# The script makes NO changes outside:
#   - $MODELS_DIR  (default ~/models)
#   - $WORK_DIR    (default ~/ds4-spark-vllm)
#   - $LOGS_DIR    (default ~/logs)
#   - $EXTRAS_DIR  (default ~/extras, created empty)
#   - the docker container (default name: vllm-ds4)
#   - /etc/systemd/system/ds4-vllm.service  (only if --systemd / opted in)
#
# License: MIT.  Source: https://github.com/Entrpi/ds4-spark-vllm

set -euo pipefail

# ============================================================================
# 0. defaults + flag parsing
# ============================================================================

HF_REPO="${HF_REPO:-bleysg/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
WORK_DIR="${WORK_DIR:-$HOME/ds4-spark-vllm}"
LOGS_DIR="${LOGS_DIR:-$HOME/logs}"
EXTRAS_DIR="${EXTRAS_DIR:-$HOME/extras}"
OVERLAY_REPO="${OVERLAY_REPO:-https://github.com/Entrpi/ds4-spark-vllm.git}"
OVERLAY_REF="${OVERLAY_REF:-main}"
BASE_IMAGE="${BASE_IMAGE:-lmxxf/vllm-deepseek-v4-dgx-spark:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ds4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4}"
PORT="${PORT:-8000}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.86}"
KV_DTYPE="${KV_DTYPE:-fp8}"
ATTN_BACKEND="${ATTN_BACKEND:-FLASHINFER}"
HF_TOKEN="${HF_TOKEN:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Behavior flags
NON_INTERACTIVE=0
FORCE_HW=0
SKIP_DOWNLOAD=0
SKIP_VERIFY=0
SKIP_SMOKE=0
NO_START=0
INSTALL_SYSTEMD=
UNINSTALL=0
# Performance: enforce-eager disables torch.compile + CUDA graphs (a multi-x
# decode penalty). The lmxxf sparse-MLA path advertises cudagraph-safety
# (`triton_sparse_mla_cudagraphs_allowed` returns True when no spec-dec is
# configured), but in practice CUDA graph capture failed silently mid-init
# on Spark — engine worker vanishes with no traceback before the first KV
# cache allocation. Until that's bisected, default ENFORCE_EAGER=1 to ship
# a working server at the (slow) bring-up perf level. The flag and the
# VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1 env var stay wired so the
# perf-on path can be re-attempted with --no-enforce-eager once we find
# the missing piece (mhc_pre / fp8_utils kernel-config cache miss are the
# leading suspects).
ENFORCE_EAGER=1

usage() {
  sed -n 's/^# \{0,1\}//p' "$0" | sed -n '/^install\.sh /,/^License:/p'
  cat <<EOF

Flags (all optional):
  --hf-token TOKEN       HuggingFace access token (also \$HF_TOKEN)
  --models-dir DIR       Where to store model shards (default: $MODELS_DIR)
  --work-dir DIR         Overlay repo checkout (default: $WORK_DIR)
  --logs-dir DIR         Container log dir (default: $LOGS_DIR)
  --extras-dir DIR       Optional extras (DeepGEMM etc.) (default: $EXTRAS_DIR)
  --hf-repo REPO         HF repo to download (default: $HF_REPO)
  --base-image IMAGE     vLLM base image (default: $BASE_IMAGE)
  --container-name NAME  Docker container name (default: $CONTAINER_NAME)
  --port N               Listen port (default: $PORT)
  --max-model-len N      Sequence length cap (default: $MAX_MODEL_LEN)
  --gpu-util F           --gpu-memory-utilization (default: $GPU_UTIL)
  --workers N            hf download parallel workers (default: $NUM_WORKERS)

  --skip-download        Skip model download (assume already present)
  --skip-verify          Skip SHA256SUMS check
  --skip-smoke           Skip the first-token smoke test
  --enforce-eager        Force torch.compile + CUDA graphs off (default ON during bring-up)
  --no-enforce-eager     Try the perf-on path (currently fails silently — see notes)
  --no-start             Set up everything but do not start the container
  --systemd              Install systemd unit non-interactively
  --no-systemd           Skip the systemd prompt
  --uninstall            Stop + remove container, leave models in place
  --non-interactive      Refuse anything that would prompt; fail instead
  --force                Bypass hardware/memory checks
  -h, --help             Show this help

Environment overrides:
  All defaults above can be set via env vars of the same name.

EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)         HF_TOKEN="$2"; shift 2 ;;
    --models-dir)       MODELS_DIR="$2"; shift 2 ;;
    --work-dir)         WORK_DIR="$2"; shift 2 ;;
    --logs-dir)         LOGS_DIR="$2"; shift 2 ;;
    --extras-dir)       EXTRAS_DIR="$2"; shift 2 ;;
    --hf-repo)          HF_REPO="$2"; shift 2 ;;
    --base-image)       BASE_IMAGE="$2"; shift 2 ;;
    --container-name)   CONTAINER_NAME="$2"; shift 2 ;;
    --port)             PORT="$2"; shift 2 ;;
    --max-model-len)    MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-util)         GPU_UTIL="$2"; shift 2 ;;
    --workers)          NUM_WORKERS="$2"; shift 2 ;;
    --skip-download)    SKIP_DOWNLOAD=1; shift ;;
    --skip-verify)      SKIP_VERIFY=1; shift ;;
    --skip-smoke)       SKIP_SMOKE=1; shift ;;
    --enforce-eager)    ENFORCE_EAGER=1; shift ;;
    --no-enforce-eager) ENFORCE_EAGER=0; shift ;;
    --no-start)         NO_START=1; shift ;;
    --systemd)          INSTALL_SYSTEMD=1; shift ;;
    --no-systemd)       INSTALL_SYSTEMD=0; shift ;;
    --uninstall)        UNINSTALL=1; shift ;;
    --non-interactive)  NON_INTERACTIVE=1; shift ;;
    --force)            FORCE_HW=1; shift ;;
    -h|--help)          usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

MODEL_DIR="$MODELS_DIR/$MODEL_DIR_NAME"

# ============================================================================
# log helpers
# ============================================================================

C_INFO=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'
C_ERR=$'\033[1;31m'; C_DIM=$'\033[2m';   C_END=$'\033[0m'

say()  { printf '%s[ds4-install]%s %s\n'   "$C_INFO" "$C_END" "$*"; }
ok()   { printf '%s[ds4-install] ✓%s %s\n' "$C_OK"   "$C_END" "$*"; }
warn() { printf '%s[ds4-install] ⚠%s %s\n' "$C_WARN" "$C_END" "$*" >&2; }
err()  { printf '%s[ds4-install ERROR]%s %s\n' "$C_ERR" "$C_END" "$*" >&2; exit 1; }
hint() { printf '%s   %s%s\n' "$C_DIM" "$*" "$C_END" >&2; }

# Detect if stdin is a TTY (false when piped via curl|bash). Some prompts
# fall back to /dev/tty in that case.
is_tty() { [[ -t 0 ]]; }
have_tty_input() { [[ -r /dev/tty ]]; }

prompt_or_die() {
  local prompt="$1" var_name="$2" default_val="${3:-}" silent="${4:-0}"
  if [[ "$NON_INTERACTIVE" -eq 1 ]]; then
    err "missing $var_name and --non-interactive set; provide via flag or env"
  fi
  if ! have_tty_input; then
    err "missing $var_name and no TTY available (running under curl|bash with closed stdin); re-run as: bash install.sh"
  fi
  local reply
  if [[ -n "$default_val" ]]; then
    printf '%s [%s]: ' "$prompt" "$default_val" >/dev/tty
  else
    printf '%s: ' "$prompt" >/dev/tty
  fi
  if [[ "$silent" -eq 1 ]]; then
    stty -echo </dev/tty
    read -r reply </dev/tty || true
    stty echo </dev/tty
    printf '\n' >/dev/tty
  else
    read -r reply </dev/tty
  fi
  if [[ -z "$reply" && -n "$default_val" ]]; then
    reply="$default_val"
  fi
  printf -v "$var_name" '%s' "$reply"
}

confirm() {
  local prompt="$1" default="${2:-n}"
  if [[ "$NON_INTERACTIVE" -eq 1 ]]; then
    [[ "$default" == "y" ]] && return 0 || return 1
  fi
  if ! have_tty_input; then
    warn "no TTY, defaulting to '$default' for: $prompt"
    [[ "$default" == "y" ]] && return 0 || return 1
  fi
  local reply
  if [[ "$default" == "y" ]]; then
    printf '%s [Y/n]: ' "$prompt" >/dev/tty
  else
    printf '%s [y/N]: ' "$prompt" >/dev/tty
  fi
  read -r reply </dev/tty || reply=""
  reply="${reply:-$default}"
  [[ "${reply,,}" == "y" || "${reply,,}" == "yes" ]]
}

# ============================================================================
# 1. uninstall short-circuit
# ============================================================================

if [[ "$UNINSTALL" -eq 1 ]]; then
  say "uninstalling: stopping and removing container '$CONTAINER_NAME'"
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true
    ok "container removed"
  else
    say "no container named '$CONTAINER_NAME' found"
  fi
  if [[ -f /etc/systemd/system/ds4-vllm.service ]]; then
    if confirm "remove systemd unit /etc/systemd/system/ds4-vllm.service?" "y"; then
      sudo systemctl disable --now ds4-vllm.service 2>/dev/null || true
      sudo rm -f /etc/systemd/system/ds4-vllm.service
      sudo systemctl daemon-reload
      ok "systemd unit removed"
    fi
  fi
  say "model dir preserved: $MODEL_DIR"
  say "  remove with: rm -rf '$MODEL_DIR'"
  exit 0
fi

# ============================================================================
# 2. preflight: hardware, memory, docker, dirs
# ============================================================================

say "DSv4-Flash 2-bit hybrid installer for vLLM"
say "  HF repo:        $HF_REPO"
say "  model dir:      $MODEL_DIR"
say "  base image:     $BASE_IMAGE"
say "  container:      $CONTAINER_NAME (port $PORT, max_model_len=$MAX_MODEL_LEN)"
echo

# --- 2a. CPU arch ---------------------------------------------------------
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64)
    ok "CPU arch: $ARCH (consistent with DGX Spark / Grace)"
    ;;
  x86_64)
    if [[ "$FORCE_HW" -eq 0 ]]; then
      warn "CPU arch: $ARCH — DGX Spark uses aarch64 (Grace)."
      hint "The base image and overlay are built for aarch64. x86_64 may work"
      hint "with H100/B200 if you've prepared an x86_64 base image, but is"
      hint "untested. Re-run with --force to override."
      err "non-aarch64 host without --force"
    else
      warn "CPU arch: $ARCH (forced past check)"
    fi
    ;;
  *)
    [[ "$FORCE_HW" -eq 1 ]] || err "unsupported CPU arch: $ARCH (use --force to override)"
    warn "CPU arch: $ARCH (forced past check)"
    ;;
esac

# --- 2b. GPU + compute capability ----------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  [[ "$FORCE_HW" -eq 1 ]] || err "nvidia-smi not found — install NVIDIA drivers first (use --force to skip)"
  warn "nvidia-smi missing (forced past check)"
else
  GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true)"
  if [[ -z "$GPU_INFO" ]]; then
    [[ "$FORCE_HW" -eq 1 ]] || err "nvidia-smi returned no GPUs"
    warn "no GPUs reported (forced past check)"
  else
    GPU_NAME="$(printf '%s\n' "$GPU_INFO" | head -1 | awk -F', *' '{print $1}')"
    GPU_CAP="$(printf '%s\n'  "$GPU_INFO" | head -1 | awk -F', *' '{print $2}')"
    GPU_MEM="$(printf '%s\n'  "$GPU_INFO" | head -1 | awk -F', *' '{print $3}')"
    say "GPU: $GPU_NAME  compute_cap=$GPU_CAP  vram=$GPU_MEM"

    # SM detection: 12.1 = SM121 (GB10/Spark), 12.0 = data-center Blackwell.
    case "$GPU_CAP" in
      12.1)
        if printf '%s' "$GPU_NAME" | grep -qiE 'GB10|Spark'; then
          ok "DGX Spark / GB10 detected (compute_cap=12.1, name matches)"
        else
          ok "SM121 detected (compute_cap=12.1) — GPU name '$GPU_NAME' is unfamiliar but compute capability matches"
        fi
        ;;
      12.0)
        warn "compute_cap=12.0 — datacenter Blackwell (B100/B200), NOT SM121."
        hint "This checkpoint and the surrounding kernel selection were tuned"
        hint "for SM121 / consumer Blackwell (~99 KiB shmem/SM). It will likely"
        hint "work on B200 (228 KiB shmem) but the SM121-specific decode-kernel"
        hint "workaround (VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0) may not be"
        hint "needed. Continuing — re-run with --force to silence this warning."
        ;;
      9.*|10.*|11.*)
        warn "compute_cap=$GPU_CAP — Hopper / Ada / pre-Blackwell. FP8 path may"
        hint "work on H100 (9.0) but the install has only been validated on SM121."
        if [[ "$FORCE_HW" -eq 0 ]] && ! confirm "Continue anyway?" "n"; then
          err "aborted by user"
        fi
        ;;
      *)
        if [[ "$FORCE_HW" -eq 0 ]]; then
          err "compute_cap=$GPU_CAP unsupported. Need ≥9.0 (H100/Hopper) or ≥12.0 (Blackwell). Use --force to override."
        else
          warn "compute_cap=$GPU_CAP (forced past check)"
        fi
        ;;
    esac
  fi
fi

# --- 2c. system memory ---------------------------------------------------
# Note: actual DGX Spark reports MemTotal=119 GiB (not the marketed 128 GiB)
# because firmware and reserved regions consume the rest. Resident is ~110
# GiB during serving, so 118 GiB is a safe floor with headroom for the OS.
MEM_KB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
MEM_GIB=$(( MEM_KB / 1024 / 1024 ))
MEM_FLOOR=118
if [[ "$MEM_GIB" -lt "$MEM_FLOOR" ]]; then
  if [[ "$FORCE_HW" -eq 0 ]]; then
    err "system memory: ${MEM_GIB} GiB — need ≥${MEM_FLOOR} GiB (resident during serving ~110 GiB). Use --force to override."
  else
    warn "system memory: ${MEM_GIB} GiB (forced past check, expect OOM)"
  fi
else
  ok "system memory: ${MEM_GIB} GiB (≥${MEM_FLOOR} required)"
fi

# --- 2d. disk space (need ≥100 GiB free where MODELS_DIR will live) -------
mkdir -p "$MODELS_DIR" "$LOGS_DIR" "$EXTRAS_DIR"
FREE_KB="$(df -k --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -d ' ')"
FREE_GIB=$(( ${FREE_KB:-0} / 1024 / 1024 ))
if [[ "$FREE_GIB" -lt 100 ]]; then
  if [[ "$FORCE_HW" -eq 0 ]]; then
    err "free disk on $MODELS_DIR: ${FREE_GIB} GiB — need ≥100 GiB for the checkpoint. Use --force to override."
  else
    warn "free disk: ${FREE_GIB} GiB (forced past check)"
  fi
else
  ok "free disk on $MODELS_DIR: ${FREE_GIB} GiB"
fi

# --- 2e. docker + GPU runtime --------------------------------------------
command -v docker >/dev/null 2>&1 || err "docker not found — install Docker Engine and the NVIDIA Container Toolkit first"
docker info >/dev/null 2>&1 || err "docker daemon not reachable for $(id -un); add user to 'docker' group or run with sudo"
if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  ok "docker + nvidia runtime available"
else
  warn "nvidia container runtime not detected — '--gpus all' will fail"
  hint "install nvidia-container-toolkit and restart docker"
fi

# --- 2f. python (only needed for hf-cli install if missing) --------------
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found — required for the hf CLI"
fi

# ============================================================================
# 3. ensure hf CLI
# ============================================================================

if ! command -v hf >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/hf" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if ! command -v hf >/dev/null 2>&1; then
  say "installing huggingface_hub[cli,hf_xet] (user-local)..."
  if ! python3 -m pip install --user --quiet --upgrade 'huggingface_hub[cli,hf_xet]' 2>/dev/null; then
    # Fallback for distros that block --user installs without --break-system-packages
    python3 -m pip install --user --quiet --break-system-packages --upgrade 'huggingface_hub[cli,hf_xet]'
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v hf >/dev/null 2>&1 || err "hf CLI install succeeded but binary not on PATH; check $HOME/.local/bin"
fi
ok "hf CLI: $(hf --version 2>&1)"

# Make sure hf_xet is available — the upload/download speedup is huge.
if ! python3 -c 'import hf_xet' >/dev/null 2>&1; then
  say "installing hf_xet for faster downloads..."
  python3 -m pip install --user --quiet --upgrade hf_xet 2>/dev/null \
    || python3 -m pip install --user --quiet --break-system-packages --upgrade hf_xet
fi

# ============================================================================
# 4. authenticate with HuggingFace
# ============================================================================

WHOAMI=""
if [[ -n "$HF_TOKEN" ]]; then
  say "authenticating with provided HF token..."
  HF_TOKEN="$HF_TOKEN" hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
fi

if hf auth whoami >/dev/null 2>&1; then
  WHOAMI="$(hf auth whoami 2>&1 | awk '/^[[:space:]]*user:/ {print $2; exit}')"
  ok "HF authenticated as: ${WHOAMI:-unknown}"
else
  say "no active HF login. The model is public so a token is OPTIONAL but"
  say "strongly recommended (rate limits, gated dataset compatibility)."
  if [[ "$NON_INTERACTIVE" -eq 0 ]] && have_tty_input; then
    if confirm "Provide an HF token now?" "y"; then
      prompt_or_die "HF token (input hidden)" HF_TOKEN "" 1
      if [[ -n "$HF_TOKEN" ]]; then
        if hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1; then
          WHOAMI="$(hf auth whoami 2>&1 | awk '/^[[:space:]]*user:/ {print $2; exit}')"
          ok "HF authenticated as: ${WHOAMI:-unknown}"
        else
          warn "token rejected, continuing unauthenticated"
        fi
      fi
    else
      warn "continuing unauthenticated (public download still works)"
    fi
  else
    warn "continuing unauthenticated"
  fi
fi

# ============================================================================
# 5. download model
# ============================================================================

if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
  say "skipping download (--skip-download set)"
  [[ -d "$MODEL_DIR" ]] || err "no model dir at $MODEL_DIR but --skip-download given"
else
  mkdir -p "$MODEL_DIR"
  say "downloading $HF_REPO → $MODEL_DIR  (workers=$NUM_WORKERS, ~85 GiB)"
  say "this is resumable; re-running picks up where it left off"
  hf download "$HF_REPO" \
    --repo-type model \
    --local-dir "$MODEL_DIR" \
    --max-workers "$NUM_WORKERS"
  ok "checkpoint downloaded"
fi

# ============================================================================
# 6. integrity check
# ============================================================================

if [[ "$SKIP_VERIFY" -eq 1 ]]; then
  say "skipping SHA256SUMS verification (--skip-verify set)"
elif [[ -f "$MODEL_DIR/SHA256SUMS" ]]; then
  say "verifying SHA256SUMS..."
  if ( cd "$MODEL_DIR" && sha256sum -c --quiet SHA256SUMS ); then
    ok "all $(wc -l < "$MODEL_DIR/SHA256SUMS") files verified"
  else
    err "SHA256SUMS check FAILED — re-run install or remove $MODEL_DIR and retry"
  fi
else
  warn "no SHA256SUMS in $MODEL_DIR; skipping verification"
fi

# ============================================================================
# 7. pull base image
# ============================================================================

say "pulling docker image: $BASE_IMAGE"
docker pull "$BASE_IMAGE"
ok "base image ready"

# ============================================================================
# 8. clone / pull overlay repo
# ============================================================================

if [[ -d "$WORK_DIR/.git" ]]; then
  say "updating overlay repo at $WORK_DIR (branch=$OVERLAY_REF)"
  git -C "$WORK_DIR" fetch --quiet origin "$OVERLAY_REF" || true
  git -C "$WORK_DIR" checkout --quiet "$OVERLAY_REF" || true
  git -C "$WORK_DIR" pull --ff-only --quiet || warn "could not fast-forward overlay repo (local changes?)"
else
  say "cloning overlay repo: $OVERLAY_REPO → $WORK_DIR (branch=$OVERLAY_REF)"
  git clone --quiet --branch "$OVERLAY_REF" "$OVERLAY_REPO" "$WORK_DIR"
fi
ok "overlay at: $WORK_DIR"

# Sanity-check the overlay has the run script we need.
RUN_SH_REL="eugr_mod/mods/ds4-2bit-deepseek-v4-flash/run-on-lmxxf.sh"
[[ -f "$WORK_DIR/$RUN_SH_REL" ]] || err "expected $WORK_DIR/$RUN_SH_REL not found in overlay; check OVERLAY_REF"

# ============================================================================
# 9. start container
# ============================================================================

if [[ "$NO_START" -eq 1 ]]; then
  say "skipping container start (--no-start set)"
else
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    say "container '$CONTAINER_NAME' already exists; stopping + removing for clean restart"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  say "starting vLLM container '$CONTAINER_NAME'..."
  # Build env-var args. DG_LOCAL is intentionally only set if the dir exists,
  # so we don't break the in-container fallback to the image's built-in DeepGEMM.
  ENV_ARGS=(
    -e "LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so"
    -e "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0"
    -e "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1"
    -e "DS4_CKPT_DIR=/models/$MODEL_DIR_NAME"
  )
  if [[ -d "$EXTRAS_DIR/DeepGEMM" ]]; then
    ENV_ARGS+=( -e "DG_LOCAL=/extras/DeepGEMM" )
  fi

  # Write the in-container command to a file under WORK_DIR so it's auditable
  # and reusable by the systemd unit (next step). Using a here-doc inline in
  # `docker run -c "..."` would mean we couldn't grep it later.
  RUN_INNER="$WORK_DIR/scripts/_vllm-serve-inner.sh"
  mkdir -p "$WORK_DIR/scripts"

  # `--enforce-eager` is opt-in (debug-only). Default OFF so torch.compile +
  # CUDA graphs run, which is the difference between ~1.7 t/s and decode
  # throughput in the expected single-stream range on Spark.
  if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
    ENFORCE_EAGER_ARG=" --enforce-eager"
  else
    ENFORCE_EAGER_ARG=""
  fi

  cat >"$RUN_INNER" <<EOSH
#!/bin/bash
# Auto-generated by install.sh — invoked inside the lmxxf container.
# Do NOT edit by hand; re-run install.sh to regenerate.
set -e
bash /work/$RUN_SH_REL > /logs/serve-mod.log 2>&1
exec vllm serve /models/$MODEL_DIR_NAME \\
  --served-model-name $SERVED_MODEL_NAME \\
  --quantization deepseek_v4_hybrid_iq2 \\
  --port $PORT --host $HOST_BIND \\
  --max-model-len $MAX_MODEL_LEN \\
  --gpu-memory-utilization $GPU_UTIL \\
  --kv-cache-dtype $KV_DTYPE \\
  --attention-backend $ATTN_BACKEND${ENFORCE_EAGER_ARG} 2>&1 | tee /logs/serve.log
EOSH
  chmod +x "$RUN_INNER"

  docker run -d \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --network host \
    --restart unless-stopped \
    -v "$MODELS_DIR":/models \
    -v "$WORK_DIR":/work \
    -v "$LOGS_DIR":/logs \
    -v "$EXTRAS_DIR":/extras \
    "${ENV_ARGS[@]}" \
    --entrypoint bash \
    "$BASE_IMAGE" \
    -c "bash /work/scripts/_vllm-serve-inner.sh"
  ok "container started; logs at $LOGS_DIR/serve.log + serve-mod.log"
fi

# ============================================================================
# 10. wait for /health + smoke test
# ============================================================================

if [[ "$NO_START" -eq 0 && "$SKIP_SMOKE" -eq 0 ]]; then
  HEALTH_URL="http://localhost:$PORT/health"
  COMPLETIONS_URL="http://localhost:$PORT/v1/completions"

  say "waiting for vLLM to be ready at $HEALTH_URL ..."
  say "first start can take 5-10 min (pip install of overlay, model load)"
  TIMEOUT_S=900   # 15 min cap
  WAITED=0
  while ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; do
    if [[ "$WAITED" -ge "$TIMEOUT_S" ]]; then
      warn "vLLM did not become ready within $TIMEOUT_S s"
      hint "check logs:  tail -200 $LOGS_DIR/serve.log"
      hint "             tail -200 $LOGS_DIR/serve-mod.log"
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      warn "container '$CONTAINER_NAME' is no longer running"
      hint "docker logs $CONTAINER_NAME --tail 100"
      break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if (( WAITED % 60 == 0 )); then
      say "  ... still waiting (${WAITED}s elapsed)"
    fi
  done

  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    ok "vLLM is healthy"
    SMOKE_SH="$WORK_DIR/scripts/smoke-test.sh"
    if [[ -x "$SMOKE_SH" ]]; then
      say "running smoke test: $SMOKE_SH"
      # Don't let a smoke-test failure abort the installer — surface it but
      # leave the container running so the user can debug.
      set +e
      "$SMOKE_SH" --port "$PORT" --model "$SERVED_MODEL_NAME"
      SMOKE_RC=$?
      set -e
      if [[ "$SMOKE_RC" -ne 0 ]]; then
        warn "smoke test exited with status $SMOKE_RC"
        hint "container is still running; debug with:"
        hint "  $SMOKE_SH --port $PORT --model $SERVED_MODEL_NAME --strict --verbose"
      fi
    else
      warn "smoke-test.sh not found at $SMOKE_SH; skipping"
    fi
  fi
fi

# ============================================================================
# 11. systemd unit (optional)
# ============================================================================

maybe_install_systemd() {
  local unit=/etc/systemd/system/ds4-vllm.service
  say "installing systemd unit at $unit ..."
  sudo tee "$unit" >/dev/null <<EOF
[Unit]
Description=DeepSeek-V4-Flash 2-bit hybrid (vLLM) on DGX Spark
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=10
ExecStartPre=-/usr/bin/docker stop $CONTAINER_NAME
ExecStartPre=-/usr/bin/docker rm $CONTAINER_NAME
ExecStart=/usr/bin/docker run --rm \\
  --gpus all \\
  --name $CONTAINER_NAME \\
  --network host \\
  -v $MODELS_DIR:/models \\
  -v $WORK_DIR:/work \\
  -v $LOGS_DIR:/logs \\
  -v $EXTRAS_DIR:/extras \\
  -e LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so \\
  -e VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 \\
  -e VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1 \\
  -e DS4_CKPT_DIR=/models/$MODEL_DIR_NAME \\
  --entrypoint bash \\
  $BASE_IMAGE \\
  -c "bash /work/scripts/_vllm-serve-inner.sh"
ExecStop=/usr/bin/docker stop $CONTAINER_NAME

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable ds4-vllm.service
  ok "systemd unit installed and enabled"
  hint "the running 'docker run' (with --restart unless-stopped) is a separate"
  hint "instance from systemd's. On next boot systemd will own the lifecycle."
  hint "  status:  sudo systemctl status ds4-vllm"
  hint "  logs:    sudo journalctl -u ds4-vllm -f"
}

if [[ "$NO_START" -eq 0 ]]; then
  if [[ -z "${INSTALL_SYSTEMD}" ]]; then
    say ""
    say "the container was started with '--restart unless-stopped', so it will"
    say "already come back automatically after host reboot. A systemd unit is"
    say "OPTIONAL — only useful if you want 'systemctl status / stop / start'"
    say "as the management surface (e.g. for monitoring integration)."
    if confirm "Install systemd unit anyway?" "n"; then
      INSTALL_SYSTEMD=1
    else
      INSTALL_SYSTEMD=0
    fi
  fi
  if [[ "$INSTALL_SYSTEMD" -eq 1 ]]; then
    maybe_install_systemd
  fi
fi

# ============================================================================
# 12. summary
# ============================================================================

echo
ok "install complete"
echo
say "endpoint:    http://localhost:$PORT/v1"
say "model name:  $SERVED_MODEL_NAME"
say "logs:        tail -f $LOGS_DIR/serve.log"
say "container:   docker logs -f $CONTAINER_NAME"
echo
say "test it:"
cat <<EOF
  curl -s http://localhost:$PORT/v1/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"$SERVED_MODEL_NAME","prompt":"The capital of France is","max_tokens":50,"temperature":0}' \\
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"])'
EOF
echo
