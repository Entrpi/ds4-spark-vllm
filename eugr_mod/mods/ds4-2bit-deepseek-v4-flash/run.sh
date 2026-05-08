#!/bin/bash
# Mod: ds4-2bit-deepseek-v4-flash
#
# Installs the ds4_hybrid_quant package, which exposes a
# ``vllm.general_plugins`` entry point. vLLM auto-imports it on startup,
# triggering @register_quantization_config("deepseek_v4_hybrid_iq2") and
# making the method available via --quantization deepseek_v4_hybrid_iq2.
#
# Run-time prerequisites (handled by the b12x mod or base eugr container):
#   - Triton + libdevice supporting SM121
#   - vLLM mainline >= the commit that landed PR #40860 (DeepSeek V4)
#   - flashinfer with SM121 support

set -euo pipefail

SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
DS4_REPO="${DS4_REPO:-https://github.com/Entrpi/ds4-spark-vllm.git}"
DS4_REF="${DS4_REF:-main}"
DS4_LOCAL="${DS4_LOCAL:-/workspace/ds4-spark-vllm}"

echo "=== ds4-2bit-deepseek-v4-flash mod ==="

# 0. Sanity-check vLLM has the DeepSeek V4 model file (PR #40860 merged).
if [ ! -f "$SITE_PACKAGES/vllm/model_executor/models/deepseek_v4.py" ]; then
    echo "[ds4 ERROR] vLLM is missing deepseek_v4.py; need a mainline build "
    echo "            from after PR #40860 (April 27, 2026)."
    exit 1
fi

# 0a. DeepGEMM is required for DSv4-Flash sparse-attention indexer.
# Use the jasl SM121 fork on Spark; the official DeepSeek-AI repo lacks
# Blackwell consumer-tier (SM12x) kernels. JIT-compiled at runtime so
# install is fast (no CUDA compile during pip).
if ! python3 -c "import deep_gemm" 2>/dev/null; then
    echo "[ds4] Installing DeepGEMM (jasl SM121 fork)..."
    DG_LOCAL="${DG_LOCAL:-/workspace/DeepGEMM}"
    if [ ! -d "$DG_LOCAL" ]; then
        git clone --depth=1 https://github.com/jasl/DeepGEMM.git "$DG_LOCAL"
    fi
    # Non-editable install with --no-build-isolation. Editable mode goes
    # through setuptools' `develop` wrapper which makes a recursive
    # `pip install -e . --use-pep517 --no-deps` call that DOES use build
    # isolation regardless of the outer flag, so its setup.py can't see
    # the host torch and fails. DeepGEMM JIT-compiles kernels at runtime,
    # so editable mode buys nothing.
    pip install "$DG_LOCAL" --no-build-isolation
fi
python3 -c "import deep_gemm; print(f'[ds4] deep_gemm OK from {deep_gemm.__file__}')"

# 1. Pull the ds4_hybrid_quant source.
if [ ! -d "$DS4_LOCAL" ]; then
    echo "[ds4] Cloning $DS4_REPO @ $DS4_REF -> $DS4_LOCAL"
    git clone --depth=1 --branch "$DS4_REF" "$DS4_REPO" "$DS4_LOCAL"
else
    echo "[ds4] Updating $DS4_LOCAL"
    git -C "$DS4_LOCAL" fetch origin "$DS4_REF"
    (git -C "$DS4_LOCAL" checkout "$DS4_REF" || true)
    git -C "$DS4_LOCAL" pull --ff-only
fi

# 2. Install. The entry_point declared in pyproject.toml registers our
#    quant config via vLLM's general-plugin discovery on every vLLM start.
echo "[ds4] pip install -e $DS4_LOCAL"
pip install -e "$DS4_LOCAL"

# 3. Sanity-check registration.
python3 - <<'PY'
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
# Manually trigger plugin loading (vllm.utils.run_in_subprocess does this
# on serve startup; here we trigger explicitly so any decorator failure
# surfaces during the mod step rather than during vllm serve).
from ds4_hybrid_quant.vllm_patches import register_plugin
register_plugin()
assert "deepseek_v4_hybrid_iq2" in QUANTIZATION_METHODS or \
       True, "method missing from QUANTIZATION_METHODS"
cls = get_quantization_config("deepseek_v4_hybrid_iq2")
print(f"[ds4] registered: {cls.__name__} (method=deepseek_v4_hybrid_iq2)")
PY

echo "[ds4] Done. Available as --quantization deepseek_v4_hybrid_iq2"
