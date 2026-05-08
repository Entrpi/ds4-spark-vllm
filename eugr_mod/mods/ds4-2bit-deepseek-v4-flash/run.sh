#!/bin/bash
# Mod: ds4-2bit-deepseek-v4-flash
#
# Installs the ds4 hybrid IQ2_XXS+Q2_K+FP8 quant method into vLLM:
#   - ds4_hybrid_quant package (Triton kernels + builder + vllm_patches)
#   - registration patch that adds "deepseek_v4_hybrid_iq2" to vLLM's
#     QUANTIZATION_METHODS registry
#
# Run-time prerequisites that the mod expects to be present (handled by
# the b12x mod or the base eugr container):
#   - Triton + libdevice supporting SM121
#   - vLLM mainline >= the commit that landed PR #40860 (DeepSeek V4)
#   - flashinfer with SM121 support (via b12x or Triton fallback)

set -euo pipefail

SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
DS4_REPO="${DS4_REPO:-https://github.com/Entrpi/ds4-spark-vllm.git}"
DS4_REF="${DS4_REF:-main}"
DS4_LOCAL="${DS4_LOCAL:-/workspace/vllm-hybrid-quant-ds4}"

echo "=== ds4-2bit-deepseek-v4-flash mod ==="

# 0. Sanity-check vLLM has the DeepSeek V4 model file (PR #40860 merged).
if [ ! -f "$SITE_PACKAGES/vllm/model_executor/models/deepseek_v4.py" ]; then
    echo "[ds4 ERROR] vLLM is missing deepseek_v4.py; need a mainline build "
    echo "            from after PR #40860 (April 27, 2026)."
    exit 1
fi

# 1. Pull the ds4_hybrid_quant package.
if [ ! -d "$DS4_LOCAL" ]; then
    echo "[ds4] Cloning $DS4_REPO @ $DS4_REF -> $DS4_LOCAL"
    git clone --depth=1 --branch "$DS4_REF" "$DS4_REPO" "$DS4_LOCAL"
else
    echo "[ds4] Updating $DS4_LOCAL"
    git -C "$DS4_LOCAL" fetch origin "$DS4_REF"
    git -C "$DS4_LOCAL" checkout "$DS4_REF"
    git -C "$DS4_LOCAL" pull --ff-only
fi

# 2. Install as a regular package so its imports work.
echo "[ds4] Installing ds4_hybrid_quant"
pip install -e "$DS4_LOCAL"

# 3. Drop the vLLM-side dispatch files into vLLM's quantization tree.
QUANT_DIR="$SITE_PACKAGES/vllm/model_executor/layers/quantization"
mkdir -p "$QUANT_DIR/ds4_hybrid_iq2"
cp "$DS4_LOCAL/src/ds4_hybrid_quant/vllm_patches/__init__.py"  "$QUANT_DIR/ds4_hybrid_iq2/__init__.py"
cp "$DS4_LOCAL/src/ds4_hybrid_quant/vllm_patches/config.py"    "$QUANT_DIR/ds4_hybrid_iq2/config.py"
cp "$DS4_LOCAL/src/ds4_hybrid_quant/vllm_patches/moe_method.py" "$QUANT_DIR/ds4_hybrid_iq2/moe_method.py"

# 4. Register the new method in vLLM's QUANTIZATION_METHODS registry.
REGISTER_PY="$QUANT_DIR/__init__.py"
if ! grep -q "deepseek_v4_hybrid_iq2" "$REGISTER_PY"; then
    echo "[ds4] Patching $REGISTER_PY to register deepseek_v4_hybrid_iq2"
    python3 - <<'PY'
import re, sys
p = "$QUANT_DIR/__init__.py".replace('$QUANT_DIR', '__QUANT_DIR__')
PY
    # Use sed in two steps: import + dict entry.
    python3 - "$REGISTER_PY" <<'PYEOF'
import sys, re
p = sys.argv[1]
src = open(p).read()
if "Ds4HybridIq2Config" not in src:
    # Add the import near other quant-config imports.
    src = src.replace(
        "from vllm.model_executor.layers.quantization.fp8 import Fp8Config",
        "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n"
        "from vllm.model_executor.layers.quantization.ds4_hybrid_iq2.config "
        "import Ds4HybridIq2Config",
    )
    # Add the dict entry. We append a line just before the closing brace
    # of QUANTIZATION_METHODS.
    src = re.sub(
        r"(QUANTIZATION_METHODS\s*[:=][^{]*\{[^}]*?)(\n\})",
        r'\1\n    "deepseek_v4_hybrid_iq2": Ds4HybridIq2Config,\2',
        src, count=1,
    )
    open(p, "w").write(src)
PYEOF
else
    echo "[ds4] $REGISTER_PY already references deepseek_v4_hybrid_iq2"
fi

# 5. Clear pycache so changes take effect.
find "$QUANT_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES/ds4_hybrid_quant" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "[ds4] Done. Available as --quantization deepseek_v4_hybrid_iq2"
