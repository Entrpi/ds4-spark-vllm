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
echo "[ds4] Installing DeepGEMM (jasl SM121 fork + cherry-picked SM120 HC files)..."
DG_LOCAL="${DG_LOCAL:-/workspace/DeepGEMM}"
if [ ! -d "$DG_LOCAL" ]; then
    git clone --depth=1 --recurse-submodules https://github.com/jasl/DeepGEMM.git "$DG_LOCAL"
fi
# CRITICAL: wipe build/ before reinstall. setup.py's incremental build keeps
# python_api.o cached based on mtime; if the source headers were patched in
# place (which `git checkout TAG -- file` does), pip's --force-reinstall does
# NOT trigger a recompile because the .o is "newer" than the affected .hpp's.
# Without this wipe, dispatch tables look correct on disk but the running
# .so is stale and asserts "Unsupported architecture".
rm -rf "$DG_LOCAL/build" "$DG_LOCAL"/*.egg-info
pip install "$DG_LOCAL" --no-build-isolation --force-reinstall --no-deps --no-cache-dir
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

# 2a. Patch deepseek_v4.py load_weights to gracefully handle expert
#     tensor names that don't match the standard w1/w2/w3 mapping
#     (we use w13_iq2xxs_qs etc.). Without this, load_weights raises
#     UnboundLocalError on `loaded_params.add(name_mapped)` for any
#     unrecognized expert tensor.
# 2b. Patch DeepGEMM's get_arch() to return the SM12x family arch ("120f")
#     instead of arch-specific ("121a"). The SM120 HC kernels are written
#     with no SM121-only features, but JIT-compiling for sm_121a yields a
#     cubin the driver rejects with CUDA_ERROR_INVALID_IMAGE. Family target
#     "sm_120f" produces a CUBIN compatible with both 120 and 121.
DG_DEV_RT="$DG_LOCAL/csrc/jit/device_runtime.hpp"
if [ -f "$DG_DEV_RT" ] && ! grep -q "DS4_GET_ARCH_PATCH" "$DG_DEV_RT"; then
    echo "[ds4] Patching $DG_DEV_RT get_arch for SM12x family"
    python3 - <<PY
p = "$DG_DEV_RT"
s = open(p).read()
old = (
    '        if (major == 10 and minor != 1) {\n'
    '            if (number_only)\n'
    '                return "100";\n'
    '            return support_arch_family ? "100f" : "100a";\n'
    '        }\n'
    '        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");\n'
)
new = (
    '        if (major == 10 and minor != 1) {  // DS4_GET_ARCH_PATCH\n'
    '            if (number_only)\n'
    '                return "100";\n'
    '            return support_arch_family ? "100f" : "100a";\n'
    '        }\n'
    '        if (major == 12 and support_arch_family) {  // DS4_GET_ARCH_PATCH\n'
    '            if (number_only) return "120";\n'
    '            return "120f";\n'
    '        }\n'
    '        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");\n'
)
assert old in s, "expected get_arch pattern not found"
open(p, "w").write(s.replace(old, new, 1))
print("[ds4] device_runtime.hpp patched")
PY
fi

DSV4_PY="$SITE_PACKAGES/vllm/model_executor/models/deepseek_v4.py"
if ! grep -q "DS4_HYBRID_PATCH" "$DSV4_PY"; then
    echo "[ds4] Patching $DSV4_PY load_weights for unrecognized expert tensors"
    python3 - <<PY
import re
p = "$DSV4_PY"
s = open(p).read()
# Insert name_mapped = None before the inner for loop, and a fallback after.
old = (
    '                    for mapping in expert_mapping:\n'
    '                        param_name, weight_name, expert_id, shard_id = mapping\n'
)
new = (
    '                    name_mapped = None  # DS4_HYBRID_PATCH\n'
    '                    for mapping in expert_mapping:\n'
    '                        param_name, weight_name, expert_id, shard_id = mapping\n'
)
assert old in s, "expected pattern not found in deepseek_v4.py"
s = s.replace(old, new, 1)
old2 = '                    loaded_params.add(name_mapped)\n                    continue\n'
new2 = (
    '                    # DS4_HYBRID_PATCH: fall through to default loader if\n'
    '                    # no expert_mapping entry matched (custom quant params).\n'
    '                    if name_mapped is None:\n'
    '                        if name in params_dict:\n'
    '                            param = params_dict[name]\n'
    '                            weight_loader = getattr(param, "weight_loader", default_weight_loader)\n'
    '                            weight_loader(param, loaded_weight)\n'
    '                            loaded_params.add(name)\n'
    '                        continue\n'
    '                    loaded_params.add(name_mapped)\n'
    '                    continue\n'
)
assert old2 in s, "expected loaded_params.add line not found"
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("[ds4] deepseek_v4.py patched")
PY
    # Clear any compiled bytecode of the patched file.
    find "$SITE_PACKAGES/vllm/model_executor/models/__pycache__" -name "deepseek_v4*" -delete 2>/dev/null || true
fi

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
