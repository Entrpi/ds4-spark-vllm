#!/bin/bash
# Slim mod runner for the lmxxf/vllm-deepseek-v4-dgx-spark image.
#
# That image already ships:
#   - vLLM mainline (post PR #40860) with full SM12x dispatch in
#     vllm/utils/deep_gemm.py for HC, fp8_mqa_logits, fp8_paged_mqa_logits
#   - vllm/v1/attention/ops/deepseek_v4_ops/ with Triton fp8_einsum etc.
#   - a working DeepGEMM build for SM121
#
# So we only need to:
#   1. install ds4_hybrid_quant (registers the deepseek_v4_hybrid_iq2 method)
#   2. patch the deepseek_v4.py load_weights bug for our hybrid expert names
# Skip the DeepGEMM source patches and JIT cache wipes from run.sh — those
# fight lmxxf's working setup.

set -euo pipefail

SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
DS4_REPO="${DS4_REPO:-https://github.com/Entrpi/ds4-spark-vllm.git}"
DS4_REF="${DS4_REF:-main}"
DS4_LOCAL="${DS4_LOCAL:-/workspace/ds4-spark-vllm}"

echo "=== ds4-2bit-deepseek-v4-flash :: lmxxf base ==="

if [ ! -f "$SITE_PACKAGES/vllm/model_executor/models/deepseek_v4.py" ]; then
    echo "[ds4 ERROR] lmxxf image missing deepseek_v4.py"
    exit 1
fi

if [ ! -d "$DS4_LOCAL" ]; then
    git clone --depth=1 --branch "$DS4_REF" "$DS4_REPO" "$DS4_LOCAL"
else
    git -C "$DS4_LOCAL" fetch origin "$DS4_REF" 2>&1 | tail -3
    (git -C "$DS4_LOCAL" checkout "$DS4_REF" 2>&1 | tail -3 || true)
    git -C "$DS4_LOCAL" pull --ff-only 2>&1 | tail -3
fi

echo "[ds4] pip install -e $DS4_LOCAL"
pip install -e "$DS4_LOCAL"

# Patch deepseek_v4.py load_weights to gracefully handle expert tensor names
# that don't match the standard w1/w2/w3 mapping (we use w13_iq2xxs_qs etc).
# This is OUR patch — not in lmxxf's image.
DSV4_PY="$SITE_PACKAGES/vllm/model_executor/models/deepseek_v4.py"
if ! grep -q "DS4_HYBRID_PATCH" "$DSV4_PY"; then
    echo "[ds4] Patching $DSV4_PY load_weights for unrecognized expert tensors"
    python3 - <<PY
import re
p = "$DSV4_PY"
s = open(p).read()
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
    '                    if name_mapped is None:  # DS4_HYBRID_PATCH\n'
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
    find "$SITE_PACKAGES/vllm/model_executor/models/__pycache__" -name "deepseek_v4*" -delete 2>/dev/null || true
fi

# Sanity-check our quant config registered.
python3 - <<'PY'
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, get_quantization_config,
)
from ds4_hybrid_quant.vllm_patches import register_plugin
register_plugin()
cls = get_quantization_config("deepseek_v4_hybrid_iq2")
print(f"[ds4] registered: {cls.__name__} (method=deepseek_v4_hybrid_iq2)")
PY

echo "[ds4] Done. Available as --quantization deepseek_v4_hybrid_iq2"
