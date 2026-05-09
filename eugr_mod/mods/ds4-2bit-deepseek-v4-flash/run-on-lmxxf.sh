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
    '                    _ds4_load_succeeded = False  # DS4_HYBRID_PATCH\n'
    '                    for mapping in expert_mapping:\n'
    '                        param_name, weight_name, expert_id, shard_id = mapping\n'
)
assert old in s, "expected pattern not found in deepseek_v4.py"
s = s.replace(old, new, 1)

# Track whether the inner expert_mapping loop's loader actually succeeded.
# Without this, name_mapped gets set by ANY mapping whose weight_name is a
# substring of our 2-bit tensor names (e.g. "scales" matches w2_q2k_scales),
# the FusedMoE per-expert loader returns success=False, but our fall-through
# was gated on name_mapped is None — so it didn't fire and the param went
# unloaded.
old_succ = (
    '                        if success:\n'
    '                            name = name_mapped\n'
    '                            break\n'
)
new_succ = (
    '                        if success:\n'
    '                            _ds4_load_succeeded = True  # DS4_HYBRID_PATCH\n'
    '                            name = name_mapped\n'
    '                            break\n'
)
assert old_succ in s, "expected 'if success' block not found"
s = s.replace(old_succ, new_succ, 1)

old2 = '                    loaded_params.add(name_mapped)\n                    continue\n'
new2 = (
    '                    if not _ds4_load_succeeded:  # DS4_HYBRID_PATCH\n'
    '                        # safetensor names may have had a "model." prefix stripped\n'
    '                        # by an upstream WeightsMapper; AND our converter used\n'
    '                        # mlp.experts.* while DSv4-Flash uses ffn.experts.* in\n'
    '                        # the live module tree — so try multiple rewrites.\n'
    '                        _candidates = [\n'
    '                            name,\n'
    '                            f"model.{name}",\n'
    '                            name.replace("mlp.experts", "ffn.experts"),\n'
    '                            f"model.{name}".replace("mlp.experts", "ffn.experts"),\n'
    '                        ]\n'
    '                        target = next((c for c in _candidates if c in params_dict), None)\n'
    '                        if target is not None:\n'
    '                            param = params_dict[target]\n'
    '                            weight_loader = getattr(param, "weight_loader", default_weight_loader)\n'
    '                            weight_loader(param, loaded_weight)\n'
    '                            loaded_params.add(target)\n'
    '                        else:\n'
    '                            if "iq2xxs" in name or "q2k" in name:\n'
    '                                print(f"[DS4_FT_DBG] no match for {name!r} (tried 4 candidates)", flush=True)\n'
    '                        continue\n'
    '                    loaded_params.add(name_mapped)\n'
    '                    continue\n'
)
assert old2 in s, "expected loaded_params.add line not found"
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("[ds4] deepseek_v4.py patched")
PY
    # Insert a post-load completion call right before `return loaded_params`.
    # Delegates to ds4_hybrid_quant.load_completion which (a) default-inits
    # scale/bias/norm params and (b) direct-loads from safetensor anything
    # vLLM's standard pipeline missed (mostly fused-attn / compressor /
    # shared-experts tensor families).
    python3 - <<PY2
p = "$DSV4_PY"
s = open(p).read()
needle = "        return loaded_params\n"
inject = (
    "        # DS4_HYBRID_PATCH post-load completion\n"
    "        try:\n"
    "            import os as _os\n"
    "            from ds4_hybrid_quant.load_completion import complete_load\n"
    "            complete_load(\n"
    "                params_dict, loaded_params,\n"
    "                ckpt_dir=_os.environ.get('DS4_CKPT_DIR', '/models/deepseek-v4-flash-ds4-q2'),\n"
    "            )\n"
    "        except Exception as _e:\n"
    "            print(f'[DS4_LOAD_COMP] FAILED: {_e!r}', flush=True)\n"
    "            import traceback as _tb; _tb.print_exc()\n"
    "        return loaded_params\n"
)
assert needle in s, "expected 'return loaded_params' line not found"
s = s.replace(needle, inject, 1)
open(p, 'w').write(s)
print('[ds4] deepseek_v4.py post-load completion patched')
PY2
    # Outer-class load_weights diagnostic: dump lm_head + embed_tokens stats
    # post-AutoWeightsLoader, comparing against on-disk safetensor data.
    # This isolates whether the outer-class load actually populated those.
    python3 - <<PY3
p = "$DSV4_PY"
s = open(p).read()
needle = (
    "        loader = AutoWeightsLoader(self, skip_substrs=[\"mtp.\"])\n"
    "        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
)
assert needle in s, "outer-class loader pattern not found"
inject = (
    "        loader = AutoWeightsLoader(self, skip_substrs=[\"mtp.\"])\n"
    "        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
    "        # DS4_HEAD_DBG: verify lm_head + embed_tokens + hc_head_* + sample\n"
    "        # per-layer HC tensors loaded post-AutoWeightsLoader.\n"
    "        try:\n"
    "            import torch as _t\n"
    "            _wanted_substrs = ('lm_head', 'embed_tokens', 'hc_head', 'hc_attn', 'hc_ffn')\n"
    "            for _attr in ('lm_head', 'model'):\n"
    "                _m = getattr(self, _attr, None)\n"
    "                if _m is None:\n"
    "                    continue\n"
    "                for _name, _p in _m.named_parameters():\n"
    "                    full = f'{_attr}.{_name}'\n"
    "                    # filter to wanted, AND keep at most 2 per-layer HC samples (layer 0 and layer 30)\n"
    "                    if not any(s in full for s in _wanted_substrs):\n"
    "                        continue\n"
    "                    if 'layers.' in full and 'layers.0.' not in full and 'layers.30.' not in full:\n"
    "                        continue\n"
    "                    if _t.is_floating_point(_p):\n"
    "                        _max = float(_p.abs().max().item())\n"
    "                        _mean = float(_p.abs().mean().item())\n"
    "                        _nf = int((~_t.isfinite(_p)).sum().item())\n"
    "                        print(f'[DS4_HEAD_DBG] {full}: shape={tuple(_p.shape)} dtype={_p.dtype} |max|={_max:.3e} |mean|={_mean:.3e} non_finite={_nf}', flush=True)\n"
    "        except Exception as _e:\n"
    "            print(f'[DS4_HEAD_DBG] error: {_e!r}', flush=True)\n"
)
s = s.replace(needle, inject, 1)
open(p, 'w').write(s)
print('[ds4] deepseek_v4.py outer head-dbg patched')
PY3
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
