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
    # Insert a one-shot post-load summary right before the final `return loaded_params`
    # in the patched load_weights. Surfaces params that didn't get loaded — most
    # likely cause of all-NaN activations on first real inference.
    python3 - <<PY2
p = "$DSV4_PY"
s = open(p).read()
needle = "        return loaded_params\n"
inject = (
    "        # DS4_HYBRID_PATCH post-load summary + default-init for unloaded params.\n"
    "        # Our converter only emits routed-expert weights + sgl-FP8 base; the\n"
    "        # vLLM model registers some additional params (per-layer FP8 scales for\n"
    "        # BF16 paths, indexer k-norm) that have no corresponding safetensor.\n"
    "        # Without init these are torch.empty() garbage → all-NaN forward.\n"
    "        try:\n"
    "            import torch as _t\n"
    "            unloaded = [k for k in params_dict if k not in loaded_params]\n"
    "            # Bucket by suffix pattern\n"
    "            _bucket = {}\n"
    "            for k in unloaded:\n"
    "                if k.endswith('.weight_scale_inv') or k.endswith('.weight_scale') or k.endswith('.scale'):\n"
    "                    _bucket.setdefault('scale->1.0', []).append(k)\n"
    "                elif k.endswith('.bias'):\n"
    "                    _bucket.setdefault('bias->0.0', []).append(k)\n"
    "                elif k.endswith('_norm.weight') or k.endswith('norm.weight'):\n"
    "                    _bucket.setdefault('norm.weight->1.0', []).append(k)\n"
    "                else:\n"
    "                    _bucket.setdefault('OTHER (no default)', []).append(k)\n"
    "            print(f'[DS4_LOAD_SUMMARY] params_dict={len(params_dict)} loaded={len(loaded_params)} unloaded={len(unloaded)}', flush=True)\n"
    "            for cat, ks in _bucket.items():\n"
    "                print(f'[DS4_LOAD_SUMMARY] bucket {cat!r}: {len(ks)} params; sample: {ks[:3]}', flush=True)\n"
    "            # Initialize\n"
    "            with _t.no_grad():\n"
    "                for k in unloaded:\n"
    "                    pp = params_dict[k]\n"
    "                    if k.endswith('.weight_scale_inv') or k.endswith('.weight_scale') or k.endswith('.scale'):\n"
    "                        pp.data.fill_(1.0)\n"
    "                        loaded_params.add(k)\n"
    "                    elif k.endswith('.bias'):\n"
    "                        pp.data.zero_()\n"
    "                        loaded_params.add(k)\n"
    "                    elif k.endswith('_norm.weight') or k.endswith('norm.weight'):\n"
    "                        pp.data.fill_(1.0)\n"
    "                        loaded_params.add(k)\n"
    "            still_unloaded = [k for k in params_dict if k not in loaded_params]\n"
    "            print(f'[DS4_LOAD_SUMMARY] after default-init still_unloaded={len(still_unloaded)}', flush=True)\n"
    "            # Bucket still-unloaded by suffix-stem to see patterns at a glance\n"
    "            _stems = {}\n"
    "            for k in still_unloaded:\n"
    "                # last 2 dot-segments as stem (e.g. 'hc_attn_fn', 'hc_ffn_scale')\n"
    "                parts = k.split('.')\n"
    "                stem = '.'.join(parts[-2:]) if len(parts) >= 2 else k\n"
    "                _stems.setdefault(stem, []).append(k)\n"
    "            for stem in sorted(_stems, key=lambda x: -len(_stems[x])):\n"
    "                pp = params_dict[_stems[stem][0]]\n"
    "                print(f'[DS4_LOAD_SUMMARY] STEM {stem!r}: {len(_stems[stem])} params, e.g. {_stems[stem][0]} shape={tuple(pp.shape)} dtype={pp.dtype}', flush=True)\n"
    "        except Exception as _e:\n"
    "            print(f'[DS4_LOAD_SUMMARY] dump error: {_e!r}', flush=True)\n"
    "        return loaded_params\n"
)
assert needle in s, "expected 'return loaded_params' line not found"
s = s.replace(needle, inject, 1)
open(p, 'w').write(s)
print('[ds4] deepseek_v4.py post-load summary patched')
PY2
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
