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

# Replace tilelang's incomplete libcudart_stub.so with the real CUDA runtime.
# Tilelang ships a stub library with only ~55 symbols (no cudaDeviceReset).
# flashinfer.comm.cuda_ipc.CudaRTLibrary() scans /proc/self/maps for any
# library whose name contains "libcudart" and binds to the first match,
# which becomes tilelang's stub once tilelang is imported. Without
# cudaDeviceReset, flashinfer crashes at module load. Replacing the stub
# file with a copy of the real libcudart.so.13 makes tilelang's dlopen
# (by absolute path) still succeed and gives flashinfer the symbols it
# needs.  Idempotent: only swaps if cudaDeviceReset is missing.
TILELANG_STUB="$SITE_PACKAGES/tilelang/lib/libcudart_stub.so"
REAL_CUDART="/usr/local/cuda/lib64/libcudart.so.13"
if [ -f "$TILELANG_STUB" ] && [ -f "$REAL_CUDART" ]; then
    if ! nm -D "$TILELANG_STUB" 2>/dev/null | grep -q cudaDeviceReset; then
        echo "[ds4] Replacing tilelang's libcudart_stub.so with real libcudart (cudaDeviceReset fix)"
        cp -f "$TILELANG_STUB" "$TILELANG_STUB.bak.$(date +%s)" 2>/dev/null || true
        cp -f "$REAL_CUDART" "$TILELANG_STUB"
        echo "[ds4] libcudart stub patched; cudaDeviceReset now present"
    fi
fi

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

# Optional: overlay our vllm fork's python tree over the installed vllm.
# When DS4_VLLM_OVERLAY=1 (default if /workspace/vllm-fork exists), replaces
# site-packages/vllm/**/*.py with our fork's source. This lets us run
# Entrpi/vllm@kv-layout-on-jasl-c2d4811 (jasl's current PR #41834 head + our
# V1/V2/V3 KV cache fixes) against the lmxxf image's compiled extensions
# (vllm/_C.so etc.) without a full image rebuild. Pure-python + Triton
# changes only — no C++ ABI surface modified by either party.
VLLM_FORK_DIR="${VLLM_FORK_DIR:-/workspace/vllm-fork}"
if [ -d "$VLLM_FORK_DIR/vllm" ] && [ "${DS4_VLLM_OVERLAY:-1}" = "1" ]; then
    echo "[ds4] Overlaying $VLLM_FORK_DIR/vllm/ -> $SITE_PACKAGES/vllm/ (python files only)"
    # Copy python files only; preserve *.so / *.json / non-py artifacts.
    # NOTE: lmxxf base image doesn't ship rsync, so we do the walk in python.
    python3 - <<PY_OVERLAY
import os, shutil, sys
src = "$VLLM_FORK_DIR/vllm"
dst = "$SITE_PACKAGES/vllm"
copied = 0
skipped_dirs = {"__pycache__"}
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if d not in skipped_dirs]
    rel = os.path.relpath(root, src)
    dst_dir = dst if rel == "." else os.path.join(dst, rel)
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        if not f.endswith(".py"):
            continue
        shutil.copy2(os.path.join(root, f), os.path.join(dst_dir, f))
        copied += 1
print(f"[ds4] overlay: copied {copied} python files")
PY_OVERLAY
    # Wipe pycache so the new source loads cleanly on import.
    find "$SITE_PACKAGES/vllm" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    OVERLAY_SHA=$(git -C "$VLLM_FORK_DIR" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)
    echo "[ds4] vllm fork overlay applied at $OVERLAY_SHA"
    DS4_VLLM_OVERLAY=1
else
    echo "[ds4] vllm fork overlay skipped (DS4_VLLM_OVERLAY=${DS4_VLLM_OVERLAY:-1}, dir=$VLLM_FORK_DIR exists=$([ -d $VLLM_FORK_DIR/vllm ] && echo yes || echo no))"
    DS4_VLLM_OVERLAY=0
fi

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
    "                ckpt_dir=_os.environ.get('DS4_CKPT_DIR', '/models/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target'),\n"
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
    "        # per-layer HC + per-layer gate.weight + correction_bias for several\n"
    "        # layers. Compare live magnitudes to disk safetensor values.\n"
    "        try:\n"
    "            import torch as _t\n"
    "            import json as _json\n"
    "            from pathlib import Path as _Path\n"
    "            from safetensors import safe_open as _safe_open\n"
    "            _wanted_substrs = ('lm_head', 'embed_tokens', 'hc_head', 'hc_attn', 'hc_ffn',\n"
    "                               'ffn.gate.weight', 'ffn.gate.e_score_correction_bias')\n"
    "            for _attr in ('lm_head', 'model'):\n"
    "                _m = getattr(self, _attr, None)\n"
    "                if _m is None:\n"
    "                    continue\n"
    "                for _name, _p in _m.named_parameters():\n"
    "                    full = f'{_attr}.{_name}'\n"
    "                    if not any(s in full for s in _wanted_substrs):\n"
    "                        continue\n"
    "                    # Sample a few specific layer indices for per-layer params\n"
    "                    if 'layers.' in full:\n"
    "                        ok = False\n"
    "                        for _li in (0, 1, 5, 11, 20, 35, 36, 40):\n"
    "                            if f'layers.{_li}.' in full:\n"
    "                                ok = True\n"
    "                                break\n"
    "                        if not ok:\n"
    "                            continue\n"
    "                    if _t.is_floating_point(_p):\n"
    "                        _max = float(_p.abs().max().item())\n"
    "                        _mean = float(_p.abs().mean().item())\n"
    "                        _nf = int((~_t.isfinite(_p)).sum().item())\n"
    "                        print(f'[DS4_HEAD_DBG] {full}: shape={tuple(_p.shape)} dtype={_p.dtype} |max|={_max:.3e} |mean|={_mean:.3e} non_finite={_nf}', flush=True)\n"
    "            # Disk comparison for ffn.gate.weight at sample layers — load directly\n"
    "            import os as _os\n"
    "            _ckpt = _os.environ.get('DS4_CKPT_DIR', '/models/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target')\n"
    "            _idx = _json.loads((_Path(_ckpt) / 'model.safetensors.index.json').read_text())['weight_map']\n"
    "            for _li in (0, 1, 5, 11, 20, 35, 36, 40):\n"
    "                _disk_name = f'layers.{_li}.ffn.gate.weight'\n"
    "                if _disk_name not in _idx:\n"
    "                    continue\n"
    "                with _safe_open(str(_Path(_ckpt) / _idx[_disk_name]), framework='pt') as _f:\n"
    "                    _dt = _f.get_tensor(_disk_name)\n"
    "                _dmax = float(_dt.abs().max().item()); _dmean = float(_dt.abs().mean().item())\n"
    "                print(f'[DS4_HEAD_DBG] DISK {_disk_name}: |max|={_dmax:.3e} |mean|={_dmean:.3e}', flush=True)\n"
    "        except Exception as _e:\n"
    "            print(f'[DS4_HEAD_DBG] error: {_e!r}', flush=True)\n"
    "            import traceback as _tb; _tb.print_exc()\n"
)
s = s.replace(needle, inject, 1)
open(p, 'w').write(s)
print('[ds4] deepseek_v4.py outer head-dbg patched')
PY3
    find "$SITE_PACKAGES/vllm/model_executor/models/__pycache__" -name "deepseek_v4*" -delete 2>/dev/null || true
fi

# KV cache compressor-state spec fix.
# Vanilla SlidingWindowMLASpec bounds admission by max_num_batched_tokens, so
# the DSv4 compressor state cache (a fixed-size state-space model with a
# sliding_window of 8 or 128 tokens) gets sized as if it could hold up to
# max_num_batched_tokens worth of uncompressed state. For V4-Flash at
# max_model_len=16384, that's ~16 GiB of compressor state vs the ~50 MiB
# actually needed. Subclassing to bound by sliding_window only drops the
# pool to its architectural floor; pairs with the V4 paper §3.5.1 description
# of the state cache as a state-space model.
KV_INTERFACE_PY="$SITE_PACKAGES/vllm/v1/kv_cache_interface.py"
DEEPSEEK_COMPRESSOR_PY="$SITE_PACKAGES/vllm/model_executor/layers/deepseek_compressor.py"
if [ "$DS4_VLLM_OVERLAY" = "1" ]; then
    echo "[ds4] V1 monkey-patch skipped — overlay supplies CompressorStateMLASpec natively"
elif ! grep -q "DS4_KV_PATCH_V1" "$KV_INTERFACE_PY"; then
    echo "[ds4] Patching KV cache compressor-state spec (~300x over-allocation fix)"
    python3 - <<PY_KV
import os
kv_p = "$KV_INTERFACE_PY"
comp_p = "$DEEPSEEK_COMPRESSOR_PY"

# 1. Append CompressorStateMLASpec to kv_cache_interface.py.
s = open(kv_p).read()
assert "class SlidingWindowMLASpec(SlidingWindowSpec):" in s, \
    "expected SlidingWindowMLASpec class in kv_cache_interface.py"
assert "DS4_KV_PATCH_V1" not in s, "patch already applied (kv_cache_interface)"
new_class = '''


@dataclass(frozen=True, kw_only=True)
class CompressorStateMLASpec(SlidingWindowMLASpec):
    """DS4_KV_PATCH_V1 — DeepseekV4 compressor state cache spec.

    Vanilla SlidingWindowSpec.max_admission_blocks_per_request bounds by
    min(sliding_window - 1 + max_num_batched_tokens, max_model_len), under
    the chunked-prefill assumption that the cache may transiently hold up
    to max_num_batched_tokens uncompressed tokens. The DSv4 compressor
    state is a fixed-size state-space model (paper §3.5.1: "regarded as a
    sequence-specific state that depends solely on the current position")
    — it never holds more than sliding_window tokens regardless of batch
    size. Bounding by sliding_window drops the per-request startup budget
    by ~1300x for CSA (sliding_window=8, block_size=4) and ~120x for HCA
    (sliding_window=128, block_size=8).
    """

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        # +1 because the window may not start from a block boundary,
        # same convention as the parent class.
        return cdiv(self.sliding_window, self.block_size) + 1
'''
# Append at end of file (idempotent: check via DS4_KV_PATCH_V1 marker above).
with open(kv_p, "w") as f:
    f.write(s.rstrip() + new_class + "\n")
print("[ds4] kv_cache_interface.py: CompressorStateMLASpec appended")

# 2. Wire CompressorStateCache.get_kv_cache_spec to use it.
s = open(comp_p).read()
assert "DS4_KV_PATCH_V1" not in s, "patch already applied (deepseek_compressor)"
old_import = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    ")"
)
new_import = (
    "from vllm.v1.kv_cache_interface import (  # DS4_KV_PATCH_V1\n"
    "    CompressorStateMLASpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    ")"
)
assert old_import in s, "expected exact import block in deepseek_compressor.py"
s = s.replace(old_import, new_import, 1)

old_call = "return SlidingWindowMLASpec(  # only has one vector instead of K + V"
new_call = "return CompressorStateMLASpec(  # DS4_KV_PATCH_V1 (was SlidingWindowMLASpec) only has one vector instead of K + V"
assert old_call in s, "expected SlidingWindowMLASpec construction in CompressorStateCache.get_kv_cache_spec"
s = s.replace(old_call, new_call, 1)

with open(comp_p, "w") as f:
    f.write(s)
print("[ds4] deepseek_compressor.py: CompressorStateCache.get_kv_cache_spec rewired")

# 3. Wipe pycache so the patches load on next import.
import shutil
for d in ("$SITE_PACKAGES/vllm/v1/__pycache__",
          "$SITE_PACKAGES/vllm/model_executor/layers/__pycache__"):
    for f in os.listdir(d) if os.path.isdir(d) else []:
        if "kv_cache_interface" in f or "deepseek_compressor" in f:
            os.remove(os.path.join(d, f))
print("[ds4] KV cache patches applied; pycache wiped for affected modules")
PY_KV
fi

# KV cache SWA spec fix (Patch 2 in the V4 KV-layout series).
# DeepseekV4SWACache.get_kv_cache_spec returned vanilla SlidingWindowMLASpec,
# inheriting the same max_num_batched_tokens-based admission bound the C4/C128
# compressor states had pre-DS4_KV_PATCH_V1. The SWA cache is the same shape of
# bug: it's a fixed-size sliding-window buffer (sliding_window=128 tokens) but
# was sized as if it could transiently hold up to max_num_batched_tokens. With
# 43 SWA cache instances and 257 blocks/request × 37,440 B/block per instance,
# this contributes ~395 MiB per request — the dominant remaining KV
# over-allocation contributor after V1 (paper §3.5.1).
SPARSE_SWA_PY="$SITE_PACKAGES/vllm/v1/attention/backends/mla/sparse_swa.py"
if [ "$DS4_VLLM_OVERLAY" = "1" ]; then
    echo "[ds4] V2 monkey-patch skipped — overlay supplies DeepseekV4SWACache fix natively"
elif ! grep -q "DS4_KV_PATCH_V2" "$SPARSE_SWA_PY"; then
    echo "[ds4] Patching DeepseekV4SWACache spec (~85x SWA over-allocation fix)"
    python3 - <<PY_KV2
sw_p = "$SPARSE_SWA_PY"

s = open(sw_p).read()
assert "DS4_KV_PATCH_V2" not in s, "patch already applied (sparse_swa)"

# 1. Add CompressorStateMLASpec to the import.
old_import = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    ")"
)
new_import = (
    "from vllm.v1.kv_cache_interface import (  # DS4_KV_PATCH_V2\n"
    "    AttentionSpec,\n"
    "    CompressorStateMLASpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    ")"
)
assert old_import in s, "expected exact import block in sparse_swa.py"
s = s.replace(old_import, new_import, 1)

# 2. Swap SlidingWindowMLASpec -> CompressorStateMLASpec in
#    DeepseekV4SWACache.get_kv_cache_spec.
old_call = (
    "    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:\n"
    "        return SlidingWindowMLASpec(\n"
)
new_call = (
    "    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:\n"
    "        # DS4_KV_PATCH_V2: SWA cache is a fixed-size sliding-window state\n"
    "        # buffer (paper section 3.5.1); bound admission by sliding_window\n"
    "        # only via CompressorStateMLASpec. ~85x SWA over-allocation fix.\n"
    "        return CompressorStateMLASpec(\n"
)
assert old_call in s, "expected DeepseekV4SWACache.get_kv_cache_spec to return SlidingWindowMLASpec"
s = s.replace(old_call, new_call, 1)

with open(sw_p, "w") as f:
    f.write(s)
print("[ds4] sparse_swa.py: DeepseekV4SWACache.get_kv_cache_spec rewired")

# 3. Wipe pycache so the patch loads on next import.
import os
d = "$SITE_PACKAGES/vllm/v1/attention/backends/mla/__pycache__"
if os.path.isdir(d):
    for f in os.listdir(d):
        if "sparse_swa" in f:
            os.remove(os.path.join(d, f))
print("[ds4] DS4_KV_PATCH_V2 applied; pycache wiped")
PY_KV2
fi

# KV cache spec_manager_map registration fix (Patch V3).
# Without this, the new CompressorStateMLASpec subclass introduced by V1
# triggers a KeyError during EngineCore startup because spec_manager_map uses
# exact-type lookup (`type(spec)`) rather than isinstance — so the subclass
# isn't dispatched to SlidingWindowMLAManager. The KeyError kills the engine
# core silently, the API server times out and exits 0, and the container
# enters a restart loop. V1 + V2 looked fine in startup logs (KV cache pool
# numbers were emitted before the crash) but the engine never actually
# served — the reported concurrency improvements were mid-startup ghost
# numbers.
SINGLE_TYPE_MGR_PY="$SITE_PACKAGES/vllm/v1/core/single_type_kv_cache_manager.py"
if [ "$DS4_VLLM_OVERLAY" = "1" ]; then
    echo "[ds4] V3 monkey-patch skipped — overlay supplies spec_manager_map entry natively"
elif ! grep -q "DS4_KV_PATCH_V3" "$SINGLE_TYPE_MGR_PY"; then
    echo "[ds4] Patching spec_manager_map to register CompressorStateMLASpec"
    python3 - <<PY_KV3
mgr_p = "$SINGLE_TYPE_MGR_PY"

s = open(mgr_p).read()
assert "DS4_KV_PATCH_V3" not in s, "patch already applied (single_type_kv_cache_manager)"

# 1. Add CompressorStateMLASpec to the import.
old_import = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    ChunkedLocalAttentionSpec,\n"
    "    CrossAttentionSpec,\n"
    "    FullAttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MambaSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SinkFullAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    "    SlidingWindowSpec,\n"
    "    TQFullAttentionSpec,\n"
    ")"
)
new_import = (
    "from vllm.v1.kv_cache_interface import (  # DS4_KV_PATCH_V3\n"
    "    ChunkedLocalAttentionSpec,\n"
    "    CompressorStateMLASpec,\n"
    "    CrossAttentionSpec,\n"
    "    FullAttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MambaSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SinkFullAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    "    SlidingWindowSpec,\n"
    "    TQFullAttentionSpec,\n"
    ")"
)
assert old_import in s, "expected exact import block in single_type_kv_cache_manager.py"
s = s.replace(old_import, new_import, 1)

# 2. Add CompressorStateMLASpec to spec_manager_map dispatch dict.
# NOTE: lmxxf base image uses the older vllm where MLA specs reuse
# SlidingWindowManager / FullAttentionManager (no MLA-specific manager
# classes). Match THAT layout, not the local fork's newer layout.
old_map = (
    "    SlidingWindowMLASpec: SlidingWindowManager,\n"
    "    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,"
)
new_map = (
    "    SlidingWindowMLASpec: SlidingWindowManager,\n"
    "    # DS4_KV_PATCH_V3: CompressorStateMLASpec is a subclass of\n"
    "    # SlidingWindowMLASpec; spec_manager_map uses exact-type lookup so\n"
    "    # the entry is required. Reuses SlidingWindowManager (only the\n"
    "    # admission bound differs, computed via the spec method).\n"
    "    CompressorStateMLASpec: SlidingWindowManager,\n"
    "    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,"
)
assert old_map in s, "expected exact spec_manager_map block"
s = s.replace(old_map, new_map, 1)

with open(mgr_p, "w") as f:
    f.write(s)
print("[ds4] single_type_kv_cache_manager.py: CompressorStateMLASpec registered")

# 3. Wipe pycache so the patch loads on next import.
import os
d = "$SITE_PACKAGES/vllm/v1/core/__pycache__"
if os.path.isdir(d):
    for f in os.listdir(d):
        if "single_type_kv_cache_manager" in f:
            os.remove(os.path.join(d, f))
print("[ds4] DS4_KV_PATCH_V3 applied; pycache wiped")
PY_KV3
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
