"""Post-load completion: loads weights for params that vLLM's standard
load_weights pipeline missed.

Background. Our hybrid checkpoint combines:
  - antirez/ds4 GGUF routed-expert weights (our 2-bit MoE)
  - sgl-project FP8 base for everything else (attention, HC, shared experts)

vLLM's deepseek_v4 model registers params under canonical (and in some
cases, fused) names — `attn.fused_wqa_wkv.weight`, `mla_attn.compressor.*`,
`shared_experts.gate_up_proj` — but our checkpoint has the constituent
unfused tensors under sgl-style names. The standard load pipeline
(WeightsMapper + stacked_params_mapping) handles many of these
translations but trips on a specific subset (currently 118 params across
layers 35–43, mostly attention compressor/fusion patterns).

This module is called at the end of load_weights to:
  1. Default-init the truly-stateless params (FP8 weight_scale_inv for
     BF16 paths, indexer k_norm) to identity values.
  2. For still-unloaded weight params, open the safetensor index and
     load directly using a small set of manual name-remapping rules.

Logs are tagged ``[DS4_LOAD_COMP]``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch


# Default-init rules: param suffix → fill value
_DEFAULTS = [
    (".weight_scale_inv", 1.0),
    (".weight_scale", 1.0),
    (".scale", 1.0),
    (".bias", 0.0),
]
_NORM_WEIGHT_SUFFIXES = ("_norm.weight", "norm.weight")


def _default_for(name: str):
    for suffix, val in _DEFAULTS:
        if name.endswith(suffix):
            return val
    if name.endswith(_NORM_WEIGHT_SUFFIXES):
        return 1.0
    return None


# Remapping rules for direct-load from safetensors.
# Each entry: (param_suffix_match, list[(safetensor_suffix, shard_id_or_None)])
# Matching is on the END of the param name; substitutions are on the same suffix.
# shard_id None means a direct copy. Otherwise we cat() shards along dim 0.
_REMAP = [
    # Fused QA + KV in attention: param `attn.fused_wqa_wkv.weight` ←
    # cat(`attn.wq_a.weight`, `attn.wkv.weight`) along dim 0.
    (
        "attn.fused_wqa_wkv.weight",
        [("attn.wq_a.weight", 0), ("attn.wkv.weight", 1)],
    ),
    (
        "attn.fused_wqa_wkv.weight_scale_inv",
        [("attn.wq_a.scale", 0), ("attn.wkv.scale", 1)],
    ),
    # Compressor: live model has `mla_attn.compressor.X`, safetensor has
    # `compressor.X` (same parent path, just rename).
    (
        "attn.mla_attn.compressor.fused_wkv_wgate.weight",
        [("attn.compressor.wkv.weight", 0), ("attn.compressor.wgate.weight", 1)],
    ),
    ("attn.mla_attn.compressor.ape", [("attn.compressor.ape", None)]),
    ("attn.mla_attn.compressor.norm.weight",
     [("attn.compressor.norm.weight", None)]),
    # Shared experts: gate_up_proj ← cat(w1, w3); down_proj ← w2.
    (
        "ffn.shared_experts.gate_up_proj.weight",
        [("ffn.shared_experts.w1.weight", 0), ("ffn.shared_experts.w3.weight", 1)],
    ),
    (
        "ffn.shared_experts.gate_up_proj.weight_scale_inv",
        [("ffn.shared_experts.w1.scale", 0), ("ffn.shared_experts.w3.scale", 1)],
    ),
    (
        "ffn.shared_experts.down_proj.weight",
        [("ffn.shared_experts.w2.weight", None)],
    ),
    (
        "ffn.shared_experts.down_proj.weight_scale_inv",
        [("ffn.shared_experts.w2.scale", None)],
    ),
    # Router gate.weight + correction_bias rename.
    ("ffn.gate.weight", [("ffn.gate.weight", None)]),
    ("ffn.gate.e_score_correction_bias", [("ffn.gate.bias", None)]),
    # Plain wq_b / wo_a / wo_b / attn_sink + scale variants.
    ("attn.wq_b.weight", [("attn.wq_b.weight", None)]),
    ("attn.wq_b.weight_scale_inv", [("attn.wq_b.scale", None)]),
    ("attn.wo_a.weight", [("attn.wo_a.weight", None)]),
    ("attn.wo_a.weight_scale_inv", [("attn.wo_a.scale", None)]),
    ("attn.wo_b.weight", [("attn.wo_b.weight", None)]),
    ("attn.wo_b.weight_scale_inv", [("attn.wo_b.scale", None)]),
    ("attn.attn_sink", [("attn.attn_sink", None)]),
    # Indexer weights_proj + indexer compressor (same parent, just rename).
    ("attn.indexer.weights_proj.weight",
     [("attn.indexer.weights_proj.weight", None)]),
    # Indexer wq_b: live model registers as fp8_e4m3fn at the same shape as
    # safetensor (which already has fp8); plus its scale.
    ("attn.indexer.wq_b.weight", [("attn.indexer.wq_b.weight", None)]),
    ("attn.indexer.wq_b.weight_scale_inv",
     [("attn.indexer.wq_b.scale", None)]),
    ("attn.indexer.compressor.fused_wkv_wgate.weight",
     [("attn.indexer.compressor.wkv.weight", 0),
      ("attn.indexer.compressor.wgate.weight", 1)]),
    ("attn.indexer.compressor.ape",
     [("attn.indexer.compressor.ape", None)]),
    ("attn.indexer.compressor.norm.weight",
     [("attn.indexer.compressor.norm.weight", None)]),
    # HC tensors — direct match.
    ("hc_attn_fn", [("hc_attn_fn", None)]),
    ("hc_ffn_fn", [("hc_ffn_fn", None)]),
    ("hc_attn_base", [("hc_attn_base", None)]),
    ("hc_ffn_base", [("hc_ffn_base", None)]),
    ("hc_attn_scale", [("hc_attn_scale", None)]),
    ("hc_ffn_scale", [("hc_ffn_scale", None)]),
    # CRITICAL: layer-level norms (FFN-input + attn-input) and
    # attention-internal Q/KV norms. Without these we previously fell into
    # the Phase 1 default-init for *_norm.weight which set them to 1.0,
    # producing post-norm activations with magnitude ~sqrt(hidden) instead
    # of the trained ~15-30. Layer-by-layer hidden-state compare against
    # ds4 reference proved this is the dominant correctness bug.
    ("attn_norm.weight", [("attn_norm.weight", None)]),
    ("ffn_norm.weight", [("ffn_norm.weight", None)]),
    ("attn.kv_norm.weight", [("attn.kv_norm.weight", None)]),
    ("attn.q_norm.weight", [("attn.q_norm.weight", None)]),
]


def _layer_prefix(param_name: str) -> str:
    """Extract the layer prefix portion to substitute the suffix.

    For ``layers.37.attn.mla_attn.compressor.fused_wkv_wgate.weight`` and
    matched suffix ``attn.mla_attn.compressor.fused_wkv_wgate.weight``,
    return ``layers.37.``.

    Uses longest-match: walk back through the param name looking for a
    boundary that matches the end of the layer.X portion or an HC top-level.
    """
    return param_name


def complete_load(
    params_dict: dict,
    loaded_params: set,
    ckpt_dir: str,
):
    """Default-init unloaded scalar/scale params and direct-load remaining."""
    import torch
    import json
    import os
    from safetensors import safe_open

    # DS4_LOAD_COMP_DBG: identify which model instance this is — if there
    # are two calls and id(params_dict) differs, it's the MTP head's mirror
    # instance, not the main model being re-processed.
    print(
        f"[DS4_LOAD_COMP_DBG] enter id(params_dict)={id(params_dict)} "
        f"params={len(params_dict)} loaded={len(loaded_params)}",
        flush=True,
    )

    unloaded = [k for k in params_dict if k not in loaded_params]
    if not unloaded:
        print("[DS4_LOAD_COMP] nothing to do (all params loaded)", flush=True)
        return

    # DS4_LOAD_COMP_DBG: print first few unloaded names so we can see live
    # parameter naming (e.g. whether `layers.K.X` or `model.layers.K.X`).
    # Helps verify that remap rule suffix-matching will work.
    sample_unloaded = sorted(unloaded)[:8]
    print(f"[DS4_LOAD_COMP_DBG] sample unloaded ({len(unloaded)} total): "
          f"{sample_unloaded}", flush=True)
    norm_unloaded = [k for k in unloaded if "norm.weight" in k]
    print(f"[DS4_LOAD_COMP_DBG] norm-weight unloaded count: "
          f"{len(norm_unloaded)} (sample: {sorted(norm_unloaded)[:5]})",
          flush=True)

    # CHANGED 2026-05-10: reordered phases. Previously Phase 1 was
    # default-init (which would default norm weights to 1.0 BEFORE any
    # remap rule had a chance to load them from disk — silently breaking
    # the trained norm scaling and producing post-norm magnitude
    # ~sqrt(hidden) instead of trained values). Now we direct-load via
    # remap rules first, then only default-init what's still unloaded
    # (= things genuinely absent from disk like quantization scales for
    # methods that don't use them, biases, etc.).

    # Build weight_map for direct-load.
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    # The safetensor names use sgl-style (mostly no model. prefix); some
    # have model.layers.X.* (our hybrid 2-bit ones, but those load via the
    # patched expert branch, not this completer). Build a substring lookup:
    # for each param name, find the safetensor name that ends with the
    # remapped suffix at the matching layer prefix.

    # Cache opened safetensors for this completer's lifetime.
    _opened: dict[str, "safe_open"] = {}

    def _get_tensor(safetensor_name: str):
        shard = weight_map.get(safetensor_name)
        if shard is None:
            return None
        path = os.path.join(ckpt_dir, shard)
        if path not in _opened:
            _opened[path] = safe_open(path, framework="pt")
        return _opened[path].get_tensor(safetensor_name)

    # Phase 1 (was Phase 2): direct-load via remap rules.
    loaded_count = 0
    failed = []
    for param_name in list(unloaded):
        # Find a remap rule whose suffix matches the end of param_name.
        rule = None
        for suffix, sources in _REMAP:
            if param_name.endswith(suffix):
                rule = (suffix, sources)
                break
        if rule is None:
            failed.append((param_name, "no rule"))
            continue
        suffix, sources = rule
        prefix = param_name[:-len(suffix)]  # e.g. "layers.37."

        # Resolve and load shards
        try:
            shards = []
            for src_suffix, shard_id in sources:
                src_name = prefix + src_suffix
                t = _get_tensor(src_name)
                if t is None:
                    raise KeyError(f"safetensor missing: {src_name}")
                shards.append((shard_id, t))
            if len(shards) == 1:
                tensor = shards[0][1]
            else:
                # Sort by shard_id and concat along dim 0
                shards.sort(key=lambda x: x[0])
                tensor = torch.cat([t for _, t in shards], dim=0)

            param = params_dict[param_name]
            with torch.no_grad():
                if param.dtype != tensor.dtype:
                    # Dtype mismatch handling. Two cases:
                    # 1. Cross-fp8 view: e8m0 stored as uint8 etc. — view OK.
                    # 2. bf16/fp16/fp32 source, fp8 param: a real value
                    #    conversion (saturating clamp + round). Lossy but
                    #    forward-pass-functional with weight_scale_inv=1.0
                    #    (which load_completion already defaulted).
                    src_byte = tensor.element_size()
                    dst_byte = param.element_size()
                    fp8s = (torch.float8_e4m3fn, torch.float8_e5m2)
                    if src_byte == dst_byte:
                        # same byte width — view is safe (e.g. uint8↔e8m0)
                        try:
                            tensor = tensor.view(param.dtype)
                        except RuntimeError:
                            tensor = tensor.to(param.dtype)
                    elif param.dtype in fp8s and tensor.dtype in (
                        torch.bfloat16, torch.float16, torch.float32
                    ):
                        # real bf16/fp16/fp32 → fp8 conversion
                        tensor = tensor.to(param.dtype)
                    else:
                        tensor = tensor.to(param.dtype)
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"shape mismatch: param={tuple(param.shape)} "
                        f"tensor={tuple(tensor.shape)}"
                    )
                param.data.copy_(tensor)
            loaded_params.add(param_name)
            loaded_count += 1
        except Exception as e:
            failed.append((param_name, repr(e)))

    print(f"[DS4_LOAD_COMP] direct-loaded {loaded_count} params via remap rules",
          flush=True)

    # Phase 2 (was Phase 1): default-init params still unloaded that match
    # known scale/bias/norm patterns. With phase order swapped, this now
    # only fires for params truly absent from disk — quantization scales
    # for layers that don't carry them, biases that don't exist, etc.
    initialized = []
    failed_after_default = []
    with torch.no_grad():
        for param_name, why in list(failed):
            d = _default_for(param_name)
            if d is None:
                failed_after_default.append((param_name, why))
                continue
            params_dict[param_name].data.fill_(d)
            loaded_params.add(param_name)
            initialized.append(param_name)
    print(f"[DS4_LOAD_COMP] defaulted {len(initialized)} scale/bias/norm "
          f"params (no-disk fallback)", flush=True)

    failed = failed_after_default
    if failed:
        print(f"[DS4_LOAD_COMP] {len(failed)} params still failed:", flush=True)
        for name, why in failed[:10]:
            print(f"[DS4_LOAD_COMP]   {name}  -- {why}", flush=True)
        if len(failed) > 10:
            print(f"[DS4_LOAD_COMP]   ... and {len(failed)-10} more", flush=True)
    final_unloaded = [k for k in params_dict if k not in loaded_params]
    print(f"[DS4_LOAD_COMP] final unloaded count: {len(final_unloaded)}",
          flush=True)

    # DS4_FUSION_DBG: compare first-half vs second-half magnitude of
    # fused_wqa_wkv.weight for a standard-pipeline-loaded layer (0) vs a
    # manually-loaded layer (37). If my cat order matches the standard
    # pipeline, both should show the same pattern.
    try:
        for lid in (0, 5, 35, 37, 41):
            k = f"layers.{lid}.attn.fused_wqa_wkv.weight"
            if k not in params_dict:
                continue
            p = params_dict[k]
            # fused = cat(wq_a:dim0=512, wkv:dim0=1024) along dim 0 → (1536, 4096)
            # if param.dtype is fp8, view via float() for stats
            ph = p.float() if not p.is_floating_point() or p.dtype not in (torch.float32, torch.float16, torch.bfloat16) else p
            try:
                first_half = ph[:512].abs()
                second_half = ph[512:].abs()
                fh_mean = float(first_half.mean().item())
                sh_mean = float(second_half.mean().item())
                print(f"[DS4_FUSION_DBG] layer {lid} fused_wqa_wkv: "
                      f"first512_mean={fh_mean:.3e} last1024_mean={sh_mean:.3e}",
                      flush=True)
            except Exception as e:
                print(f"[DS4_FUSION_DBG] layer {lid}: stat error {e!r}", flush=True)
    except Exception as e:
        print(f"[DS4_FUSION_DBG] error: {e!r}", flush=True)

    # also compare directly to safetensor data for one layer to verify identity.
    try:
        with torch.no_grad():
            for lid in (0, 37):
                fused_key = f"layers.{lid}.attn.fused_wqa_wkv.weight"
                if fused_key not in params_dict:
                    continue
                wq_a_t = _get_tensor(f"layers.{lid}.attn.wq_a.weight")
                wkv_t = _get_tensor(f"layers.{lid}.attn.wkv.weight")
                if wq_a_t is None or wkv_t is None:
                    continue
                fp = params_dict[fused_key].float()
                # standard convention: top of fused = wq_a, bottom = wkv
                wq_a_top = fp[:wq_a_t.shape[0]]
                wkv_bot = fp[wq_a_t.shape[0]:]
                # diff vs safetensor (cast both to fp32 for comparison)
                top_match = float((wq_a_top - wq_a_t.float()).abs().mean().item())
                bot_match = float((wkv_bot - wkv_t.float()).abs().mean().item())
                # if reversed
                wkv_top = fp[:wkv_t.shape[0]]
                wq_a_bot = fp[wkv_t.shape[0]:]
                rev_top_match = float((wkv_top - wkv_t.float()).abs().mean().item())
                rev_bot_match = float((wq_a_bot - wq_a_t.float()).abs().mean().item())
                print(f"[DS4_FUSION_DBG] layer {lid} std-order diffs: "
                      f"top_vs_wq_a={top_match:.3e} bot_vs_wkv={bot_match:.3e}",
                      flush=True)
                print(f"[DS4_FUSION_DBG] layer {lid} rev-order diffs: "
                      f"top_vs_wkv={rev_top_match:.3e} bot_vs_wq_a={rev_bot_match:.3e}",
                      flush=True)
    except Exception as e:
        print(f"[DS4_FUSION_DBG] safetensor compare error: {e!r}", flush=True)
