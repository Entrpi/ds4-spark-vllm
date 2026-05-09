"""FusedMoE method implementing the routed IQ2_XXS gate/up + Q2_K down path.

Weight tensors registered on the layer (matching the converter's output
naming):

    w13_iq2xxs_qs : uint8   (n_experts, 2*intermediate, n_blocks_in, 64)
    w13_iq2xxs_d  : float16 (n_experts, 2*intermediate, n_blocks_in)

    w2_q2k_qs     : uint8   (n_experts, hidden,         n_blocks_int, 64)
    w2_q2k_scales : uint8   (n_experts, hidden,         n_blocks_int, 16)
    w2_q2k_d      : float16 (n_experts, hidden,         n_blocks_int)
    w2_q2k_dmin   : float16 (n_experts, hidden,         n_blocks_int)

Where ``n_blocks_in = hidden / 256`` and ``n_blocks_int = intermediate / 256``.

The forward (``apply``) does, per token:

    1. Quantize the activation x to Q8_K  (uses :func:`quantize_q8_K_triton`)
    2. For each selected expert e:
         a. Compute mid = silu(gate_e . x_q) * (up_e . x_q)
            via :func:`iq2_xxs_pair_dot_triton` (one call per expert, two
            outputs per row)
         b. Quantize mid to Q8_K
    3. For each output row r of the down projection:
         out[r] = sum_e (down_e[r] . mid_q[e])
         via :func:`q2_K_accum_dot_triton`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
    from torch import nn
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.utils import set_weight_attrs

    HAVE_VLLM = True
except ImportError:  # pragma: no cover
    HAVE_VLLM = False
    FusedMoEMethodBase = object  # type: ignore[assignment, misc]

from ..lookup_tables import QK_K

# DS4_FWD_DBG: module-level counter so apply() calls have a sequential id.
# Useful for tracing magnitude evolution layer-by-layer during a single
# forward pass (DSv4-Flash has 43 layers → expect 43 calls per forward).
_apply_call_count = 0

if TYPE_CHECKING:  # pragma: no cover
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
    )


class Iq2XxsQ2KFusedMoEMethod(FusedMoEMethodBase):
    """vLLM FusedMoE method implementing the ds4 2-bit recipe."""

    def __init__(self, moe: "FusedMoEConfig") -> None:
        super().__init__(moe)
        # DS4_TRACE: confirm we're being instantiated; one print per instance.
        self._ds4_inst_id = id(self)
        print(
            f"[DS4_TRACE] __init__ Iq2XxsQ2KFusedMoEMethod inst_id={self._ds4_inst_id}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Weight registration
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: "nn.Module",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: "torch.dtype",
        **extra_weight_attrs,
    ) -> None:
        # DS4_TRACE: log every layer that gets our quant method.
        _layer_prefix = getattr(layer, "prefix", "?")
        print(
            f"[DS4_TRACE] create_weights inst={self._ds4_inst_id} "
            f"layer_id={id(layer)} layer.prefix={_layer_prefix!r} "
            f"num_experts={num_experts} hidden={hidden_size} "
            f"intermediate={intermediate_size_per_partition}",
            flush=True,
        )
        if hidden_size % QK_K != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be a multiple of {QK_K}"
            )
        if intermediate_size_per_partition % QK_K != 0:
            raise ValueError(
                f"intermediate_size {intermediate_size_per_partition} must "
                f"be a multiple of {QK_K}"
            )

        n_blocks_in = hidden_size // QK_K
        n_blocks_int = intermediate_size_per_partition // QK_K
        two_int = 2 * intermediate_size_per_partition

        # Gate+up packed: 2 rows of intermediate per expert (gate then up).
        layer.register_parameter(
            "w13_iq2xxs_qs",
            nn.Parameter(
                torch.empty(num_experts, two_int, n_blocks_in, 64,
                            dtype=torch.uint8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_iq2xxs_d",
            nn.Parameter(
                torch.empty(num_experts, two_int, n_blocks_in,
                            dtype=torch.float16),
                requires_grad=False,
            ),
        )

        # Down projection: hidden output rows per expert.
        layer.register_parameter(
            "w2_q2k_qs",
            nn.Parameter(
                torch.empty(num_experts, hidden_size, n_blocks_int, 64,
                            dtype=torch.uint8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_q2k_scales",
            nn.Parameter(
                torch.empty(num_experts, hidden_size, n_blocks_int, 16,
                            dtype=torch.uint8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_q2k_d",
            nn.Parameter(
                torch.empty(num_experts, hidden_size, n_blocks_int,
                            dtype=torch.float16),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_q2k_dmin",
            nn.Parameter(
                torch.empty(num_experts, hidden_size, n_blocks_int,
                            dtype=torch.float16),
                requires_grad=False,
            ),
        )

        # Set extra weight attrs but override the weight_loader. vLLM's
        # FusedMoE passes its per-expert-fusing loader in extra_weight_attrs;
        # our tensors are already pre-fused on the expert dim (shape
        # (E, 2I, ...) etc.), so we want a plain pass-through copy. Without
        # this override, the FusedMoE loader silently does the wrong thing
        # for unrecognized source names and leaves our params at empty()'s
        # uninitialized memory — manifests as fp16-max-valued garbage and
        # NaN/Inf at apply() time.
        def _passthrough_loader(param, loaded_weight, *args, **kwargs):
            # DS4_LOAD_DEBUG: dump first call's view of loaded_weight to verify
            # whether the bytes we receive are already corrupt (fastsafetensors)
            # or whether corruption happens later (e.g. param being reset).
            try:
                if not getattr(_passthrough_loader, "_dumped", False):
                    if torch.is_floating_point(loaded_weight):
                        nf = int((~torch.isfinite(loaded_weight)).sum().item())
                    else:
                        nf = -1
                    print(
                        f"[ds4_load_dbg/loader] FIRST call: "
                        f"param.shape={tuple(param.shape)} param.dtype={param.dtype} "
                        f"lw.shape={tuple(loaded_weight.shape)} lw.dtype={loaded_weight.dtype} "
                        f"non_finite_in_lw={nf} args={args} kwargs={list(kwargs.keys())}",
                        flush=True,
                    )
                    _passthrough_loader._dumped = True
            except Exception as e:
                print(f"[ds4_load_dbg/loader] dump error: {e!r}", flush=True)
            param.data.copy_(loaded_weight)

        attrs_no_loader = {
            k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"
        }
        for name in (
            "w13_iq2xxs_qs", "w13_iq2xxs_d",
            "w2_q2k_qs", "w2_q2k_scales", "w2_q2k_d", "w2_q2k_dmin",
        ):
            param = getattr(layer, name)
            set_weight_attrs(param, attrs_no_loader)
            param.weight_loader = _passthrough_loader

    def get_fused_moe_quant_config(
        self, layer: "nn.Module",
    ) -> "FusedMoEQuantConfig | None":
        # No standard FusedMoEQuantConfig matches this scheme; we own the
        # full forward path.
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def apply(
        self,
        layer: "FusedMoE",
        x: "torch.Tensor",                         # (T, hidden)
        topk_weights: "torch.Tensor",              # (T, top_k)
        topk_ids: "torch.Tensor",                  # (T, top_k)
        shared_experts_input: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Run the IQ2_XXS+Q2_K MoE forward.

        For the first cut we materialize per-token-per-expert work via
        Python loops over selected experts; this is correct but slow.
        Performance tuning (one fused launch per layer) is a v2 task
        once correctness is validated against ds4 logits.
        """
        from ..triton_kernels.q8_K_quantize import quantize_q8_K_triton
        from ..triton_kernels.iq2_xxs_pair_dot import iq2_xxs_pair_dot_triton
        from ..triton_kernels.q2_K_accum_dot import q2_K_accum_dot_triton

        T, hidden = x.shape
        top_k = topk_ids.shape[1]
        device = x.device

        n_experts = layer.w13_iq2xxs_qs.shape[0]
        intermediate = layer.w13_iq2xxs_qs.shape[1] // 2
        n_blocks_in = hidden // QK_K
        n_blocks_int = intermediate // QK_K

        # DS4_TRACE: first-call-per-layer marker — proves apply() is reaching
        # this specific FusedMoE instance.
        if not getattr(layer, "_ds4_apply_first_logged", False):
            layer._ds4_apply_first_logged = True
            print(
                f"[DS4_TRACE] apply() FIRST inst={self._ds4_inst_id} "
                f"layer_id={id(layer)} layer.prefix={getattr(layer, 'prefix', '?')!r} "
                f"x.shape={tuple(x.shape)} x.dtype={x.dtype} "
                f"topk_ids.shape={tuple(topk_ids.shape)}",
                flush=True,
            )

        out = torch.zeros((T, hidden), dtype=x.dtype, device=device)

        # DS4_TRACE: file-toggled no-op mode. If /logs/ds4_moe_noop exists,
        # skip the entire MoE computation and return zeros. Comparing the
        # model output between normal and no-op modes proves whether our
        # apply() result is actually plumbed into the residual stream.
        import os as _os
        if _os.path.exists("/logs/ds4_moe_noop"):
            if T <= 16:
                print(
                    f"[DS4_TRACE] apply() NOOP MODE: returning zeros for "
                    f"T={T} hidden={hidden}",
                    flush=True,
                )
            return out

        # DS4_NAN_DEBUG: helper to fail fast at the first NaN/Inf and
        # report the call-site stage name. Disabled at end-to-end perf
        # but invaluable for first-light debugging.
        def _nanchk(t_, name):
            if torch.is_floating_point(t_):
                if torch.isnan(t_).any() or torch.isinf(t_).any():
                    bad_n = int(torch.isnan(t_).sum().item())
                    bad_i = int(torch.isinf(t_).sum().item())
                    finite = t_[torch.isfinite(t_)]
                    fmin = float(finite.min().item()) if finite.numel() else float("nan")
                    fmax = float(finite.max().item()) if finite.numel() else float("nan")
                    raise RuntimeError(
                        f"[ds4_apply] NaN/Inf at {name}: shape={tuple(t_.shape)} "
                        f"dtype={t_.dtype} nans={bad_n} infs={bad_i} "
                        f"finite_min={fmin:.4e} finite_max={fmax:.4e}"
                    )

        # DS4_LOAD_DEBUG: one-shot dump on first apply() call to verify
        # the weights actually loaded (vs uninitialized empty() memory).
        # Compares expert 0 (sanity-checked clean on disk) to expert 254
        # (the one that's been NaN'ing) for w13_iq2xxs_d, the most
        # diagnostic tensor (fp16, small, all finite when correctly
        # loaded).
        if not getattr(layer, "_ds4_dumped", False):
            wl = getattr(layer.w13_iq2xxs_qs, "weight_loader", None)
            wl_name = getattr(wl, "__name__", repr(wl))
            wl_qual = getattr(wl, "__qualname__", "?")
            d = layer.w13_iq2xxs_d
            d0_finite = int(torch.isfinite(d[0]).sum().item())
            d0_max = float(d[0].abs().max().item())
            d0_min = float(d[0].abs().min().item())
            d254_finite = int(torch.isfinite(d[254]).sum().item())
            d254_max = float(d[254].abs().max().item())
            d254_min = float(d[254].abs().min().item())
            d254_nans = int(torch.isnan(d[254]).sum().item())
            print(
                f"[ds4_load_dbg] weight_loader={wl_name} ({wl_qual})  "
                f"E0: finite={d0_finite}/{d[0].numel()} |d|_min={d0_min:.3e} |d|_max={d0_max:.3e}  "
                f"E254: finite={d254_finite}/{d[254].numel()} |d|_min={d254_min:.3e} "
                f"|d|_max={d254_max:.3e} nans={d254_nans}",
                flush=True,
            )
            layer._ds4_dumped = True

        # DS4_FWD_DBG: per-call magnitude trace, only for small batches
        # (i.e. real inference, not profile_run dummies which use T=64+).
        global _apply_call_count
        _apply_call_count += 1
        _call_n = _apply_call_count
        # DS4_ROUTE: per-call routing entropy, gated by /logs/ds4_route_arm
        # existence so the dump only fires during real inference (not
        # warmup / autotuning). Touch the arm file before curl, rm after.
        # Also extracts layer index from layer.prefix so we can correlate
        # entropy across the 43-layer stack.
        try:
            import os as _os
            _arm = _os.path.exists("/logs/ds4_route_arm")
        except Exception:
            _arm = False
        if T <= 16 and _arm:
            try:
                import re as _re
                _prefix = getattr(layer, "prefix", "") or ""
                _lidx_attr = getattr(layer, "layer_idx", None)
                _m = _re.search(r"layers\.(\d+)", _prefix)
                _lidx = (
                    _lidx_attr
                    if _lidx_attr is not None
                    else (int(_m.group(1)) if _m else -1)
                )
                _tw = topk_weights.detach().float()
                # Normalize per-token (since vLLM's topk_weights are scaled
                # by routed_scaling_factor and don't sum to 1).
                _tw_norm = _tw / _tw.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                _ent = -(_tw_norm * _tw_norm.clamp(min=1e-12).log()).sum(dim=-1)
                _ent_mean = float(_ent.mean().item())
                _max_ent = float(torch.log(torch.tensor(float(top_k))).item())
                _max_w = float(_tw_norm.max().item())
                _min_w = float(_tw_norm.min().item())
                # Show top-1 expert IDs across tokens (for variety check)
                _top1 = topk_weights.argmax(dim=-1)
                _top1_ids = topk_ids.gather(-1, _top1.unsqueeze(-1)).squeeze(-1).tolist()
                # Once-per-layer prefix dump so we can map call→layer
                if not getattr(layer, "_ds4_prefix_dumped", False):
                    layer._ds4_prefix_dumped = True
                    print(
                        f"[DS4_ROUTE_PFX] call={_call_n} layer={_lidx} prefix={_prefix!r}",
                        flush=True,
                    )
                print(
                    f"[DS4_ROUTE] call={_call_n} layer={_lidx} T={T} "
                    f"ent={_ent_mean:.3f}/{_max_ent:.3f} "
                    f"max_w={_max_w:.3f} min_w={_min_w:.3f} "
                    f"top1_ids={_top1_ids}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[DS4_ROUTE] error: {_e!r}", flush=True)
        if T <= 16:
            _xfin = x[torch.isfinite(x)] if torch.is_floating_point(x) else x.float()
            if _xfin.numel():
                _xmax = float(_xfin.abs().max().item())
                _xmean = float(_xfin.abs().mean().item())
            else:
                _xmax = _xmean = float("nan")
            print(
                f"[DS4_FWD #{_call_n}] T={T} hidden={hidden} |x|max={_xmax:.3e} "
                f"|x|mean={_xmean:.3e}",
                flush=True,
            )

        _nanchk(x, "input x")

        # Quantize all token activations in one shot.
        x_blocks = x.reshape(T * n_blocks_in, QK_K).to(torch.float32)
        _nanchk(x_blocks, "x_blocks (post fp32 cast)")
        x_q_qs, x_q_d, x_q_bsums = quantize_q8_K_triton(x_blocks)
        _nanchk(x_q_d, "x_q_d (Q8_K activation scale)")
        x_q_qs = x_q_qs.reshape(T, n_blocks_in, QK_K)
        x_q_d = x_q_d.reshape(T, n_blocks_in)
        x_q_bsums = x_q_bsums.reshape(T, n_blocks_in, 16)

        for t in range(T):
            for k in range(top_k):
                expert = int(topk_ids[t, k].item())
                weight = topk_weights[t, k]

                # Gate / up: w13 is laid out as [n_experts, gate_rows | up_rows, ...]
                gate_qs = layer.w13_iq2xxs_qs[expert, :intermediate]
                gate_d = layer.w13_iq2xxs_d[expert, :intermediate]
                up_qs = layer.w13_iq2xxs_qs[expert, intermediate:]
                up_d = layer.w13_iq2xxs_d[expert, intermediate:]
                _nanchk(gate_d, f"gate_d (t={t},k={k},expert={expert})")
                _nanchk(up_d,   f"up_d   (t={t},k={k},expert={expert})")

                gate_out, up_out = iq2_xxs_pair_dot_triton(
                    gate_qs, gate_d, up_qs, up_d,
                    x_q_qs[t], x_q_d[t],
                )
                _nanchk(gate_out, f"gate_out post iq2_pair (t={t},k={k},expert={expert})")
                _nanchk(up_out,   f"up_out   post iq2_pair (t={t},k={k},expert={expert})")

                # SwiGLU: silu(gate) * up.
                mid = torch.nn.functional.silu(gate_out) * up_out
                _nanchk(mid, f"mid post SwiGLU (t={t},k={k},expert={expert})")

                # Quantize mid to Q8_K.
                mid_blocks = mid.reshape(n_blocks_int, QK_K).to(torch.float32)
                _nanchk(mid_blocks, f"mid_blocks (t={t},k={k},expert={expert})")
                mid_qs, mid_d, mid_bsums = quantize_q8_K_triton(mid_blocks)
                _nanchk(mid_d, f"mid_d post Q8_K quant (t={t},k={k},expert={expert})")

                # Down projection (single expert via accum kernel with n_experts=1).
                w_scales = layer.w2_q2k_scales[expert].unsqueeze(0)
                w_qs = layer.w2_q2k_qs[expert].unsqueeze(0)
                w_d = layer.w2_q2k_d[expert].unsqueeze(0)
                w_dmin = layer.w2_q2k_dmin[expert].unsqueeze(0)
                _nanchk(w_d,    f"w_d    (t={t},k={k},expert={expert})")
                _nanchk(w_dmin, f"w_dmin (t={t},k={k},expert={expert})")

                down_out = q2_K_accum_dot_triton(
                    w_scales, w_qs, w_d, w_dmin,
                    mid_qs.unsqueeze(0), mid_d.unsqueeze(0),
                    mid_bsums.unsqueeze(0),
                )
                _nanchk(down_out, f"down_out post q2_K (t={t},k={k},expert={expert})")
                _nanchk(weight,   f"topk weight (t={t},k={k},expert={expert})")
                out[t].add_(down_out.to(x.dtype) * weight)
                _nanchk(out[t], f"out[t] post accum (t={t},k={k},expert={expert})")

        _nanchk(out, "final out")

        # DS4_FWD_DBG: dump output magnitude paired with the input one above.
        # Plus a checksum (sum of squares of float32 values) so we can detect
        # whether two calls with the same input produce the same output.
        if T <= 16:
            _ofin = out[torch.isfinite(out)]
            if _ofin.numel():
                _omax = float(_ofin.abs().max().item())
                _omean = float(_ofin.abs().mean().item())
            else:
                _omax = _omean = float("nan")
            _ocksum = float(out.float().pow(2).sum().item())
            print(
                f"[DS4_FWD #{_call_n}] OUT |out|max={_omax:.3e} "
                f"|out|mean={_omean:.3e} sum2={_ocksum:.6e}",
                flush=True,
            )

        return out
