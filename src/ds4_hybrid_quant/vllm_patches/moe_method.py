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

The forward (``apply``) groups tokens by selected expert and dispatches
one batched Triton call per active expert:

    1. Quantize all T token activations to Q8_K in one batched launch.
    2. argsort tokens by expert id, build per-expert offset table via
       bincount + cumsum (single ``.cpu()`` sync per layer).
    3. For each expert e with M_e routed tokens:
         a. gate_out, up_out = iq2_xxs_pair_dot_triton(W13[e], x_q[tokens_e])
            -- shape (M_e, intermediate)
         b. mid = silu(gate_out) * up_out, quantized to Q8_K
         c. down_out = q2_K_accum_dot_triton(W2[e], mid_q)
            -- shape (M_e, hidden)
         d. out.index_add_(tokens_e, down_out * topk_weights_e)
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

# Module-level call counter used by the env-gated DS4_ROUTE / DS4_HDUMP
# diagnostic paths in apply(). DSv4-Flash has 43 layers → expect 43
# increments per forward.
_apply_call_count = 0

if TYPE_CHECKING:  # pragma: no cover
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
    )


class Iq2XxsQ2KFusedMoEMethod(FusedMoEMethodBase):
    """vLLM FusedMoE method implementing the ds4 2-bit recipe."""

    # DS4_LAYER_SEQ: class-level registry mapping id(layer) -> sequence
    # index, populated in create_weights. Needed because layer.prefix and
    # layer.layer_idx are not set on FusedMoE in our build (both None).
    # vLLM constructs DSv4 layers 0..42 in order, so the create_weights
    # call sequence corresponds to layer indices.
    _ds4_layer_seq: dict = {}

    def __init__(self, moe: "FusedMoEConfig") -> None:
        super().__init__(moe)

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
        # Register layer id → sequence index. Used by the env-gated HDUMP
        # path to identify which transformer layer a FusedMoE belongs to
        # (layer.prefix and layer.layer_idx are unset on FusedMoE in this
        # build). Construction is sequential 0..42 so seq == layer index.
        if id(layer) not in self.__class__._ds4_layer_seq:
            seq = len(self.__class__._ds4_layer_seq)
            self.__class__._ds4_layer_seq[id(layer)] = seq
        else:
            seq = self.__class__._ds4_layer_seq[id(layer)]
        layer._ds4_seq = seq
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
        """Run the IQ2_XXS+Q2_K MoE forward with batched per-expert dispatch.

        Strategy: argsort tokens by selected expert id, dispatch one
        batched Triton call per active expert, scatter results back via
        ``index_add_``. One ``.cpu().tolist()`` sync per layer (on the
        n_experts-sized offset table) is the only host round-trip,
        replacing the previous per-token ``int(topk_ids[t,k].item())``
        sync inside an inner loop.

        ``shared_experts_input`` is intentionally ignored — vLLM's
        ``MoeRunner`` handles shared experts independently and adds
        their output to the value we return (see
        ``moe_runner.py:679``).
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

        out = torch.zeros((T, hidden), dtype=x.dtype, device=device)

        # Diagnostic paths below are skipped during torch.compile tracing
        # and during cudagraph stream capture. The capture-guard pattern
        # mirrors fused_batched_moe.py:619-627 in vllm. Reason: file
        # checks, .item() calls, and .cpu() transfers are forbidden inside
        # cudagraph capture; skipping them defensively makes apply()
        # capture-safe even if it ends up inside a captured region.
        _diag_safe = not (
            torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
        )

        # DS4_TRACE: file-toggled diagnostic modes.
        # /logs/ds4_moe_noop exists → return zeros (no contribution)
        # /logs/ds4_moe_passthrough exists → return x (input passes through)
        # Comparing all three modes (normal / noop / passthrough) isolates
        # whether the 2-bit math itself is the problem.
        if _diag_safe:
            import os as _os
            if _os.path.exists("/logs/ds4_moe_noop"):
                if T <= 16:
                    print(
                        f"[DS4_TRACE] apply() NOOP MODE: returning zeros for "
                        f"T={T} hidden={hidden}",
                        flush=True,
                    )
                return out
            if _os.path.exists("/logs/ds4_moe_passthrough"):
                if T <= 16:
                    print(
                        f"[DS4_TRACE] apply() PASSTHROUGH MODE: returning x for "
                        f"T={T} hidden={hidden}",
                        flush=True,
                    )
                return x.clone()

        # _apply_call_count fuels the env-gated DS4_ROUTE / DS4_HDUMP paths
        # below; it's incremented unconditionally so call IDs stay monotonic.
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
            _arm = _diag_safe and _os.path.exists("/logs/ds4_route_arm")
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
        # Quantize all token activations in one shot — already batched.
        x_blocks = x.reshape(T * n_blocks_in, QK_K).to(torch.float32)
        x_q_qs, x_q_d, _x_q_bsums_unused = quantize_q8_K_triton(x_blocks)
        x_q_qs = x_q_qs.reshape(T, n_blocks_in, QK_K)
        x_q_d = x_q_d.reshape(T, n_blocks_in)

        # Build per-expert dispatch table. All GPU until the single
        # offsets.cpu() sync below.
        flat_ids = topk_ids.reshape(-1).to(torch.int64)        # (T*top_k,)
        flat_weights = topk_weights.reshape(-1)                # (T*top_k,)
        sort_order = flat_ids.argsort()
        sorted_eids = flat_ids[sort_order]
        sorted_tids = sort_order // top_k                      # (T*top_k,) token idx per slot
        sorted_w = flat_weights[sort_order]

        counts = torch.bincount(sorted_eids, minlength=n_experts)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        # ONE sync per layer on a (n_experts+1,) tensor (~257 ints).
        # Matches the pattern in vllm/.../cpu_fused_moe.py:440-476.
        offsets_h = offsets.cpu().tolist()

        for e in range(n_experts):
            s = offsets_h[e]
            t_end = offsets_h[e + 1]
            M_e = t_end - s
            if M_e == 0:
                continue

            tokens_e = sorted_tids[s:t_end]                    # (M_e,) GPU slice
            weights_e = sorted_w[s:t_end]                      # (M_e,)

            # Gather this expert's M_e activations (contiguous copy).
            x_q_qs_e = x_q_qs[tokens_e].contiguous()           # (M_e, n_blocks_in, QK_K)
            x_q_d_e = x_q_d[tokens_e].contiguous()             # (M_e, n_blocks_in)

            # Gate / up: w13 is laid out as [n_experts, gate_rows | up_rows, ...]
            gate_qs = layer.w13_iq2xxs_qs[e, :intermediate]
            gate_d = layer.w13_iq2xxs_d[e, :intermediate]
            up_qs = layer.w13_iq2xxs_qs[e, intermediate:]
            up_d = layer.w13_iq2xxs_d[e, intermediate:]

            gate_out, up_out = iq2_xxs_pair_dot_triton(
                gate_qs, gate_d, up_qs, up_d,
                x_q_qs_e, x_q_d_e,
            )                                                   # each (M_e, intermediate)

            # SwiGLU: silu(gate) * up.
            mid = torch.nn.functional.silu(gate_out) * up_out   # (M_e, intermediate)

            # Quantize mid to Q8_K — flatten M dim into the block dim.
            mid_blocks = mid.reshape(M_e * n_blocks_int, QK_K).to(torch.float32)
            mid_qs, mid_d, mid_bsums = quantize_q8_K_triton(mid_blocks)
            mid_qs = mid_qs.reshape(M_e, n_blocks_int, QK_K)
            mid_d = mid_d.reshape(M_e, n_blocks_int)
            mid_bsums = mid_bsums.reshape(M_e, n_blocks_int, 16)

            # Down projection — single expert × M_e tokens.
            down_out = q2_K_accum_dot_triton(
                layer.w2_q2k_scales[e], layer.w2_q2k_qs[e],
                layer.w2_q2k_d[e],      layer.w2_q2k_dmin[e],
                mid_qs, mid_d, mid_bsums,
            )                                                   # (M_e, hidden)

            weighted = down_out.to(x.dtype) * weights_e.unsqueeze(1)
            out.index_add_(0, tokens_e, weighted)

        # DS4_HDUMP: hidden-state dump for layer-by-layer comparison vs ds4
        # reference. Gated by /logs/ds4_dump_arm so it only fires during a
        # prepared compare run. Dumps last-token row of MoE input (x) and
        # MoE output (out) to /logs/ds4_hdump/vllm_{in,out}_L{K}.f32 — fp32
        # raw bytes, 4096 floats per file (16 KB). 43 layers × 2 files = 86
        # files per run, ~1.4 MB total.
        #
        # Layer K extracted from layer.prefix ("model.layers.K.mlp"). T > 1
        # gates to prefill (last position = T-1), since decode (T=1) would
        # overwrite prefill dumps as generation proceeds.
        try:
            import os as _os2
            if _diag_safe and T >= 1 and _os2.path.exists("/logs/ds4_dump_arm"):
                # Resolve layer index (seq registry is the only thing
                # that actually works in our build).
                _seq = getattr(layer, "_ds4_seq", None)
                if _seq is not None:
                    _lidx = int(_seq)
                else:
                    _lidx = None
                # Per-layer call counter: C0 = first armed call (prefill),
                # C1 = second (first decode), C2 = third (second decode), ...
                _cseq = getattr(layer, "_ds4_hdump_cseq", 0)
                layer._ds4_hdump_cseq = _cseq + 1
                if not getattr(layer, "_ds4_hdump_logged", False):
                    layer._ds4_hdump_logged = True
                    _prefix = getattr(layer, "prefix", "") or ""
                    print(
                        f"[DS4_HDUMP_ENTER] _lidx={_lidx} prefix={_prefix!r} "
                        f"T={T} cseq={_cseq}",
                        flush=True,
                    )
                if _lidx is not None:
                    _dir = "/logs/ds4_hdump"
                    _os2.makedirs(_dir, exist_ok=True)
                    # Dump last position. For prefill (T=14) this is the
                    # logit-emitting position. For decode (T=1) there's
                    # only one position.
                    _x_last = x[T - 1, :].detach().to(torch.float32).cpu().contiguous().numpy()
                    _o_last = out[T - 1, :].detach().to(torch.float32).cpu().contiguous().numpy()
                    # Both new (with C{n}) and legacy (without) names —
                    # legacy keeps backward compat with prior compare run.
                    _x_last.tofile(f"{_dir}/vllm_in_L{_lidx}_C{_cseq}.f32")
                    _o_last.tofile(f"{_dir}/vllm_out_L{_lidx}_C{_cseq}.f32")
                    if _cseq == 0:
                        # Also write legacy-named file (no C suffix) so
                        # the existing compare script keeps working for
                        # the prefill case.
                        _x_last.tofile(f"{_dir}/vllm_in_L{_lidx}.f32")
                        _o_last.tofile(f"{_dir}/vllm_out_L{_lidx}.f32")
                    if _lidx == 0 or _lidx == 42:
                        print(
                            f"[DS4_HDUMP] layer={_lidx} T={T} cseq={_cseq} "
                            f"|x|mean={float(abs(_x_last).mean()):.3e} "
                            f"|out|mean={float(abs(_o_last).mean()):.3e}",
                            flush=True,
                        )
        except Exception as _e:
            print(f"[DS4_HDUMP] error: {_e!r}", flush=True)

        return out
