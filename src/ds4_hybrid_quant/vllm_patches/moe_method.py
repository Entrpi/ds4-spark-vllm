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
        def _passthrough_loader(param, loaded_weight):
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

        out = torch.zeros((T, hidden), dtype=x.dtype, device=device)

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
        return out
