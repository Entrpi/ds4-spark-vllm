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

        # Mark all of them as MoE expert weights so vLLM's loader routes
        # them to the per-expert weight loader path.
        for name in (
            "w13_iq2xxs_qs", "w13_iq2xxs_d",
            "w2_q2k_qs", "w2_q2k_scales", "w2_q2k_d", "w2_q2k_dmin",
        ):
            set_weight_attrs(getattr(layer, name), extra_weight_attrs)

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

        # Quantize all token activations in one shot.
        x_blocks = x.reshape(T * n_blocks_in, QK_K).to(torch.float32)
        x_q_qs, x_q_d, x_q_bsums = quantize_q8_K_triton(x_blocks)
        x_q_qs = x_q_qs.reshape(T, n_blocks_in, QK_K)
        x_q_d = x_q_d.reshape(T, n_blocks_in)
        x_q_bsums = x_q_bsums.reshape(T, n_blocks_in, 16)

        for t in range(T):
            for k in range(top_k):
                expert = int(topk_ids[t, k].item())
                weight = topk_weights[t, k]

                # Gate / up: w13 is laid out as [n_experts, gate_rows | up_rows, ...]
                # Slice the two halves.
                gate_qs = layer.w13_iq2xxs_qs[expert, :intermediate]
                gate_d = layer.w13_iq2xxs_d[expert, :intermediate]
                up_qs = layer.w13_iq2xxs_qs[expert, intermediate:]
                up_d = layer.w13_iq2xxs_d[expert, intermediate:]

                gate_out, up_out = iq2_xxs_pair_dot_triton(
                    gate_qs, gate_d, up_qs, up_d,
                    x_q_qs[t], x_q_d[t],
                )
                # gate_out: (intermediate,), up_out: (intermediate,)

                # SwiGLU: silu(gate) * up. (No clamp by default; the model's
                # ``swiglu_limit`` is exposed through the layer's pre-kernel
                # path elsewhere — for the first cut we ignore it.)
                mid = torch.nn.functional.silu(gate_out) * up_out

                # Quantize mid to Q8_K.
                mid_blocks = mid.reshape(n_blocks_int, QK_K).to(torch.float32)
                mid_qs, mid_d, mid_bsums = quantize_q8_K_triton(mid_blocks)

                # Down projection. We re-use the q2_K_accum kernel even
                # though we have a single expert here — pass n_experts=1
                # arrays.
                w_scales = layer.w2_q2k_scales[expert].unsqueeze(0)
                w_qs = layer.w2_q2k_qs[expert].unsqueeze(0)
                w_d = layer.w2_q2k_d[expert].unsqueeze(0)
                w_dmin = layer.w2_q2k_dmin[expert].unsqueeze(0)

                down_out = q2_K_accum_dot_triton(
                    w_scales, w_qs, w_d, w_dmin,
                    mid_qs.unsqueeze(0), mid_d.unsqueeze(0),
                    mid_bsums.unsqueeze(0),
                )
                out[t].add_(down_out.to(x.dtype) * weight)

        return out
