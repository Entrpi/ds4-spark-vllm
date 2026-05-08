"""DeepseekV4 hybrid IQ2_XXS + Q2_K + FP8 quantization config for vLLM.

Mirrors the rmstxrx/vllm hybrid-fp8 pattern in shape: a top-level
QuantizationConfig that holds an embedded Fp8Config for dense layers and
dispatches FusedMoE layers to a custom MoE method.

This file imports vLLM modules and is intended to live under
``vllm/model_executor/layers/quantization/`` once installed by the eugr mod.
On Mac/Linux dev (no vLLM), the imports are conditional so the module can
still be parsed for static analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    HAVE_VLLM = True
except ImportError:  # pragma: no cover - dev path without vLLM installed
    HAVE_VLLM = False
    QuantizationConfig = object  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: F811


_QUANT_METHOD_NAME = "deepseek_v4_hybrid_iq2"


class Ds4HybridIq2Config(QuantizationConfig):
    """Hybrid 2-bit-experts + FP8-dense quantization for DeepSeek V4 Flash.

    Recipe (lifted from antirez/ds4):
        - routed gate/up : IQ2_XXS  (~2.06 bpw)
        - routed down    : Q2_K     (~2.62 bpw)
        - dense / attn   : FP8 E4M3 block-128 (UE8M0 scales from sgl-project)
        - everything else: source dtype (BF16 typically)

    The config is auto-detected from a checkpoint whose ``config.json`` has
    ``quantization_config.quant_method == "deepseek_v4_hybrid_iq2"``. Users
    can also opt in explicitly via ``--quantization deepseek_v4_hybrid_iq2``.
    """

    def __init__(
        self,
        weight_block_size: list[int] | None = None,
        scale_fmt: str = "ue8m0",
        activation_scheme: str = "dynamic",
    ) -> None:
        super().__init__()
        if not HAVE_VLLM:
            raise RuntimeError("vLLM is not installed")

        self.weight_block_size = weight_block_size or [128, 128]
        self.scale_fmt = scale_fmt
        self.activation_scheme = activation_scheme

        # The dense layers go through this embedded FP8 config. Matches the
        # sgl-project DeepSeek-V4-Flash-FP8 layout (UE8M0 scales, block-128
        # weights, dynamic activations).
        self._fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=self.activation_scheme,
            weight_block_size=self.weight_block_size,
        )

    # ------------------------------------------------------------------
    # vLLM QuantizationConfig contract
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return _QUANT_METHOD_NAME

    @classmethod
    def get_supported_act_dtypes(cls) -> list["torch.dtype"]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Targets Blackwell SM121 (DGX Spark). Older SMs lack the FP8 block
        # GEMM kernels and Triton perf for our paths.
        return 90  # H100+; SM121 is included

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # config lives in the model's config.json

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Ds4HybridIq2Config":
        return cls(
            weight_block_size=config.get("weight_block_size", [128, 128]),
            scale_fmt=config.get("scale_fmt", "ue8m0"),
            activation_scheme=config.get("activation_scheme", "dynamic"),
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any] | None, user_quant: str | None,
        hf_config=None,
    ) -> str | None:
        """Auto-select this method when the checkpoint says so."""
        if isinstance(hf_quant_cfg, dict) and \
                hf_quant_cfg.get("quant_method") == _QUANT_METHOD_NAME:
            return _QUANT_METHOD_NAME
        if user_quant == _QUANT_METHOD_NAME:
            return _QUANT_METHOD_NAME
        return None

    def get_quant_method(
        self, layer: "torch.nn.Module", prefix: str,
    ) -> "QuantizeMethodBase | None":
        """Per-layer dispatch: FusedMoE -> our method, others -> FP8 config."""
        # Routed-MoE layers get the IQ2_XXS+Q2_K kernels.
        if isinstance(layer, FusedMoE):
            # Local import to avoid circular dep when vLLM tooling imports
            # this module for the first time.
            from .moe_method import Iq2XxsQ2KFusedMoEMethod
            return Iq2XxsQ2KFusedMoEMethod(layer.moe_config)

        # Everything else (attention linears, shared experts, embeddings)
        # routes through the embedded FP8 config. That config also returns
        # UnquantizedLinearMethod for layers without FP8 weights (e.g.
        # norms), which is the right behavior.
        if isinstance(layer, LinearBase):
            return self._fp8_config.get_quant_method(layer, prefix)

        # Fallback: defer to FP8 (handles ParallelLMHead etc.).
        return self._fp8_config.get_quant_method(layer, prefix)
