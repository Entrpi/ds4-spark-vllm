"""vLLM-side dispatch glue for the ds4 hybrid IQ2_XXS+Q2_K+FP8 recipe.

Registered as a vLLM quantization method named ``deepseek_v4_hybrid_iq2``.
On checkpoint load, dispatches:

- ``FusedMoE`` layers   -> :class:`Iq2XxsQ2KFusedMoEMethod` (our Triton kernels)
- Other linears         -> the embedded :class:`Fp8Config` (block-FP8 path)

Activation reuse: dense / shared-expert / attention layers go through vLLM's
existing FP8 LinearMethod, which already runs on SM121 via the eugr Spark
container. Only the routed-MoE path is novel.

Installation: drop these files into vLLM's
``model_executor/layers/quantization/`` directory and add
``"deepseek_v4_hybrid_iq2": Ds4HybridIq2Config`` to the
``QUANTIZATION_METHODS`` registry. The eugr mod's ``run.sh`` does that.
"""

from .config import Ds4HybridIq2Config
from .moe_method import Iq2XxsQ2KFusedMoEMethod

__all__ = ["Ds4HybridIq2Config", "Iq2XxsQ2KFusedMoEMethod"]
