"""vLLM-side dispatch glue for the ds4 hybrid IQ2_XXS+Q2_K+FP8 recipe.

Registered as a vLLM quantization method named ``deepseek_v4_hybrid_iq2``.
On checkpoint load, dispatches:

- ``FusedMoE`` layers   -> :class:`Iq2XxsQ2KFusedMoEMethod` (our Triton kernels)
- Other linears         -> the embedded :class:`Fp8Config` (block-FP8 path)

Activation reuse: dense / shared-expert / attention layers go through vLLM's
existing FP8 LinearMethod, which already runs on SM121 via the eugr Spark
container. Only the routed-MoE path is novel.

Installation: ``pip install`` the package. We register a vLLM general plugin
via the ``vllm.general_plugins`` entry point so vLLM auto-discovers and
imports our config (which uses :func:`register_quantization_config` to add
the method to the registry).
"""


def register_plugin() -> None:
    """vLLM general-plugin entry point.

    Importing :mod:`.config` triggers the ``@register_quantization_config``
    decorator and adds ``deepseek_v4_hybrid_iq2`` to vLLM's customized
    quantization map.
    """
    from . import config  # noqa: F401  (import for side effect)


__all__ = ["register_plugin"]
