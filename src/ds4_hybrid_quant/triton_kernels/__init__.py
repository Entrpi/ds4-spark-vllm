"""Triton kernels for the ds4 2-bit recipe.

Three kernels, all targeting NVIDIA Blackwell SM121 (DGX Spark):

- :mod:`q8_K_quantize`: float -> Q8_K activation quantization
- :mod:`iq2_xxs_pair_dot`: fused gate+up matmul (IQ2_XXS x Q8_K)
- :mod:`q2_K_accum_dot`: down matmul accumulated across experts (Q2_K x Q8_K)

Each module exposes both a Triton implementation (Spark-only) and a numpy
"block-level reference" that performs the same per-block computation. Tests
on the Mac use the numpy reference; tests on the Spark validate the actual
Triton kernel against both the reference and ds4's C output.
"""
