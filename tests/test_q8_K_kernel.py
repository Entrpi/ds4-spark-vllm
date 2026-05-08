"""Tests for the Q8_K quantization kernel.

The numpy reference must match the high-level reference in
``ds4_hybrid_quant.reference``, which itself is bit-exact against ds4's
C scalar fallback. The Triton path is exercised on CUDA only and is
skipped when triton/torch.cuda is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds4_hybrid_quant.block_layouts import QK_K
from ds4_hybrid_quant.reference import quantize_q8_K
from ds4_hybrid_quant.triton_kernels.q8_K_quantize import (
    HAVE_TRITON,
    quantize_q8_K_numpy,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xCAFE)


@pytest.mark.parametrize("n_blocks", [1, 4, 17])
def test_numpy_kernel_matches_reference(n_blocks: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal((n_blocks, QK_K)).astype(np.float32) * 1.5
    qs_k, d_k, bsums_k = quantize_q8_K_numpy(x)

    ref = quantize_q8_K(x.reshape(-1))

    np.testing.assert_array_equal(qs_k, ref.qs)
    np.testing.assert_array_equal(bsums_k, ref.bsums)
    np.testing.assert_array_equal(d_k, ref.d)


def test_zero_block_handling() -> None:
    x = np.zeros((1, QK_K), dtype=np.float32)
    qs, d, bsums = quantize_q8_K_numpy(x)
    assert (qs == 0).all()
    assert d[0] == 0.0
    assert (bsums == 0).all()


def test_uniform_block(rng: np.random.Generator) -> None:
    """Block where all values equal the same magnitude — exercises tie-break."""
    x = np.full((1, QK_K), 3.0, dtype=np.float32)
    qs, d, bsums = quantize_q8_K_numpy(x)
    # max_signed = 3.0, iscale = -127/3.0 ≈ -42.33, qs[j] ≈ -127, d ≈ -3/127.
    assert (qs == -127).all()
    np.testing.assert_allclose(d[0], -3.0 / 127.0 * (-1.0) * -1.0, rtol=1e-6)
    # bsums: each sub-block of 16 = -127*16 = -2032
    assert (bsums == -2032).all()


@pytest.mark.skipif(not HAVE_TRITON, reason="triton not installed (Mac)")
def test_triton_kernel_matches_numpy(rng: np.random.Generator) -> None:
    import torch
    from ds4_hybrid_quant.triton_kernels.q8_K_quantize import quantize_q8_K_triton

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n_blocks = 16
    x_np = rng.standard_normal((n_blocks, QK_K)).astype(np.float32) * 1.2
    x_t = torch.from_numpy(x_np).cuda()

    qs_t, d_t, bsums_t = quantize_q8_K_triton(x_t)
    qs_n, d_n, bsums_n = quantize_q8_K_numpy(x_np)

    np.testing.assert_array_equal(qs_t.cpu().numpy(), qs_n)
    np.testing.assert_array_equal(bsums_t.cpu().numpy(), bsums_n)
    np.testing.assert_array_equal(d_t.cpu().numpy(), d_n)
