"""Tests for the IQ2_XXS pair dot kernel."""

from __future__ import annotations

import numpy as np
import pytest

from ds4_hybrid_quant.block_layouts import QK_K
from ds4_hybrid_quant.reference import (
    quantize_q8_K,
    vec_dot_iq2_xxs_pair_q8_K,
)
from ds4_hybrid_quant.test_helpers import make_iq2_xxs_blocks
from ds4_hybrid_quant.triton_kernels.iq2_xxs_pair_dot import (
    HAVE_TRITON,
    SIGNED_GRID,
    build_signed_grid,
    vec_dot_iq2_xxs_pair_kernel_numpy,
)


def test_signed_grid_construction() -> None:
    """signed_grid is a deterministic build; verify shape and a few entries."""
    sg = build_signed_grid()
    assert sg.shape == (256, 128, 8)
    assert sg.dtype == np.int8
    # Sign index 0 = ksigns_iq2xs[0] = 0, so all signs are positive.
    # Grid 0 = 0x0808080808080808 -> all bytes 0x08.
    assert (sg[0, 0] == 8).all()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xBEEF)


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_kernel_numpy_matches_reference(n_blocks: int, rng: np.random.Generator) -> None:
    a = make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.011)
    b = make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.014)
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.5
    q8 = quantize_q8_K(x)

    ka, kb = vec_dot_iq2_xxs_pair_kernel_numpy(a, b, q8)
    ra, rb = vec_dot_iq2_xxs_pair_q8_K(a, b, q8)

    np.testing.assert_allclose(ka, ra, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(kb, rb, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAVE_TRITON, reason="triton not installed (Mac)")
def test_triton_pair_matches_numpy_kernel(rng: np.random.Generator) -> None:
    import torch
    from ds4_hybrid_quant.triton_kernels.iq2_xxs_pair_dot import iq2_xxs_pair_dot_triton

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n_rows, n_blocks = 4, 2
    rows_a = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.01) for _ in range(n_rows)]
    rows_b = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.01) for _ in range(n_rows)]
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
    q8 = quantize_q8_K(x)

    w_a_qs = torch.from_numpy(np.stack([r.qs for r in rows_a])).cuda()
    w_a_d = torch.from_numpy(np.stack([r.d for r in rows_a])).cuda()
    w_b_qs = torch.from_numpy(np.stack([r.qs for r in rows_b])).cuda()
    w_b_d = torch.from_numpy(np.stack([r.d for r in rows_b])).cuda()
    q8_qs_t = torch.from_numpy(q8.qs).cuda()
    q8_d_t = torch.from_numpy(q8.d).cuda()

    out_a, out_b = iq2_xxs_pair_dot_triton(
        w_a_qs, w_a_d, w_b_qs, w_b_d, q8_qs_t, q8_d_t,
    )

    for i in range(n_rows):
        ra, rb = vec_dot_iq2_xxs_pair_kernel_numpy(rows_a[i], rows_b[i], q8)
        np.testing.assert_allclose(out_a[i].item(), ra, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out_b[i].item(), rb, rtol=1e-5, atol=1e-5)
