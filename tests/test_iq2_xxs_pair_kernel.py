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
    """M=1 case: kernel result matches the per-row numpy reference."""
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
    # M=1: add leading batch dim.
    q8_qs_t = torch.from_numpy(q8.qs).unsqueeze(0).cuda()  # (1, n_blocks, 256)
    q8_d_t = torch.from_numpy(q8.d).unsqueeze(0).cuda()    # (1, n_blocks)

    out_a, out_b = iq2_xxs_pair_dot_triton(
        w_a_qs, w_a_d, w_b_qs, w_b_d, q8_qs_t, q8_d_t,
    )
    assert out_a.shape == (1, n_rows)
    assert out_b.shape == (1, n_rows)

    for i in range(n_rows):
        ra, rb = vec_dot_iq2_xxs_pair_kernel_numpy(rows_a[i], rows_b[i], q8)
        np.testing.assert_allclose(out_a[0, i].item(), ra, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out_b[0, i].item(), rb, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAVE_TRITON, reason="triton not installed (Mac)")
@pytest.mark.parametrize("M", [1, 2, 5, 17])
def test_triton_pair_batched_matches_per_token(M: int, rng: np.random.Generator) -> None:
    """Batched M-token kernel result matches calling the M=1 kernel M times."""
    import torch
    from ds4_hybrid_quant.triton_kernels.iq2_xxs_pair_dot import iq2_xxs_pair_dot_triton

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n_rows, n_blocks = 6, 3
    rows_a = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.012) for _ in range(n_rows)]
    rows_b = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.013) for _ in range(n_rows)]

    # Build M distinct activations.
    q8s = []
    for _ in range(M):
        x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
        q8s.append(quantize_q8_K(x))

    w_a_qs = torch.from_numpy(np.stack([r.qs for r in rows_a])).cuda()
    w_a_d = torch.from_numpy(np.stack([r.d for r in rows_a])).cuda()
    w_b_qs = torch.from_numpy(np.stack([r.qs for r in rows_b])).cuda()
    w_b_d = torch.from_numpy(np.stack([r.d for r in rows_b])).cuda()

    # Stack q8 across M.
    q8_qs_batched = torch.from_numpy(np.stack([q.qs for q in q8s])).cuda()  # (M, n_blocks, 256)
    q8_d_batched = torch.from_numpy(np.stack([q.d for q in q8s])).cuda()    # (M, n_blocks)

    out_a, out_b = iq2_xxs_pair_dot_triton(
        w_a_qs, w_a_d, w_b_qs, w_b_d, q8_qs_batched, q8_d_batched,
    )
    assert out_a.shape == (M, n_rows)
    assert out_b.shape == (M, n_rows)

    # Compare each (m, row) against numpy reference for that token.
    for m in range(M):
        for i in range(n_rows):
            ra, rb = vec_dot_iq2_xxs_pair_kernel_numpy(rows_a[i], rows_b[i], q8s[m])
            np.testing.assert_allclose(out_a[m, i].item(), ra, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(out_b[m, i].item(), rb, rtol=1e-5, atol=1e-5)

    # Also bit-exact vs the M=1 path: calling per-token with unsqueeze(0)
    # should give numerically identical results to the batched call.
    for m in range(M):
        single_qs = q8_qs_batched[m].unsqueeze(0).contiguous()
        single_d = q8_d_batched[m].unsqueeze(0).contiguous()
        oa1, ob1 = iq2_xxs_pair_dot_triton(
            w_a_qs, w_a_d, w_b_qs, w_b_d, single_qs, single_d,
        )
        torch.testing.assert_close(oa1[0], out_a[m], rtol=0.0, atol=0.0)
        torch.testing.assert_close(ob1[0], out_b[m], rtol=0.0, atol=0.0)
