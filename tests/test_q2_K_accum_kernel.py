"""Tests for the Q2_K accumulated dot kernel."""

from __future__ import annotations

import numpy as np
import pytest

from ds4_hybrid_quant.block_layouts import QK_K, Q2KTensors
from ds4_hybrid_quant.reference import quantize_q8_K, vec_dot_q2_K_q8_K
from ds4_hybrid_quant.test_helpers import make_q2_K_blocks
from ds4_hybrid_quant.triton_kernels.q2_K_accum_dot import (
    HAVE_TRITON,
    vec_dot_q2_K_accum_kernel_numpy,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xACC)


@pytest.mark.parametrize("n_experts", [1, 3])
@pytest.mark.parametrize("n_blocks", [1, 4])
def test_kernel_numpy_matches_reference_sum(
    n_experts: int, n_blocks: int, rng: np.random.Generator,
) -> None:
    """Sum of vec_dot across experts should match the kernel's accumulation."""
    weights = []
    activations = []
    for _ in range(n_experts):
        weights.append(make_q2_K_blocks(n_blocks=n_blocks, rng=rng))
        x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
        activations.append(quantize_q8_K(x))

    expected = sum(
        vec_dot_q2_K_q8_K(w, q8) for w, q8 in zip(weights, activations)
    )
    got = vec_dot_q2_K_accum_kernel_numpy(weights, activations)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAVE_TRITON, reason="triton not installed (Mac)")
def test_triton_accum_matches_numpy(rng: np.random.Generator) -> None:
    import torch
    from ds4_hybrid_quant.triton_kernels.q2_K_accum_dot import q2_K_accum_dot_triton

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n_experts, n_rows, n_blocks = 2, 4, 2
    rng_np = np.random.default_rng(0xAA)

    # Build (n_experts, n_rows) Q2_K rows + (n_experts,) Q8 activations.
    rows: list[list[Q2KTensors]] = []
    acts = []
    for e in range(n_experts):
        rows_e = [make_q2_K_blocks(n_blocks=n_blocks, rng=rng_np) for _ in range(n_rows)]
        rows.append(rows_e)
        x = rng_np.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
        acts.append(quantize_q8_K(x))

    # Pack into 4D tensors (e, r, blk, ...) for the kernel.
    w_scales = torch.from_numpy(np.stack([
        np.stack([r.scales for r in row]) for row in rows
    ])).cuda()
    w_qs = torch.from_numpy(np.stack([
        np.stack([r.qs for r in row]) for row in rows
    ])).cuda()
    w_d = torch.from_numpy(np.stack([
        np.stack([r.d for r in row]) for row in rows
    ])).cuda()
    w_dmin = torch.from_numpy(np.stack([
        np.stack([r.dmin for r in row]) for row in rows
    ])).cuda()
    q8_qs = torch.from_numpy(np.stack([a.qs for a in acts])).cuda()
    q8_d = torch.from_numpy(np.stack([a.d for a in acts])).cuda()
    q8_bsums = torch.from_numpy(np.stack([a.bsums for a in acts])).cuda()

    out = q2_K_accum_dot_triton(w_scales, w_qs, w_d, w_dmin, q8_qs, q8_d, q8_bsums)

    for r in range(n_rows):
        expected = vec_dot_q2_K_accum_kernel_numpy(
            [rows[e][r] for e in range(n_experts)],
            acts,
        )
        np.testing.assert_allclose(out[r].item(), expected, rtol=1e-5, atol=1e-5)
