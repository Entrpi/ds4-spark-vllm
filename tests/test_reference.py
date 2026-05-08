"""Self-consistency tests for the pure-Python reference implementation.

Strategy: construct random format-valid blocks, dequantize each weight to
explicit floats, compute a naive float dot against a quantized Q8_K
activation that is *also* dequantized to floats, and compare against the
integer-arithmetic ``vec_dot_*`` path. Within FP rounding the two must agree.

This validates that ``vec_dot_*`` is internally consistent with
``dequantize_*`` and ``quantize_q8_K``. Cross-validation against the actual
ds4 C output happens in a separate harness.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds4_hybrid_quant.block_layouts import QK_K
from ds4_hybrid_quant.dequant import dequantize_iq2_xxs, dequantize_q2_K
from ds4_hybrid_quant.reference import (
    quantize_q8_K,
    vec_dot_iq2_xxs_pair_q8_K,
    vec_dot_iq2_xxs_q8_K,
    vec_dot_q2_K_q8_K,
)
from ds4_hybrid_quant.test_helpers import make_iq2_xxs_blocks, make_q2_K_blocks


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xD54)


@pytest.mark.parametrize("n_blocks", [1, 4])
def test_q8_K_quant_dequant_roundtrip(n_blocks: int, rng: np.random.Generator) -> None:
    """Q8_K(x) then * d should approximately recover x with bounded error."""
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 3.0
    q8 = quantize_q8_K(x)

    # ds4 quantize: iscale = -127 / max_signed, qs = round(iscale * x),
    # d = 1 / iscale = max_signed / -127. To recover: x ≈ qs * d.
    recon = np.empty_like(x)
    for b in range(n_blocks):
        recon[b * QK_K:(b + 1) * QK_K] = q8.qs[b].astype(np.float32) * q8.d[b]

    err = np.abs(recon - x).max()
    # Per-block max absolute error is bounded by half the quant step.
    # Quant step = |d|/127, so half-step = |d|/254. With Gaussian inputs and
    # multiple blocks we expect well under 1% relative error in practice.
    rel = err / (np.abs(x).max() + 1e-9)
    assert rel < 0.05, f"Q8_K roundtrip max relative error {rel:.4f} too large"


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_iq2_xxs_dot_matches_dequant_dot(
    n_blocks: int, rng: np.random.Generator
) -> None:
    """vec_dot_iq2_xxs_q8_K must equal sum(dequant_iq2 * dequant_q8) up to FP."""
    iq2 = make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.01)
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.5
    q8 = quantize_q8_K(x)

    # Slow oracle path: explicit float dequant + float dot.
    deq_iq2 = dequantize_iq2_xxs(iq2)
    # Q8_K dequant: per-block, qs_block * (d_block / -127)
    deq_q8 = np.empty_like(x)
    for b in range(n_blocks):
        deq_q8[b * QK_K:(b + 1) * QK_K] = q8.qs[b].astype(np.float32) * q8.d[b]
    oracle = float((deq_iq2 * deq_q8).sum())

    fast = vec_dot_iq2_xxs_q8_K(iq2, q8)
    # ``vec_dot`` uses a different convention for d (positive scale), so
    # results should match up to sign of the activation scale. ds4's
    # ``ds4_quantize_row_q8_K`` stores d such that the dot path computes
    # sum(qs * mag) * (d_iq2 * d_q8 / 8) directly. Verify that recovery.
    expected_via_dot = oracle * -127.0  # cancels the q8 scale convention diff
    # Actually the simplest: oracle should equal `fast` up to a global sign
    # because d_q8 is signed in our convention.
    np.testing.assert_allclose(fast, oracle, rtol=1e-3, atol=1e-3)
    _ = expected_via_dot  # keep flake happy if convention shifts


@pytest.mark.parametrize("n_blocks", [1, 4])
def test_iq2_xxs_pair_matches_two_singletons(
    n_blocks: int, rng: np.random.Generator
) -> None:
    iq2_a = make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.012)
    iq2_b = make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.009)
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
    q8 = quantize_q8_K(x)

    a, b = vec_dot_iq2_xxs_pair_q8_K(iq2_a, iq2_b, q8)
    a_solo = vec_dot_iq2_xxs_q8_K(iq2_a, q8)
    b_solo = vec_dot_iq2_xxs_q8_K(iq2_b, q8)

    np.testing.assert_allclose(a, a_solo, rtol=0, atol=0)
    np.testing.assert_allclose(b, b_solo, rtol=0, atol=0)


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_q2_K_dot_matches_dequant_dot(
    n_blocks: int, rng: np.random.Generator
) -> None:
    q2k = make_q2_K_blocks(n_blocks=n_blocks, rng=rng)
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.7
    q8 = quantize_q8_K(x)

    deq_q2 = dequantize_q2_K(q2k)
    deq_q8 = np.empty_like(x)
    for b in range(n_blocks):
        deq_q8[b * QK_K:(b + 1) * QK_K] = q8.qs[b].astype(np.float32) * q8.d[b]
    oracle = float((deq_q2 * deq_q8).sum())

    fast = vec_dot_q2_K_q8_K(q2k, q8)
    np.testing.assert_allclose(fast, oracle, rtol=1e-3, atol=1e-3)
