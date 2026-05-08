"""Tests for the GGUF -> safetensors block repackers in ``builder``.

Validates that ``repack_iq2_xxs_rows`` / ``repack_q2_K_rows`` can round-trip
synthetic GGUF-layout bytes through to our 6-tensor format and back to the
same dot-product output as the reference path.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ds4_hybrid_quant.block_layouts import IQ2XXSTensors, Q2KTensors, QK_K
from ds4_hybrid_quant.builder import (
    repack_iq2_xxs_rows,
    repack_q2_K_rows,
)
from ds4_hybrid_quant.reference import (
    quantize_q8_K,
    vec_dot_iq2_xxs_q8_K,
    vec_dot_q2_K_q8_K,
)
from ds4_hybrid_quant.test_helpers import make_iq2_xxs_blocks, make_q2_K_blocks


def _gguf_iq2_xxs_bytes(d_fp16: np.ndarray, qs_u8: np.ndarray) -> bytes:
    """Layout one block: u16 d + u8 qs[64] = 66 bytes."""
    buf = bytearray()
    for b in range(d_fp16.shape[0]):
        buf.extend(d_fp16[b:b + 1].view(np.uint16).tobytes())
        buf.extend(qs_u8[b].tobytes())
    return bytes(buf)


def _gguf_q2_K_bytes(
    d_fp16: np.ndarray, dmin_fp16: np.ndarray,
    scales_u8: np.ndarray, qs_u8: np.ndarray,
) -> bytes:
    """Layout one block: u8 scales[16] + u8 qs[64] + u16 d + u16 dmin = 84."""
    buf = bytearray()
    for b in range(d_fp16.shape[0]):
        buf.extend(scales_u8[b].tobytes())
        buf.extend(qs_u8[b].tobytes())
        buf.extend(d_fp16[b:b + 1].view(np.uint16).tobytes())
        buf.extend(dmin_fp16[b:b + 1].view(np.uint16).tobytes())
    return bytes(buf)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xB14D)


def test_iq2_xxs_repack_roundtrip(rng: np.random.Generator) -> None:
    """Build GGUF bytes for 3 rows of 2 blocks, repack, verify dot equality."""
    n_rows, n_blocks = 3, 2
    in_dim = n_blocks * QK_K

    rows = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.012)
            for _ in range(n_rows)]
    raw = bytearray()
    for r in rows:
        raw.extend(_gguf_iq2_xxs_bytes(r.d, r.qs))

    qs, d = repack_iq2_xxs_rows(bytes(raw), n_rows=n_rows, in_dim=in_dim)
    assert qs.shape == (n_rows, n_blocks, 64)
    assert d.shape == (n_rows, n_blocks)

    # Pick a quantized activation and verify dot product equality for each row.
    x = rng.standard_normal(in_dim).astype(np.float32) * 0.5
    q8 = quantize_q8_K(x)

    for r_idx, original in enumerate(rows):
        repacked = IQ2XXSTensors(d=d[r_idx], qs=qs[r_idx])
        a = vec_dot_iq2_xxs_q8_K(original, q8)
        b = vec_dot_iq2_xxs_q8_K(repacked, q8)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_q2_K_repack_roundtrip(rng: np.random.Generator) -> None:
    n_rows, n_blocks = 3, 2
    in_dim = n_blocks * QK_K

    rows = [make_q2_K_blocks(n_blocks=n_blocks, rng=rng) for _ in range(n_rows)]
    raw = bytearray()
    for r in rows:
        raw.extend(_gguf_q2_K_bytes(r.d, r.dmin, r.scales, r.qs))

    qs, scales, d, dmin = repack_q2_K_rows(bytes(raw), n_rows=n_rows, in_dim=in_dim)
    assert qs.shape == (n_rows, n_blocks, 64)
    assert scales.shape == (n_rows, n_blocks, 16)
    assert d.shape == dmin.shape == (n_rows, n_blocks)

    x = rng.standard_normal(in_dim).astype(np.float32) * 0.5
    q8 = quantize_q8_K(x)

    for r_idx, original in enumerate(rows):
        repacked = Q2KTensors(d=d[r_idx], dmin=dmin[r_idx],
                              scales=scales[r_idx], qs=qs[r_idx])
        a = vec_dot_q2_K_q8_K(original, q8)
        b = vec_dot_q2_K_q8_K(repacked, q8)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
