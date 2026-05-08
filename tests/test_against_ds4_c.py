"""Cross-validate the Python reference against ds4's actual C dot products.

Builds the C harness on first run and pipes raw block bytes through it.
This guards against hidden semantic mismatches in field ordering, byte
order, sign-bit interpretation, and accumulation that pure self-consistency
tests cannot catch.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ds4_hybrid_quant.block_layouts import IQ2XXSTensors, Q2KTensors, QK_K
from ds4_hybrid_quant.reference import (
    quantize_q8_K,
    vec_dot_iq2_xxs_q8_K,
    vec_dot_q2_K_q8_K,
)
from ds4_hybrid_quant.test_helpers import make_iq2_xxs_blocks, make_q2_K_blocks


HARNESS_DIR = Path(__file__).parent / "c_harness"
HARNESS_BIN = HARNESS_DIR / "ds4_dot_harness"


@pytest.fixture(scope="module")
def harness() -> Path:
    if not HARNESS_BIN.exists():
        subprocess.check_call(["make", "-C", str(HARNESS_DIR)])
    return HARNESS_BIN


# ---------- C struct serializers (must match layouts in ds4_dot.c) ----------


def pack_block_iq2_xxs(d_fp16: np.ndarray, qs_u8: np.ndarray) -> bytes:
    """One block: u16 d + u16 qs[32]. ds4_dot.c struct block_iq2_xxs."""
    buf = bytearray()
    buf.extend(d_fp16.view(np.uint16).tobytes())
    buf.extend(qs_u8.tobytes())  # 64 bytes
    return bytes(buf)


def pack_block_q2_K(d_fp16: np.ndarray, dmin_fp16: np.ndarray,
                    scales_u8: np.ndarray, qs_u8: np.ndarray) -> bytes:
    """One block: u8 scales[16] + u8 qs[64] + u16 d + u16 dmin."""
    buf = bytearray()
    buf.extend(scales_u8.tobytes())
    buf.extend(qs_u8.tobytes())
    buf.extend(d_fp16.view(np.uint16).tobytes())
    buf.extend(dmin_fp16.view(np.uint16).tobytes())
    return bytes(buf)


def pack_block_q8_K(d_fp32: float, qs_i8: np.ndarray, bsums_i16: np.ndarray) -> bytes:
    """One block: f32 d + i8 qs[256] + i16 bsums[16]."""
    buf = bytearray()
    buf.extend(struct.pack("<f", float(d_fp32)))
    buf.extend(qs_i8.tobytes())
    buf.extend(bsums_i16.tobytes())
    return bytes(buf)


def serialize_q8_K(q8) -> bytes:
    """Serialize a Q8KActivation across n_blocks."""
    nb = q8.d.shape[0]
    buf = bytearray()
    for b in range(nb):
        buf.extend(pack_block_q8_K(q8.d[b], q8.qs[b], q8.bsums[b]))
    return bytes(buf)


def call_harness_q8_K_quantize(harness: Path, x: np.ndarray) -> bytes:
    n_blocks = x.size // QK_K
    payload = b"q" + struct.pack("<I", n_blocks) + x.astype("<f4").tobytes()
    p = subprocess.run([str(harness)], input=payload, capture_output=True, check=True)
    return p.stdout


def call_harness_dot_iq2_xxs(harness: Path, iq2: IQ2XXSTensors, q8) -> float:
    n_blocks = iq2.d.shape[0]
    iq2_bytes = bytearray()
    for b in range(n_blocks):
        iq2_bytes.extend(pack_block_iq2_xxs(iq2.d[b:b + 1], iq2.qs[b]))
    payload = b"i" + struct.pack("<I", n_blocks) + bytes(iq2_bytes) + serialize_q8_K(q8)
    p = subprocess.run([str(harness)], input=payload, capture_output=True, check=True)
    return struct.unpack("<f", p.stdout)[0]


def call_harness_dot_q2_K(harness: Path, q2k: Q2KTensors, q8) -> float:
    n_blocks = q2k.d.shape[0]
    q2_bytes = bytearray()
    for b in range(n_blocks):
        q2_bytes.extend(pack_block_q2_K(
            q2k.d[b:b + 1], q2k.dmin[b:b + 1], q2k.scales[b], q2k.qs[b]
        ))
    payload = b"2" + struct.pack("<I", n_blocks) + bytes(q2_bytes) + serialize_q8_K(q8)
    p = subprocess.run([str(harness)], input=payload, capture_output=True, check=True)
    return struct.unpack("<f", p.stdout)[0]


# ----------------------------- tests -----------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xDC4)


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_q8_K_quantize_matches_c(harness: Path, n_blocks: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 1.5
    py = quantize_q8_K(x)
    c_bytes = call_harness_q8_K_quantize(harness, x)

    # Parse C output blocks
    block_sz = 4 + 256 + 32
    assert len(c_bytes) == n_blocks * block_sz
    for b in range(n_blocks):
        off = b * block_sz
        c_d = struct.unpack("<f", c_bytes[off:off + 4])[0]
        c_qs = np.frombuffer(c_bytes, dtype=np.int8, count=256, offset=off + 4)
        c_bsums = np.frombuffer(c_bytes, dtype=np.int16, count=16, offset=off + 4 + 256)

        np.testing.assert_array_equal(c_qs, py.qs[b])
        np.testing.assert_array_equal(c_bsums, py.bsums[b])
        # d is float32 in C; Python d is also float32. Should be bit-exact.
        assert c_d == py.d[b], f"block {b}: C d={c_d} Py d={py.d[b]}"


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_iq2_xxs_dot_matches_c(harness: Path, n_blocks: int, rng: np.random.Generator) -> None:
    iq2 = make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.013)
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
    q8 = quantize_q8_K(x)

    py = vec_dot_iq2_xxs_q8_K(iq2, q8)
    c = call_harness_dot_iq2_xxs(harness, iq2, q8)

    np.testing.assert_allclose(py, c, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_q2_K_dot_matches_c(harness: Path, n_blocks: int, rng: np.random.Generator) -> None:
    q2k = make_q2_K_blocks(n_blocks=n_blocks, rng=rng)
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.6
    q8 = quantize_q8_K(x)

    py = vec_dot_q2_K_q8_K(q2k, q8)
    c = call_harness_dot_q2_K(harness, q2k, q8)

    np.testing.assert_allclose(py, c, rtol=1e-5, atol=1e-5)
