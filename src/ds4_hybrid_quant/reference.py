"""Pure-Python reference implementations of the ds4 quant kernels.

These are the validation oracle for the Triton kernels: they faithfully
reproduce the scalar fallback paths in antirez/ds4's ds4.c, including the
exact integer accumulation orders and final scaling factors. Numpy is used
for vectorization where it doesn't change semantics.

Functions
---------
quantize_q8_K(x)
    Float -> Q8_K block packing. Matches ds4_quantize_row_q8_K (ds4.c:1473).
vec_dot_iq2_xxs_q8_K(iq2, q8)
    IQ2_XXS x Q8_K dot. Matches scalar branch of
    ds4_vec_dot_iq2_xxs_q8_K (ds4.c:1689).
vec_dot_q2_K_q8_K(q2k, q8)
    Q2_K x Q8_K dot. Matches scalar branch of
    ds4_vec_dot_q2_K_q8_K (ds4.c:1593).
"""

from __future__ import annotations

import numpy as np

from .block_layouts import (
    IQ2XXSTensors,
    Q2KTensors,
    Q8KActivation,
)
from .lookup_tables import IQ2XXS_GRID, KSIGNS_IQ2XS, QK_K


# ---------------------------------------------------------------------------
# Q8_K quantization (activation -> int8 with per-block fp32 scale)
# ---------------------------------------------------------------------------


def quantize_q8_K(x: np.ndarray) -> Q8KActivation:
    """Quantize a 1-D float vector to Q8_K blocks.

    Mirrors ds4_quantize_row_q8_K (ds4.c:1473). Picks the value with the
    largest absolute magnitude as the per-block scale anchor, computes a
    signed int8 quant via round-to-nearest with ties-to-even, and emits
    16 sub-block sums (bsums) used by the Q2_K dot fast path.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size % QK_K != 0:
        raise ValueError(f"Q8_K input size {x.size} is not a multiple of {QK_K}")

    nb = x.size // QK_K
    blocks = x.reshape(nb, QK_K)

    abs_blocks = np.abs(blocks)
    idx = abs_blocks.argmax(axis=1)
    amax = abs_blocks[np.arange(nb), idx]
    max_signed = blocks[np.arange(nb), idx]  # signed value at the abs-max position

    qs = np.zeros((nb, QK_K), dtype=np.int8)
    d = np.zeros(nb, dtype=np.float32)

    nonzero = amax > 0.0
    if nonzero.any():
        # iscale = -127 / max  (note: max is signed; this is the ds4 convention)
        iscale = (-127.0 / max_signed[nonzero]).astype(np.float32)
        v = blocks[nonzero] * iscale[:, None]
        # lrintf: round-to-nearest, ties-to-even (banker's). numpy's rint
        # implements the same semantics.
        v = np.rint(v)
        v = np.clip(v, -128.0, 127.0).astype(np.int8)
        qs[nonzero] = v
        d[nonzero] = (1.0 / iscale).astype(np.float32)
    # zero-amax blocks already have qs=0, d=0

    # bsums: per-16-quant sums (16 sub-block sums per block of 256 quants)
    bsums = qs.reshape(nb, QK_K // 16, 16).sum(axis=2).astype(np.int16)

    return Q8KActivation(d=d, qs=qs, bsums=bsums)


# ---------------------------------------------------------------------------
# IQ2_XXS dot product
# ---------------------------------------------------------------------------


def _grid_byte_view(grid_idx: np.ndarray) -> np.ndarray:
    """Look up IQ2_XXS grid magnitudes as 8 unsigned bytes per index.

    Returns shape ``(*grid_idx.shape, 8)`` of uint8.
    """
    # IQ2XXS_GRID is uint64; reinterpret as bytes (little-endian) so byte 0
    # is the lowest-order quant magnitude in the packed pattern.
    grid_bytes = IQ2XXS_GRID.view(np.uint8).reshape(256, 8)
    return grid_bytes[grid_idx]


def _signs_byte(sign_idx: np.ndarray) -> np.ndarray:
    """Look up the 8-bit sign byte for a 7-bit sign index."""
    return KSIGNS_IQ2XS[sign_idx]


def _expand_sign_byte(sign_byte: np.ndarray) -> np.ndarray:
    """Expand each 8-bit sign byte to 8 signs in {+1, -1} (int8).

    Returns shape ``(*sign_byte.shape, 8)`` of int8.
    """
    bits = (sign_byte[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    # bit==1 -> negative, bit==0 -> positive
    return np.where(bits == 1, np.int8(-1), np.int8(1))


def vec_dot_iq2_xxs_q8_K(iq2: IQ2XXSTensors, q8: Q8KActivation) -> float:
    """IQ2_XXS x Q8_K dot product over one row of weights and one activation.

    Both inputs cover the same number of 256-quant blocks. Mirrors the scalar
    branch of ds4_vec_dot_iq2_xxs_q8_K (ds4.c:1689), including the trailing
    ``0.125`` scale factor.

    Args:
        iq2: IQ2_XXS tensor with ``d.shape == (n_blocks,)`` and
             ``qs.shape == (n_blocks, 64)``.
        q8:  Q8_K activation with matching block count.
    """
    nb = iq2.d.shape[0]
    if q8.d.shape[0] != nb or q8.qs.shape[0] != nb:
        raise ValueError("IQ2_XXS and Q8_K block counts disagree")

    # Per-block scale (fp16 d * fp32 d -> fp32, matching the C path's order)
    block_d = iq2.d.astype(np.float32) * q8.d  # (nb,)

    # Reinterpret 64 uint8 bytes as 16 little-endian uint32 (= 32 ushorts).
    # Each pair of uint32 (8 bytes) describes one 32-quant sub-block:
    #   aux32[0]: 4 grid index bytes
    #   aux32[1]: 4 packed 7-bit sign indices + 4-bit per-subblock scale
    qs_u32 = iq2.qs.view(np.uint32).reshape(nb, QK_K // 32, 2)  # (nb, 8, 2)

    aux32_0 = qs_u32[..., 0]  # (nb, 8) -- the 4 grid index bytes
    aux32_1 = qs_u32[..., 1]  # (nb, 8) -- packed signs + scale

    # 4 grid indices per sub-block (low byte first).
    grid_idx = np.stack(
        [
            (aux32_0 >> 0) & 0xFF,
            (aux32_0 >> 8) & 0xFF,
            (aux32_0 >> 16) & 0xFF,
            (aux32_0 >> 24) & 0xFF,
        ],
        axis=-1,
    ).astype(np.uint8)  # (nb, 8, 4)

    # 4 sign indices per sub-block (7 bits each, packed in the low 28 bits).
    sign_idx = np.stack(
        [
            (aux32_1 >> 0) & 0x7F,
            (aux32_1 >> 7) & 0x7F,
            (aux32_1 >> 14) & 0x7F,
            (aux32_1 >> 21) & 0x7F,
        ],
        axis=-1,
    ).astype(np.uint8)  # (nb, 8, 4)

    # ls = 2 * (top 4 bits) + 1
    ls = (2 * (aux32_1 >> 28) + 1).astype(np.int32)  # (nb, 8)

    # Resolve grid magnitudes -> 8 signed int8 values per (sub-block, group).
    mags = _grid_byte_view(grid_idx)  # (nb, 8, 4, 8) uint8
    sign_bytes = _signs_byte(sign_idx)  # (nb, 8, 4) uint8
    signs = _expand_sign_byte(sign_bytes)  # (nb, 8, 4, 8) int8 in {-1,+1}
    signed_q2 = mags.astype(np.int16) * signs.astype(np.int16)  # (nb, 8, 4, 8) int16

    # The 32 quants per sub-block come from concatenating (4 groups, 8 vals each).
    signed_q2 = signed_q2.reshape(nb, QK_K // 32, 32).astype(np.int32)

    # Q8_K activation reshaped to match.
    q8_blk = q8.qs.reshape(nb, QK_K // 32, 32).astype(np.int32)

    # Per-sub-block dot, then weight by ls.
    sub_dots = (signed_q2 * q8_blk).sum(axis=-1)  # (nb, 8) int32
    bsum = (sub_dots * ls).sum(axis=-1)  # (nb,) int32

    sumf = float((block_d * bsum.astype(np.float32)).sum())
    return 0.125 * sumf


def vec_dot_iq2_xxs_pair_q8_K(
    iq2_a: IQ2XXSTensors,
    iq2_b: IQ2XXSTensors,
    q8: Q8KActivation,
) -> tuple[float, float]:
    """Fused gate+up dot: two IQ2_XXS rows share one Q8_K activation.

    Equivalent to calling ``vec_dot_iq2_xxs_q8_K`` twice but with shared
    activation reads — matches ds4_vec_dot_iq2_xxs_pair_q8_K (ds4.c:1722).
    """
    return (
        vec_dot_iq2_xxs_q8_K(iq2_a, q8),
        vec_dot_iq2_xxs_q8_K(iq2_b, q8),
    )


# ---------------------------------------------------------------------------
# Q2_K dot product
# ---------------------------------------------------------------------------


def vec_dot_q2_K_q8_K(q2k: Q2KTensors, q8: Q8KActivation) -> float:
    """Q2_K x Q8_K dot product. Mirrors ds4_vec_dot_q2_K_q8_K scalar branch.

    Args:
        q2k: Q2_K tensor with ``d``, ``dmin``, ``scales[16]``, ``qs[64]``
             per block.
        q8:  Q8_K activation with matching block count.
    """
    nb = q2k.d.shape[0]
    if q8.d.shape[0] != nb:
        raise ValueError("Q2_K and Q8_K block counts disagree")

    sumf = 0.0
    for i in range(nb):
        scales = q2k.scales[i]  # uint8[16]
        scale_lo = (scales & 0x0F).astype(np.int32)  # per-32-quant scale
        scale_hi = (scales >> 4).astype(np.int32)    # per-32-quant min weight

        # summs = sum_j bsums[j] * (scales[j] >> 4)
        summs = int((q8.bsums[i].astype(np.int32) * scale_hi).sum())

        dall = float(q2k.d[i].astype(np.float32) * q8.d[i])
        dmin = float(q2k.dmin[i].astype(np.float32) * q8.d[i])

        # Q2_K qs are 64 bytes packing 256 2-bit quants. The decode reads
        # them in groups of 32 bytes, applying shifts {0, 2, 4, 6} to
        # extract 4 disjoint 16-quant slices per group; each slice is
        # dotted against 16 q8 bytes and multiplied by its lower-nibble
        # scale.
        qs = q2k.qs[i]  # uint8[64]
        q8_qs = q8.qs[i]  # int8[256]

        isum = 0
        is_idx = 0
        for k in range(QK_K // 128):  # 2 outer iterations
            base_q2 = k * 32
            base_q8 = k * 128
            for j in range(4):
                shift = 2 * j
                # dot(q2[base..base+16] >> shift & 3, q8[base_q8+j*32..+16])
                lo16 = ((qs[base_q2:base_q2 + 16] >> shift) & 0x03).astype(np.int32)
                hi16 = ((qs[base_q2 + 16:base_q2 + 32] >> shift) & 0x03).astype(np.int32)
                q8_lo = q8_qs[base_q8 + j * 32:base_q8 + j * 32 + 16].astype(np.int32)
                q8_hi = q8_qs[base_q8 + j * 32 + 16:base_q8 + j * 32 + 32].astype(np.int32)

                isum += int(scale_lo[is_idx]) * int((lo16 * q8_lo).sum())
                is_idx += 1
                isum += int(scale_lo[is_idx]) * int((hi16 * q8_hi).sum())
                is_idx += 1

        sumf += dall * float(isum) - dmin * float(summs)

    return float(sumf)
