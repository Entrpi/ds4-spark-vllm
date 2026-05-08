"""Explicit dequantization helpers.

These produce the same float values that a fully-dequantized weight would
hold. They are used as a slow-but-obvious validation oracle for the dot
products in :mod:`reference`, and by the converter when sanity-checking
that re-packaged tensors round-trip correctly.
"""

from __future__ import annotations

import numpy as np

from .block_layouts import IQ2XXSTensors, Q2KTensors
from .lookup_tables import IQ2XXS_GRID, KSIGNS_IQ2XS, QK_K


def dequantize_iq2_xxs(iq2: IQ2XXSTensors) -> np.ndarray:
    """Dequantize one or more IQ2_XXS blocks back to float32.

    Returns shape ``(n_blocks * QK_K,)``.
    """
    nb = iq2.d.shape[0]
    if iq2.qs.shape != (nb, 64):
        raise ValueError(f"unexpected IQ2_XXS qs shape {iq2.qs.shape}")

    qs_u32 = iq2.qs.view(np.uint32).reshape(nb, QK_K // 32, 2)
    aux32_0 = qs_u32[..., 0]
    aux32_1 = qs_u32[..., 1]

    grid_idx = np.stack(
        [(aux32_0 >> (8 * k)) & 0xFF for k in range(4)], axis=-1
    ).astype(np.uint8)  # (nb, 8, 4)
    sign_idx = np.stack(
        [(aux32_1 >> (7 * k)) & 0x7F for k in range(4)], axis=-1
    ).astype(np.uint8)  # (nb, 8, 4)
    ls = (2 * (aux32_1 >> 28) + 1).astype(np.float32)  # (nb, 8)

    grid_bytes = IQ2XXS_GRID.view(np.uint8).reshape(256, 8)
    mags = grid_bytes[grid_idx].astype(np.float32)  # (nb, 8, 4, 8)

    sign_bytes = KSIGNS_IQ2XS[sign_idx]  # (nb, 8, 4)
    bits = (sign_bytes[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    signs = np.where(bits == 1, -1.0, 1.0).astype(np.float32)  # (nb, 8, 4, 8)

    signed_mags = mags * signs  # (nb, 8, 4, 8)
    sub = signed_mags.reshape(nb, QK_K // 32, 32)

    block_d = iq2.d.astype(np.float32).reshape(nb, 1, 1)
    scale = (ls / 8.0).reshape(nb, QK_K // 32, 1)
    out = block_d * scale * sub  # (nb, 8, 32)
    return out.reshape(nb * QK_K)


def dequantize_q2_K(q2k: Q2KTensors) -> np.ndarray:
    """Dequantize one or more Q2_K blocks back to float32.

    Returns shape ``(n_blocks * QK_K,)``. Iteration order mirrors the dot
    product so the per-16-quant scale index alignment is identical.
    """
    nb = q2k.d.shape[0]
    out = np.empty(nb * QK_K, dtype=np.float32)

    for i in range(nb):
        d = float(q2k.d[i])
        dmin = float(q2k.dmin[i])
        scales = q2k.scales[i]
        scale_lo = (scales & 0x0F).astype(np.float32)
        scale_hi = (scales >> 4).astype(np.float32)
        qs = q2k.qs[i]
        block_out = out[i * QK_K:(i + 1) * QK_K]

        is_idx = 0
        for k in range(QK_K // 128):
            base_q2 = k * 32
            base_out = k * 128
            for j in range(4):
                shift = 2 * j
                lo16 = ((qs[base_q2:base_q2 + 16] >> shift) & 0x03).astype(np.float32)
                hi16 = ((qs[base_q2 + 16:base_q2 + 32] >> shift) & 0x03).astype(np.float32)
                block_out[base_out + j * 32:base_out + j * 32 + 16] = (
                    d * scale_lo[is_idx] * lo16 - dmin * scale_hi[is_idx]
                )
                is_idx += 1
                block_out[base_out + j * 32 + 16:base_out + j * 32 + 32] = (
                    d * scale_lo[is_idx] * hi16 - dmin * scale_hi[is_idx]
                )
                is_idx += 1

    return out
