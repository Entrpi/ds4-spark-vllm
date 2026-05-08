"""Test helpers: build random valid IQ2_XXS / Q2_K block bytes from fields."""

from __future__ import annotations

import numpy as np

from .block_layouts import IQ2XXSTensors, Q2KTensors
from .lookup_tables import QK_K


def make_iq2_xxs_blocks(
    *,
    n_blocks: int,
    rng: np.random.Generator,
    d_scale: float = 1.0,
) -> IQ2XXSTensors:
    """Build ``n_blocks`` of synthetic but format-valid IQ2_XXS data.

    Picks random (uniform) grid indices, sign indices, and per-sub-block
    scales, packs them into the canonical 64-byte qs payload, and returns
    a single-row IQ2XXSTensors.
    """
    n_sub = QK_K // 32  # 8 sub-blocks per block

    grid_idx = rng.integers(0, 256, size=(n_blocks, n_sub, 4), dtype=np.uint32)
    sign_idx = rng.integers(0, 128, size=(n_blocks, n_sub, 4), dtype=np.uint32)
    scale_bits = rng.integers(0, 16, size=(n_blocks, n_sub), dtype=np.uint32)

    aux32_0 = (
        (grid_idx[..., 0] << 0)
        | (grid_idx[..., 1] << 8)
        | (grid_idx[..., 2] << 16)
        | (grid_idx[..., 3] << 24)
    )
    aux32_1 = (
        (sign_idx[..., 0] << 0)
        | (sign_idx[..., 1] << 7)
        | (sign_idx[..., 2] << 14)
        | (sign_idx[..., 3] << 21)
        | (scale_bits << 28)
    )

    pair = np.stack([aux32_0, aux32_1], axis=-1).astype(np.uint32)  # (nb, 8, 2)
    qs = pair.reshape(n_blocks, -1).view(np.uint8).reshape(n_blocks, 64)

    d = np.full(n_blocks, d_scale, dtype=np.float16)
    return IQ2XXSTensors(d=d.copy(), qs=qs.copy())


def make_q2_K_blocks(
    *,
    n_blocks: int,
    rng: np.random.Generator,
    d: float = 0.05,
    dmin: float = 0.02,
) -> Q2KTensors:
    """Build ``n_blocks`` of synthetic format-valid Q2_K data.

    Uses random nibble scales/mins and random 2-bit packed q values.
    """
    scales = rng.integers(0, 256, size=(n_blocks, 16), dtype=np.uint8)
    qs = rng.integers(0, 256, size=(n_blocks, 64), dtype=np.uint8)
    return Q2KTensors(
        d=np.full(n_blocks, d, dtype=np.float16),
        dmin=np.full(n_blocks, dmin, dtype=np.float16),
        scales=scales,
        qs=qs,
    )
