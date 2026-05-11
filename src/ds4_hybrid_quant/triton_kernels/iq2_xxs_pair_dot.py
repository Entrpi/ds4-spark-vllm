"""IQ2_XXS pair dot product kernel.

Computes ``(gate_row[i] dot q8, up_row[i] dot q8)`` for one row index ``i``,
sharing the Q8_K activation reads between the two outputs (the up/gate
projection at this layer always consumes the same activation).

Algorithm summary per (i, block) tile (matches ds4.c:1722, scalar variant):

    For each of 8 sub-blocks of 32 quants:
        Read aux32_0, aux32_1 from gate qs and up qs (8 bytes each)
        For each of 4 (grid_idx, sign_idx) pairs:
            Look up 8 signed grid magnitudes (gate and up, separately)
            Dot with 8 q8_qs values
        scale by ls (4-bit per sub-block scale, encoded in top of aux32_1)
    Outer accum scaled by d_w * d_q8 * 0.125

Lookup tables: a single ``signed_grid`` table of shape ``(256, 128, 8)``
of int8 packs the (grid index x sign index x 8 magnitudes) product, so
the inner loop becomes one indexed load per group.
"""

from __future__ import annotations

import numpy as np

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False
    triton = None
    tl = None

from ..block_layouts import IQ2XXSTensors, Q8KActivation
from ..lookup_tables import IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, QK_K


# ---------------------------------------------------------------------------
# Precomputed signed_grid table (256, 128, 8) int8.
# ---------------------------------------------------------------------------


def build_signed_grid() -> np.ndarray:
    """Construct the (256, 128, 8) int8 signed_grid lookup.

    Each entry ``signed_grid[g, s, j]`` equals
    ``grid_byte(g, j) * (signs_byte(s) & kmask[j] ? -1 : +1)``.
    """
    grid_bytes = IQ2XXS_GRID.view(np.uint8).reshape(256, 8).astype(np.int16)
    signs = KSIGNS_IQ2XS  # (128,) uint8
    bits = (signs[:, None] & KMASK_IQ2XS[None, :]) != 0  # (128, 8) bool
    sign_mul = np.where(bits, -1, 1).astype(np.int16)
    out = grid_bytes[:, None, :] * sign_mul[None, :, :]  # (256, 128, 8)
    return out.astype(np.int8)


SIGNED_GRID = build_signed_grid()  # (256, 128, 8) int8


# ---------------------------------------------------------------------------
# Numpy block-level reference (per-block, matches Triton structure)
# ---------------------------------------------------------------------------


def _decode_one_block(qs_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode one 64-byte IQ2_XXS qs payload into (signed_quants[256], ls[8]).

    ``signed_quants[i]`` is the int16 product ``signed_grid_byte * sign``,
    laid out so that block-position ``i`` corresponds to q8-position ``i``.

    ``ls`` is the per-sub-block scale (8 sub-blocks per block, integer).
    """
    qs_u32 = qs_u8.view(np.uint32).reshape(QK_K // 32, 2)
    aux0 = qs_u32[:, 0]
    aux1 = qs_u32[:, 1]

    grid_idx = np.stack(
        [(aux0 >> (8 * k)) & 0xFF for k in range(4)], axis=-1
    ).astype(np.uint8)  # (8, 4)
    sign_idx = np.stack(
        [(aux1 >> (7 * k)) & 0x7F for k in range(4)], axis=-1
    ).astype(np.uint8)  # (8, 4)
    ls = (2 * (aux1 >> 28) + 1).astype(np.int32)  # (8,)

    # signed_grid[grid_idx, sign_idx] -> (8, 4, 8) int8
    sg = SIGNED_GRID[grid_idx, sign_idx]  # (8, 4, 8)
    signed_quants = sg.reshape(QK_K).astype(np.int16)
    return signed_quants, ls


def vec_dot_iq2_xxs_pair_kernel_numpy(
    iq2_a: IQ2XXSTensors,
    iq2_b: IQ2XXSTensors,
    q8: Q8KActivation,
) -> tuple[float, float]:
    """Numpy emulation of the per-block kernel structure for the pair dot.

    Performs the same work that one Triton program (one (i, block_range)
    tile) does, returning the two scalar outputs. Strict block iteration
    so it parallels the kernel's loop structure.
    """
    nb = iq2_a.d.shape[0]
    assert iq2_b.d.shape[0] == nb and q8.d.shape[0] == nb

    sum_a = 0.0
    sum_b = 0.0
    for i in range(nb):
        sq_a, ls_a = _decode_one_block(iq2_a.qs[i])
        sq_b, ls_b = _decode_one_block(iq2_b.qs[i])
        q8_qs = q8.qs[i].astype(np.int32)

        # Per sub-block: 32 quants.
        sq_a_sub = sq_a.reshape(QK_K // 32, 32).astype(np.int32)
        sq_b_sub = sq_b.reshape(QK_K // 32, 32).astype(np.int32)
        q8_sub = q8_qs.reshape(QK_K // 32, 32)

        bsum_a = ((sq_a_sub * q8_sub).sum(axis=1) * ls_a).sum()
        bsum_b = ((sq_b_sub * q8_sub).sum(axis=1) * ls_b).sum()

        d_q8 = float(q8.d[i])
        sum_a += float(iq2_a.d[i]) * d_q8 * float(bsum_a)
        sum_b += float(iq2_b.d[i]) * d_q8 * float(bsum_b)

    return 0.125 * sum_a, 0.125 * sum_b


# ---------------------------------------------------------------------------
# Triton kernel (Spark-only)
# ---------------------------------------------------------------------------


if HAVE_TRITON:

    @triton.jit
    def _iq2_xxs_pair_dot_kernel(
        # Weight (n_rows, n_blocks, *) -- shared across all M tokens that
        # routed to this expert.
        w_a_qs_ptr,    # *uint8,   (n_rows, n_blocks, 64)
        w_a_d_ptr,     # *float16, (n_rows, n_blocks)
        w_b_qs_ptr,    # *uint8
        w_b_d_ptr,     # *float16
        # Activation (M, n_blocks, *) -- M tokens' Q8 blocks.
        q8_qs_ptr,     # *int8,    (M, n_blocks, 256)
        q8_d_ptr,      # *float32, (M, n_blocks)
        # Output (M, n_rows) for each of A and B.
        out_a_ptr,     # *float32, (M, n_rows)
        out_b_ptr,     # *float32, (M, n_rows)
        # Lookup
        signed_grid_ptr,  # *int8, (256, 128, 8)
        # Sizes / strides (runtime — M and n_rows vary per call).
        n_blocks,
        stride_q8_qs_m,    # = n_blocks * 256
        stride_q8_d_m,     # = n_blocks
        stride_out_m,      # = n_rows
        BLOCK: tl.constexpr,         # = 256 (QK_K)
        N_SUB: tl.constexpr,         # = 8
        SUB_SIZE: tl.constexpr,      # = 32
    ):
        """One program computes the two outputs for one (token, row).

        Grid is ``(M, n_rows)``. All M programs sharing a row_id read
        identical weight bytes — automatic L2 reuse on the weight tensor.
        Q8 activation is per-token (offset by ``m_id * stride_q8_*_m``).
        """
        m_id = tl.program_id(0)
        row_id = tl.program_id(1)

        # Per-row weight pointers (shared across M).
        w_a_qs_row = w_a_qs_ptr + row_id * n_blocks * 64
        w_a_d_row = w_a_d_ptr + row_id * n_blocks
        w_b_qs_row = w_b_qs_ptr + row_id * n_blocks * 64
        w_b_d_row = w_b_d_ptr + row_id * n_blocks

        # Per-token activation pointers.
        q8_qs_tok = q8_qs_ptr + m_id * stride_q8_qs_m
        q8_d_tok = q8_d_ptr + m_id * stride_q8_d_m

        sum_a = 0.0
        sum_b = 0.0

        # All-block iteration. (For a vectorized version we'd tile over
        # blocks; this scalar-block formulation is intentionally simple
        # for the first cut.)
        for blk in range(0, n_blocks):
            # --- Load aux32 pairs for both rows (8 sub-blocks * 2 uint32s) ---
            # qs is uint8[64]; reinterpret as uint32[16] for bit extraction.
            qs_offs = tl.arange(0, 16)
            a_aux = tl.load(
                w_a_qs_row + blk * 64 + qs_offs * 4,
                # cast 4 bytes -> uint32 by view (Triton has no direct view
                # primitive on uint8 -> uint32, so we manually pack):
            ).to(tl.uint32)
            b_aux = tl.load(
                w_b_qs_row + blk * 64 + qs_offs * 4,
            ).to(tl.uint32)

            # NOTE: tl.load above reads uint8 values, not uint32. We need
            # to read uint32 directly. To do that, reinterpret the qs
            # buffer with stride-4 byte offsets and read 4 bytes at a
            # time, packing manually:
            #   aux32[k] = qs[4k] | qs[4k+1]<<8 | qs[4k+2]<<16 | qs[4k+3]<<24
            # We do this below in a tighter form.

            # Per sub-block (8 of them): aux32_0, aux32_1
            for sub in tl.static_range(0, N_SUB):
                base = blk * 64 + sub * 8
                # Pack four uint8 -> uint32 (little-endian) for both halves.
                a_b0 = tl.load(w_a_qs_row + base + 0).to(tl.uint32)
                a_b1 = tl.load(w_a_qs_row + base + 1).to(tl.uint32)
                a_b2 = tl.load(w_a_qs_row + base + 2).to(tl.uint32)
                a_b3 = tl.load(w_a_qs_row + base + 3).to(tl.uint32)
                a_b4 = tl.load(w_a_qs_row + base + 4).to(tl.uint32)
                a_b5 = tl.load(w_a_qs_row + base + 5).to(tl.uint32)
                a_b6 = tl.load(w_a_qs_row + base + 6).to(tl.uint32)
                a_b7 = tl.load(w_a_qs_row + base + 7).to(tl.uint32)
                a_aux0 = a_b0 | (a_b1 << 8) | (a_b2 << 16) | (a_b3 << 24)
                a_aux1 = a_b4 | (a_b5 << 8) | (a_b6 << 16) | (a_b7 << 24)

                b_b0 = tl.load(w_b_qs_row + base + 0).to(tl.uint32)
                b_b1 = tl.load(w_b_qs_row + base + 1).to(tl.uint32)
                b_b2 = tl.load(w_b_qs_row + base + 2).to(tl.uint32)
                b_b3 = tl.load(w_b_qs_row + base + 3).to(tl.uint32)
                b_b4 = tl.load(w_b_qs_row + base + 4).to(tl.uint32)
                b_b5 = tl.load(w_b_qs_row + base + 5).to(tl.uint32)
                b_b6 = tl.load(w_b_qs_row + base + 6).to(tl.uint32)
                b_b7 = tl.load(w_b_qs_row + base + 7).to(tl.uint32)
                b_aux0 = b_b0 | (b_b1 << 8) | (b_b2 << 16) | (b_b3 << 24)
                b_aux1 = b_b4 | (b_b5 << 8) | (b_b6 << 16) | (b_b7 << 24)

                ls_a = (2 * (a_aux1 >> 28) + 1).to(tl.int32)
                ls_b = (2 * (b_aux1 >> 28) + 1).to(tl.int32)

                sub_sum_a = tl.zeros((), dtype=tl.int32)
                sub_sum_b = tl.zeros((), dtype=tl.int32)
                # 4 quads per sub-block, 8 quants per quad, 32 quants total.
                # Use static_range so loop unrolls into Triton IR.
                # Each quad: lookup 8 signed_grid values, dot with q8 chunk.
                q8_base = blk * BLOCK + sub * SUB_SIZE
                for quad in tl.static_range(0, 4):
                    grid_idx_a = (a_aux0 >> (8 * quad)) & 0xFF
                    grid_idx_b = (b_aux0 >> (8 * quad)) & 0xFF
                    sign_idx_a = (a_aux1 >> (7 * quad)) & 0x7F
                    sign_idx_b = (b_aux1 >> (7 * quad)) & 0x7F

                    # signed_grid[grid_idx, sign_idx] is 8 int8 values.
                    base_a = (grid_idx_a * 128 + sign_idx_a) * 8
                    base_b = (grid_idx_b * 128 + sign_idx_b) * 8
                    j = tl.arange(0, 8)
                    sg_a = tl.load(signed_grid_ptr + base_a + j).to(tl.int32)
                    sg_b = tl.load(signed_grid_ptr + base_b + j).to(tl.int32)

                    q8_chunk = tl.load(
                        q8_qs_tok + q8_base + quad * 8 + j
                    ).to(tl.int32)

                    sub_sum_a += tl.sum(sg_a * q8_chunk, axis=0)
                    sub_sum_b += tl.sum(sg_b * q8_chunk, axis=0)

                # Block-level: bsum += sub_sum * ls
                # Promote to float32 for mixed accumulation.
                d_q8 = tl.load(q8_d_tok + blk).to(tl.float32)
                d_a = tl.load(w_a_d_row + blk).to(tl.float32)
                d_b = tl.load(w_b_d_row + blk).to(tl.float32)

                # Scaled per-sub-block contribution: ls * sub_sum.
                # We accumulate in float to avoid 32-bit overflow in long
                # rows; ds4's bsum stays in int32 because n_blocks is small
                # (one row of one expert), but float matches the final
                # numerical result up to FMA ordering.
                sum_a += d_a * d_q8 * (ls_a * sub_sum_a).to(tl.float32)
                sum_b += d_b * d_q8 * (ls_b * sub_sum_b).to(tl.float32)

        tl.store(out_a_ptr + m_id * stride_out_m + row_id, 0.125 * sum_a)
        tl.store(out_b_ptr + m_id * stride_out_m + row_id, 0.125 * sum_b)


    def iq2_xxs_pair_dot_triton(
        w_a_qs, w_a_d,    # uint8 (n_rows, n_blocks, 64), float16 (n_rows, n_blocks)
        w_b_qs, w_b_d,
        q8_qs, q8_d,      # int8 (M, n_blocks, 256), float32 (M, n_blocks)
    ):
        """Run the batched pair-dot Triton kernel.

        ``q8_qs`` / ``q8_d`` carry an outer M (token-batch) dim. Outputs
        have shape ``(M, n_rows)``. M=1 is allowed (caller passes
        unsqueezed tensors).
        """
        import torch

        n_rows = w_a_qs.shape[0]
        n_blocks = w_a_qs.shape[1]
        if q8_qs.ndim != 3 or q8_d.ndim != 2:
            raise ValueError(
                f"q8_qs must be (M, n_blocks, 256) and q8_d (M, n_blocks); "
                f"got q8_qs.shape={tuple(q8_qs.shape)} q8_d.shape={tuple(q8_d.shape)}"
            )
        M = q8_qs.shape[0]
        if q8_d.shape[0] != M:
            raise ValueError("q8_qs and q8_d must agree on M")

        out_a = torch.empty((M, n_rows), dtype=torch.float32, device=w_a_qs.device)
        out_b = torch.empty((M, n_rows), dtype=torch.float32, device=w_a_qs.device)

        signed_grid_t = torch.from_numpy(SIGNED_GRID).to(w_a_qs.device).contiguous()

        q8_qs_c = q8_qs.contiguous()
        q8_d_c = q8_d.contiguous()
        stride_q8_qs_m = n_blocks * QK_K
        stride_q8_d_m = n_blocks
        stride_out_m = n_rows

        grid = (M, n_rows)
        _iq2_xxs_pair_dot_kernel[grid](
            w_a_qs.contiguous(), w_a_d.contiguous(),
            w_b_qs.contiguous(), w_b_d.contiguous(),
            q8_qs_c, q8_d_c,
            out_a, out_b,
            signed_grid_t,
            n_blocks,
            stride_q8_qs_m, stride_q8_d_m, stride_out_m,
            BLOCK=QK_K, N_SUB=8, SUB_SIZE=32,
        )
        return out_a, out_b


else:

    def iq2_xxs_pair_dot_triton(*args, **kwargs):
        raise RuntimeError(
            "Triton not installed; iq2_xxs_pair_dot_triton requires CUDA. "
            "Use vec_dot_iq2_xxs_pair_kernel_numpy for CPU validation."
        )
