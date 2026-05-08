"""Q2_K accumulated dot product kernel.

Computes one output row of the routed-expert down projection:

    out[r] = sum over selected experts e of
                Q2_K_row[e, r] dot Q8_K_activation[e]

Each expert provides its own quantized Q2_K weight row (the routed-expert
``w2`` parameter for that expert) and its own Q8_K-quantized post-SwiGLU
midq activation. The output row has no per-expert weighting beyond what
the SwiGLU step already applied — the experts simply add.

Per-block algorithm (matches ds4_vec_dot_q2_K_q8_K scalar, ds4.c:1593):

    summs = sum(q8_bsums * (scales >> 4))
    isum  = 0
    for k in 0..1:                           # two halves of the block
        for j in 0..3:                       # four shifts {0,2,4,6}
            isum += scale_lo[is++] * dot(q2_lo16, q8_lo16)
            isum += scale_lo[is++] * dot(q2_hi16, q8_hi16)
    block_contrib = (q8.d * d_q2k * isum) - (q8.d * dmin_q2k * summs)

Sum block_contrib over all blocks of the row, then over all experts.
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

from ..block_layouts import Q2KTensors, Q8KActivation
from ..lookup_tables import QK_K


# ---------------------------------------------------------------------------
# Numpy block-level reference
# ---------------------------------------------------------------------------


def _q2k_block_dot(
    scales: np.ndarray,  # (16,) uint8
    qs: np.ndarray,      # (64,) uint8
    d_block: float,
    dmin_block: float,
    q8_qs: np.ndarray,   # (256,) int8
    q8_bsums: np.ndarray,  # (16,) int16
    q8_d: float,
) -> float:
    """One-block Q2_K x Q8_K dot, exactly matching the scalar fallback."""
    scale_lo = (scales & 0x0F).astype(np.int32)
    scale_hi = (scales >> 4).astype(np.int32)
    summs = int((q8_bsums.astype(np.int32) * scale_hi).sum())

    isum = 0
    is_idx = 0
    for k in range(QK_K // 128):  # 2
        base_q2 = k * 32
        base_q8 = k * 128
        for j in range(4):
            shift = 2 * j
            lo16 = ((qs[base_q2:base_q2 + 16] >> shift) & 0x03).astype(np.int32)
            hi16 = ((qs[base_q2 + 16:base_q2 + 32] >> shift) & 0x03).astype(np.int32)
            q8_lo = q8_qs[base_q8 + j * 32:base_q8 + j * 32 + 16].astype(np.int32)
            q8_hi = q8_qs[base_q8 + j * 32 + 16:base_q8 + j * 32 + 32].astype(np.int32)

            isum += int(scale_lo[is_idx]) * int((lo16 * q8_lo).sum())
            is_idx += 1
            isum += int(scale_lo[is_idx]) * int((hi16 * q8_hi).sum())
            is_idx += 1

    dall = q8_d * d_block
    dmin = q8_d * dmin_block
    return dall * float(isum) - dmin * float(summs)


def vec_dot_q2_K_accum_kernel_numpy(
    weights: list[Q2KTensors],
    activations: list[Q8KActivation],
) -> float:
    """One output row's accumulated dot across experts.

    Each ``weights[e]`` is a single-row Q2_K tensor (n_blocks blocks);
    each ``activations[e]`` is the matching Q8_K activation for that
    expert.

    Returns the float scalar ``sum_e Q2K[e] . Q8K[e]``.
    """
    if len(weights) != len(activations):
        raise ValueError("weights and activations must align across experts")

    total = 0.0
    for w, q8 in zip(weights, activations):
        nb = w.d.shape[0]
        if q8.d.shape[0] != nb:
            raise ValueError("block counts disagree within an expert")
        for i in range(nb):
            total += _q2k_block_dot(
                w.scales[i], w.qs[i],
                float(w.d[i]), float(w.dmin[i]),
                q8.qs[i], q8.bsums[i], float(q8.d[i]),
            )
    return total


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


if HAVE_TRITON:

    @triton.jit
    def _q2_K_accum_dot_kernel(
        # Weight (n_experts, n_rows, n_blocks, ...)  -- packed per expert
        w_scales_ptr,   # *uint8,   (n_experts, n_rows, n_blocks, 16)
        w_qs_ptr,       # *uint8,   (n_experts, n_rows, n_blocks, 64)
        w_d_ptr,        # *float16, (n_experts, n_rows, n_blocks)
        w_dmin_ptr,     # *float16, (n_experts, n_rows, n_blocks)
        # Activation per expert
        q8_qs_ptr,      # *int8,    (n_experts, n_blocks, 256)
        q8_d_ptr,       # *float32, (n_experts, n_blocks)
        q8_bsums_ptr,   # *int16,   (n_experts, n_blocks, 16)
        # Output (n_rows,)
        out_ptr,        # *float32, (n_rows,)
        n_experts,
        n_rows,
        n_blocks,
        BLOCK: tl.constexpr,    # = 256
        N_SCALES: tl.constexpr,  # = 16
    ):
        """One program produces one output row by summing across experts."""
        row_id = tl.program_id(0)

        acc = tl.zeros((), dtype=tl.float32)

        for e in range(0, n_experts):
            # Per-(expert, row) base addresses.
            w_base_blk = e * n_rows * n_blocks
            w_scales_row = w_scales_ptr + (w_base_blk + row_id * n_blocks) * 16
            w_qs_row = w_qs_ptr + (w_base_blk + row_id * n_blocks) * 64
            w_d_row = w_d_ptr + w_base_blk + row_id * n_blocks
            w_dmin_row = w_dmin_ptr + w_base_blk + row_id * n_blocks
            q8_blk = e * n_blocks

            for blk in range(0, n_blocks):
                # Load scales[16].
                scale_offs = tl.arange(0, N_SCALES)
                scales = tl.load(w_scales_row + blk * 16 + scale_offs)
                scale_lo = (scales & 0x0F).to(tl.int32)
                scale_hi = (scales >> 4).to(tl.int32)

                # Load q8 bsums[16] and reduce summs.
                bsums = tl.load(
                    q8_bsums_ptr + (q8_blk + blk) * 16 + scale_offs
                ).to(tl.int32)
                summs = tl.sum(bsums * scale_hi, axis=0)

                # Per-block isum: 2 outer * 4 shifts * 2 halves = 16 dot pieces.
                # Each piece is a 16-element int dot.
                isum = tl.zeros((), dtype=tl.int32)
                is_idx = 0
                j16 = tl.arange(0, 16)
                for k in tl.static_range(0, 2):  # QK_K/128
                    base_q2 = blk * 64 + k * 32
                    base_q8 = (q8_blk + blk) * BLOCK + k * 128
                    for j in tl.static_range(0, 4):  # shift index
                        shift = 2 * j
                        # lo16
                        q2_lo = ((tl.load(w_qs_row + base_q2 + j16) >> shift) & 0x03).to(tl.int32)
                        q8_lo = tl.load(q8_qs_ptr + base_q8 + j * 32 + j16).to(tl.int32)
                        s_lo = tl.sum(q2_lo * q8_lo, axis=0)
                        isum = isum + scale_lo[is_idx] * s_lo
                        is_idx += 1

                        # hi16
                        q2_hi = ((tl.load(w_qs_row + base_q2 + 16 + j16) >> shift) & 0x03).to(tl.int32)
                        q8_hi = tl.load(q8_qs_ptr + base_q8 + j * 32 + 16 + j16).to(tl.int32)
                        s_hi = tl.sum(q2_hi * q8_hi, axis=0)
                        isum = isum + scale_lo[is_idx] * s_hi
                        is_idx += 1

                d_blk = tl.load(w_d_row + blk).to(tl.float32)
                dmin_blk = tl.load(w_dmin_row + blk).to(tl.float32)
                d_q8 = tl.load(q8_d_ptr + q8_blk + blk).to(tl.float32)

                acc += d_q8 * d_blk * isum.to(tl.float32) - d_q8 * dmin_blk * summs.to(tl.float32)

        tl.store(out_ptr + row_id, acc)


    def q2_K_accum_dot_triton(
        w_scales, w_qs, w_d, w_dmin,
        q8_qs, q8_d, q8_bsums,
    ):
        """Run the Q2_K accumulated dot kernel.

        Shapes:
            w_scales : (n_experts, n_rows, n_blocks, 16) uint8
            w_qs     : (n_experts, n_rows, n_blocks, 64) uint8
            w_d      : (n_experts, n_rows, n_blocks)     float16
            w_dmin   : (n_experts, n_rows, n_blocks)     float16
            q8_qs    : (n_experts, n_blocks, 256)        int8
            q8_d     : (n_experts, n_blocks)             float32
            q8_bsums : (n_experts, n_blocks, 16)         int16
        """
        import torch

        n_experts, n_rows, n_blocks = w_d.shape
        out = torch.empty((n_rows,), dtype=torch.float32, device=w_qs.device)
        grid = (n_rows,)
        _q2_K_accum_dot_kernel[grid](
            w_scales.contiguous(), w_qs.contiguous(), w_d.contiguous(), w_dmin.contiguous(),
            q8_qs.contiguous(), q8_d.contiguous(), q8_bsums.contiguous(),
            out,
            n_experts, n_rows, n_blocks,
            BLOCK=QK_K, N_SCALES=16,
        )
        return out

else:

    def q2_K_accum_dot_triton(*args, **kwargs):
        raise RuntimeError("Triton not installed; use the numpy reference on CPU.")
