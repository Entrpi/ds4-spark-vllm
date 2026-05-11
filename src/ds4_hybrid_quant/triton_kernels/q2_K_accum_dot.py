"""Q2_K batched dot product kernel.

Computes the down projection for one expert across M tokens:

    out[m, r] = Q2_K_row[r] dot Q8_K_activation[m]

The kernel takes a single expert's weights ``(n_rows, n_blocks, ...)``
plus M tokens' Q8_K-quantized post-SwiGLU activations
``(M, n_blocks, ...)`` and produces ``(M, n_rows)`` outputs. No
per-token or per-expert weighting is applied here — caller multiplies
by ``topk_weights`` and scatter-adds back to the output buffer.

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
        # Weight (n_rows, n_blocks, ...) -- single expert.
        w_scales_ptr,   # *uint8,   (n_rows, n_blocks, 16)
        w_qs_ptr,       # *uint8,   (n_rows, n_blocks, 64)
        w_d_ptr,        # *float16, (n_rows, n_blocks)
        w_dmin_ptr,     # *float16, (n_rows, n_blocks)
        # Activation (M, n_blocks, ...) -- M tokens routed to this expert.
        q8_qs_ptr,      # *int8,    (M, n_blocks, 256)
        q8_d_ptr,       # *float32, (M, n_blocks)
        q8_bsums_ptr,   # *int16,   (M, n_blocks, 16)
        # Output (M, n_rows)
        out_ptr,        # *float32, (M, n_rows)
        n_rows,
        n_blocks,
        stride_q8_qs_m,    # = n_blocks * 256
        stride_q8_d_m,     # = n_blocks
        stride_q8_bsums_m, # = n_blocks * 16
        stride_out_m,      # = n_rows
        BLOCK: tl.constexpr,    # = 256
        N_SCALES: tl.constexpr,  # = 16
    ):
        """One program produces one ``out[m, r]`` value.

        Grid is ``(M, n_rows)``. All M programs sharing a row_id read
        identical weight bytes — automatic L2 reuse on the weight tensor.
        """
        m_id = tl.program_id(0)
        row_id = tl.program_id(1)

        # Per-row weight base addresses (shared across M).
        w_scales_row = w_scales_ptr + row_id * n_blocks * 16
        w_qs_row = w_qs_ptr + row_id * n_blocks * 64
        w_d_row = w_d_ptr + row_id * n_blocks
        w_dmin_row = w_dmin_ptr + row_id * n_blocks
        # Per-token q8 base addresses.
        q8_qs_tok = q8_qs_ptr + m_id * stride_q8_qs_m
        q8_d_tok = q8_d_ptr + m_id * stride_q8_d_m
        q8_bsums_tok = q8_bsums_ptr + m_id * stride_q8_bsums_m

        acc = tl.zeros((), dtype=tl.float32)

        for blk in range(0, n_blocks):
            # Load q8 bsums[16] and reduce summs.
            scale_offs = tl.arange(0, N_SCALES)
            scales = tl.load(w_scales_row + blk * 16 + scale_offs)
            scale_hi = (scales >> 4).to(tl.int32)
            bsums = tl.load(
                q8_bsums_tok + blk * 16 + scale_offs
            ).to(tl.int32)
            summs = tl.sum(bsums * scale_hi, axis=0)

            # Per-block isum: 2 outer * 4 shifts * 2 halves = 16 dot pieces.
            # Each piece is a 16-element int dot. We load each scale_lo
            # nibble directly from memory via offset (Triton tensors
            # don't support Python-int subscripting inside @jit).
            isum = tl.zeros((), dtype=tl.int32)
            is_idx = 0
            j16 = tl.arange(0, 16)
            for k in tl.static_range(0, 2):  # QK_K/128
                base_q2 = blk * 64 + k * 32
                base_q8 = blk * BLOCK + k * 128
                for j in tl.static_range(0, 4):  # shift index
                    shift = 2 * j
                    # lo16
                    sc_lo = (tl.load(w_scales_row + blk * 16 + is_idx) & 0x0F).to(tl.int32)
                    q2_lo = ((tl.load(w_qs_row + base_q2 + j16) >> shift) & 0x03).to(tl.int32)
                    q8_lo = tl.load(q8_qs_tok + base_q8 + j * 32 + j16).to(tl.int32)
                    s_lo = tl.sum(q2_lo * q8_lo, axis=0)
                    isum = isum + sc_lo * s_lo
                    is_idx += 1

                    # hi16
                    sc_hi = (tl.load(w_scales_row + blk * 16 + is_idx) & 0x0F).to(tl.int32)
                    q2_hi = ((tl.load(w_qs_row + base_q2 + 16 + j16) >> shift) & 0x03).to(tl.int32)
                    q8_hi = tl.load(q8_qs_tok + base_q8 + j * 32 + 16 + j16).to(tl.int32)
                    s_hi = tl.sum(q2_hi * q8_hi, axis=0)
                    isum = isum + sc_hi * s_hi
                    is_idx += 1

            d_blk = tl.load(w_d_row + blk).to(tl.float32)
            dmin_blk = tl.load(w_dmin_row + blk).to(tl.float32)
            d_q8 = tl.load(q8_d_tok + blk).to(tl.float32)

            acc += d_q8 * d_blk * isum.to(tl.float32) - d_q8 * dmin_blk * summs.to(tl.float32)

        tl.store(out_ptr + m_id * stride_out_m + row_id, acc)


    def q2_K_accum_dot_triton(
        w_scales, w_qs, w_d, w_dmin,
        q8_qs, q8_d, q8_bsums,
    ):
        """Run the batched Q2_K dot kernel for one expert × M tokens.

        Shapes:
            w_scales : (n_rows, n_blocks, 16) uint8
            w_qs     : (n_rows, n_blocks, 64) uint8
            w_d      : (n_rows, n_blocks)     float16
            w_dmin   : (n_rows, n_blocks)     float16
            q8_qs    : (M, n_blocks, 256)     int8
            q8_d     : (M, n_blocks)          float32
            q8_bsums : (M, n_blocks, 16)      int16

        Returns:
            out : (M, n_rows) float32
        """
        import torch

        n_rows, n_blocks = w_d.shape
        if q8_qs.ndim != 3 or q8_d.ndim != 2 or q8_bsums.ndim != 3:
            raise ValueError(
                f"q8_qs must be (M, n_blocks, 256), q8_d (M, n_blocks), "
                f"q8_bsums (M, n_blocks, 16); got "
                f"q8_qs.shape={tuple(q8_qs.shape)} "
                f"q8_d.shape={tuple(q8_d.shape)} "
                f"q8_bsums.shape={tuple(q8_bsums.shape)}"
            )
        M = q8_qs.shape[0]
        if q8_d.shape[0] != M or q8_bsums.shape[0] != M:
            raise ValueError("q8 inputs must agree on M")

        out = torch.empty((M, n_rows), dtype=torch.float32, device=w_qs.device)

        q8_qs_c = q8_qs.contiguous()
        q8_d_c = q8_d.contiguous()
        q8_bsums_c = q8_bsums.contiguous()
        stride_q8_qs_m = n_blocks * QK_K
        stride_q8_d_m = n_blocks
        stride_q8_bsums_m = n_blocks * 16
        stride_out_m = n_rows

        grid = (M, n_rows)
        _q2_K_accum_dot_kernel[grid](
            w_scales.contiguous(), w_qs.contiguous(),
            w_d.contiguous(), w_dmin.contiguous(),
            q8_qs_c, q8_d_c, q8_bsums_c,
            out,
            n_rows, n_blocks,
            stride_q8_qs_m, stride_q8_d_m, stride_q8_bsums_m, stride_out_m,
            BLOCK=QK_K, N_SCALES=16,
        )
        return out

else:

    def q2_K_accum_dot_triton(*args, **kwargs):
        raise RuntimeError("Triton not installed; use the numpy reference on CPU.")
