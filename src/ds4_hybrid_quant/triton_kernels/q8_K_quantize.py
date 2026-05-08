"""Q8_K activation quantization kernel.

Per-block algorithm (matches ds4_quantize_row_q8_K, ds4.c:1473):

    amax       = max(|x|)
    max_signed = x[argmax(|x|)]
    iscale     = -127 / max_signed
    qs[j]      = clip(round(iscale * x[j]), -128, 127)
    bsums[k]   = sum(qs[k*16 : (k+1)*16])  for k in 0..15
    d          = 1 / iscale  = max_signed / -127

The kernel processes one 256-element block per program; the outer grid is
``(num_blocks,)``. The block layout in memory is
``(num_tokens, num_blocks_per_token, QK_K)``.

If ``amax == 0`` the block is all zeros: ``d = 0``, ``qs = 0``, ``bsums = 0``.
"""

from __future__ import annotations

import numpy as np

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover - Mac dev path
    HAVE_TRITON = False
    triton = None
    tl = None

from ..lookup_tables import QK_K


# ---------------------------------------------------------------------------
# Numpy block-level reference (runs on any platform)
# ---------------------------------------------------------------------------


def quantize_q8_K_numpy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized numpy reference matching the per-block algorithm above.

    Args:
        x: float32, shape ``(n_blocks, QK_K)``.

    Returns:
        ``(qs, d, bsums)`` with shapes
        ``((n_blocks, QK_K) int8, (n_blocks,) float32, (n_blocks, 16) int16)``.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.ndim != 2 or x.shape[1] != QK_K:
        raise ValueError(f"expected (n_blocks, {QK_K}) input, got {x.shape}")

    nb = x.shape[0]
    abs_x = np.abs(x)
    idx = abs_x.argmax(axis=1)
    amax = abs_x[np.arange(nb), idx]
    max_signed = x[np.arange(nb), idx]

    qs = np.zeros((nb, QK_K), dtype=np.int8)
    d = np.zeros(nb, dtype=np.float32)

    nz = amax > 0.0
    if nz.any():
        iscale = (-127.0 / max_signed[nz]).astype(np.float32)
        v = np.rint(x[nz] * iscale[:, None])
        v = np.clip(v, -128.0, 127.0).astype(np.int8)
        qs[nz] = v
        d[nz] = (1.0 / iscale).astype(np.float32)

    bsums = qs.reshape(nb, QK_K // 16, 16).sum(axis=2).astype(np.int16)
    return qs, d, bsums


# ---------------------------------------------------------------------------
# Triton kernel (Spark-side; not callable on Mac without triton)
# ---------------------------------------------------------------------------


if HAVE_TRITON:

    @triton.jit
    def _q8_K_quantize_kernel(
        x_ptr,           # *float32, (n_blocks, QK_K)
        qs_ptr,          # *int8,    (n_blocks, QK_K)
        d_ptr,           # *float32, (n_blocks,)
        bsums_ptr,       # *int16,   (n_blocks, 16)
        n_blocks,        # int32 (compile-time known via grid; passed for safety)
        BLOCK: tl.constexpr,    # = QK_K (256)
        SUBBLOCK: tl.constexpr,  # = 16  (BLOCK / 16 = 16 sub-blocks per block)
    ):
        """One program quantizes one Q8_K block."""
        pid = tl.program_id(0)
        # Per-program offset into the flat (n_blocks, QK_K) tensor.
        x_row_ptr = x_ptr + pid * BLOCK
        qs_row_ptr = qs_ptr + pid * BLOCK

        offs = tl.arange(0, BLOCK)
        x = tl.load(x_row_ptr + offs)              # (BLOCK,) float32
        abs_x = tl.abs(x)

        amax = tl.max(abs_x, axis=0)               # scalar float32

        # Pick a signed value at the abs-max position. Where abs_x == amax,
        # use the signed x; elsewhere -inf (so it can't win the reduction).
        # If multiple positions tie, max() picks the largest (positive) one.
        # The dequantized result is invariant under this tie-breaking.
        neg_inf = float("-inf")
        masked = tl.where(abs_x == amax, x, neg_inf)
        max_signed = tl.max(masked, axis=0)        # scalar

        # Branch on amax==0 zero block (write zeros, d=0).
        zero_block = amax == 0.0

        # Use a tiny epsilon so we never divide by zero; result is masked off
        # for zero blocks anyway. Note: max_signed is signed so iscale is
        # signed; this matches ds4 exactly.
        iscale = tl.where(zero_block, 0.0, -127.0 / max_signed)
        # Compute d directly from max_signed to match numpy / ds4 bit-exactly:
        # d = max_signed / -127 (avoids the round-trip 1/(-127/m)).
        d_val = tl.where(zero_block, 0.0, max_signed / -127.0)

        # Quantize: lrintf in ds4 uses banker's rounding (round half to even).
        # Triton provides this via libdevice.rint on the host's CUDA target.
        scaled = x * iscale
        # Use libdevice.rint which compiles to PTX cvt.rni (round-to-nearest-even).
        rounded = tl.extra.libdevice.rint(scaled)
        rounded = tl.cast(rounded, tl.int32)
        rounded = tl.where(rounded > 127, 127, rounded)
        rounded = tl.where(rounded < -128, -128, rounded)
        qs_val = tl.cast(rounded, tl.int8)
        # For zero blocks force qs to 0.
        qs_val = tl.where(zero_block, tl.cast(0, tl.int8), qs_val)

        tl.store(qs_row_ptr + offs, qs_val)
        tl.store(d_ptr + pid, d_val)

        # bsums: 16 sub-blocks of 16 quants each. Reshape via masked sums.
        # Use a tiled reduction: per-sub-block index k, sum over j in 0..16.
        sub_offs = tl.arange(0, SUBBLOCK)  # (SUBBLOCK=16,)
        for k in tl.static_range(0, SUBBLOCK):
            # Load 16 int8 values for sub-block k and sum them.
            idx = k * 16 + sub_offs
            vals = tl.load(qs_row_ptr + idx).to(tl.int32)
            bsum = tl.sum(vals, axis=0)
            tl.store(bsums_ptr + pid * SUBBLOCK + k, tl.cast(bsum, tl.int16))


    def quantize_q8_K_triton(
        x,  # torch.Tensor float32 (n_blocks, QK_K) on CUDA
    ):
        """Run the Triton Q8_K quantize kernel.

        Returns a tuple of (qs_int8, d_float32, bsums_int16) torch tensors
        on the same device as ``x``.
        """
        import torch

        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim != 2 or x.shape[1] != QK_K:
            raise ValueError(f"expected (n_blocks, {QK_K}) input, got {x.shape}")

        n_blocks = x.shape[0]
        qs = torch.empty((n_blocks, QK_K), dtype=torch.int8, device=x.device)
        d = torch.empty((n_blocks,), dtype=torch.float32, device=x.device)
        bsums = torch.empty((n_blocks, 16), dtype=torch.int16, device=x.device)

        grid = (n_blocks,)
        _q8_K_quantize_kernel[grid](
            x, qs, d, bsums, n_blocks,
            BLOCK=QK_K, SUBBLOCK=16,
        )
        return qs, d, bsums


else:

    def quantize_q8_K_triton(*args, **kwargs):
        raise RuntimeError(
            "Triton is not installed; quantize_q8_K_triton requires CUDA. "
            "Use quantize_q8_K_numpy for CPU-side validation."
        )
