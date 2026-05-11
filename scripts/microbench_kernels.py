"""Microbench the three ds4 Triton kernels at realistic DSv4-Flash shapes.

Goal: validate the kernel-bound diagnosis with hard numbers. For each
kernel, time per-call latency at M ∈ {1, 6, 16, 64, 128}. The signal
we want: per-call cost grows much less than linearly with M (the
scalar-per-row pattern is the bottleneck, not the actual GPU work),
which would confirm Path B's lift potential.

DSv4-Flash shapes:
    hidden_size = 4096   → n_blocks_in  = 16
    moe_intermediate_size = 2048 → n_blocks_int = 8
    n_routed_experts = 256, top_k = 6, layers = 43

Run inside the lmxxf container with PYTHONPATH=/work/src:
    python3 /work/scripts/microbench_kernels.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/src")  # in case PYTHONPATH not set

from ds4_hybrid_quant.lookup_tables import QK_K
from ds4_hybrid_quant.triton_kernels.iq2_xxs_pair_dot import iq2_xxs_pair_dot_triton
from ds4_hybrid_quant.triton_kernels.q2_K_accum_dot import q2_K_accum_dot_triton
from ds4_hybrid_quant.triton_kernels.q8_K_quantize import quantize_q8_K_triton


HIDDEN = 4096
INTER = 2048
N_BLOCKS_IN = HIDDEN // QK_K       # 16
N_BLOCKS_INT = INTER // QK_K       # 8
DEV = "cuda"
WARMUP = 100
ITERS = 500
MS_PER_S = 1e3


def now_ms() -> float:
    torch.cuda.synchronize()
    return time.perf_counter() * MS_PER_S


def bench_iq2_xxs_pair(M: int) -> tuple[float, dict]:
    """Time iq2_xxs_pair_dot for M tokens × n_rows=INTER (gate+up)."""
    n_rows = INTER
    n_blocks = N_BLOCKS_IN
    # Random uint8 weights — payload doesn't matter for timing.
    w_a_qs = torch.randint(0, 256, (n_rows, n_blocks, 64), dtype=torch.uint8, device=DEV)
    w_a_d = torch.randn(n_rows, n_blocks, dtype=torch.float16, device=DEV).abs() * 0.01
    w_b_qs = torch.randint(0, 256, (n_rows, n_blocks, 64), dtype=torch.uint8, device=DEV)
    w_b_d = torch.randn(n_rows, n_blocks, dtype=torch.float16, device=DEV).abs() * 0.01
    q8_qs = torch.randint(-128, 128, (M, n_blocks, 256), dtype=torch.int8, device=DEV)
    q8_d = torch.randn(M, n_blocks, dtype=torch.float32, device=DEV) * 0.01

    # Warmup.
    for _ in range(WARMUP):
        out_a, out_b = iq2_xxs_pair_dot_triton(w_a_qs, w_a_d, w_b_qs, w_b_d, q8_qs, q8_d)
    # Measure.
    t0 = now_ms()
    for _ in range(ITERS):
        out_a, out_b = iq2_xxs_pair_dot_triton(w_a_qs, w_a_d, w_b_qs, w_b_d, q8_qs, q8_d)
    elapsed = now_ms() - t0
    per_call_ms = elapsed / ITERS

    # Shape info.
    return per_call_ms, {"M": M, "n_rows": n_rows, "n_blocks": n_blocks}


def bench_q2_K_accum(M: int) -> tuple[float, dict]:
    """Time q2_K_accum_dot for M tokens × n_rows=HIDDEN (down)."""
    n_rows = HIDDEN
    n_blocks = N_BLOCKS_INT
    w_scales = torch.randint(0, 256, (n_rows, n_blocks, 16), dtype=torch.uint8, device=DEV)
    w_qs = torch.randint(0, 256, (n_rows, n_blocks, 64), dtype=torch.uint8, device=DEV)
    w_d = torch.randn(n_rows, n_blocks, dtype=torch.float16, device=DEV).abs() * 0.01
    w_dmin = torch.randn(n_rows, n_blocks, dtype=torch.float16, device=DEV).abs() * 0.005
    q8_qs = torch.randint(-128, 128, (M, n_blocks, 256), dtype=torch.int8, device=DEV)
    q8_d = torch.randn(M, n_blocks, dtype=torch.float32, device=DEV) * 0.01
    q8_bsums = torch.randint(-32, 32, (M, n_blocks, 16), dtype=torch.int16, device=DEV)

    for _ in range(WARMUP):
        out = q2_K_accum_dot_triton(w_scales, w_qs, w_d, w_dmin, q8_qs, q8_d, q8_bsums)
    t0 = now_ms()
    for _ in range(ITERS):
        out = q2_K_accum_dot_triton(w_scales, w_qs, w_d, w_dmin, q8_qs, q8_d, q8_bsums)
    elapsed = now_ms() - t0
    per_call_ms = elapsed / ITERS
    return per_call_ms, {"M": M, "n_rows": n_rows, "n_blocks": n_blocks}


def bench_q8_K_quantize(M_blocks: int) -> tuple[float, dict]:
    """Time quantize_q8_K for M_blocks total blocks (256 floats each)."""
    x = torch.randn(M_blocks, QK_K, dtype=torch.float32, device=DEV)
    for _ in range(WARMUP):
        qs, d, bsums = quantize_q8_K_triton(x)
    t0 = now_ms()
    for _ in range(ITERS):
        qs, d, bsums = quantize_q8_K_triton(x)
    elapsed = now_ms() - t0
    per_call_ms = elapsed / ITERS
    return per_call_ms, {"M_blocks": M_blocks}


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available, aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print()
    print(f"DSv4-Flash shapes: hidden={HIDDEN}, intermediate={INTER}")
    print(f"  n_blocks_in (hidden/QK_K)={N_BLOCKS_IN}, n_blocks_int (intermediate/QK_K)={N_BLOCKS_INT}")
    print(f"  Warmup iters: {WARMUP}, measurement iters: {ITERS}")
    print()

    print("=" * 78)
    print("Kernel 1: iq2_xxs_pair_dot — grid (M, n_rows=INTER) for gate+up")
    print("=" * 78)
    print(f"{'M':>6} | {'per_call_ms':>14} | {'per_token_us':>14} | {'efficiency':>10}")
    print("-" * 78)
    baseline_iq2 = None
    for M in [1, 6, 16, 64, 128]:
        t_ms, info = bench_iq2_xxs_pair(M)
        per_tok_us = (t_ms / M) * 1000
        if baseline_iq2 is None:
            baseline_iq2 = per_tok_us
            eff = "1.00× (ref)"
        else:
            eff = f"{baseline_iq2 / per_tok_us:.2f}×"
        print(f"{M:>6} | {t_ms:>14.4f} | {per_tok_us:>14.2f} | {eff:>10}")
    print()

    print("=" * 78)
    print("Kernel 2: q2_K_accum_dot — grid (M, n_rows=HIDDEN) for down")
    print("=" * 78)
    print(f"{'M':>6} | {'per_call_ms':>14} | {'per_token_us':>14} | {'efficiency':>10}")
    print("-" * 78)
    baseline_q2k = None
    for M in [1, 6, 16, 64, 128]:
        t_ms, info = bench_q2_K_accum(M)
        per_tok_us = (t_ms / M) * 1000
        if baseline_q2k is None:
            baseline_q2k = per_tok_us
            eff = "1.00× (ref)"
        else:
            eff = f"{baseline_q2k / per_tok_us:.2f}×"
        print(f"{M:>6} | {t_ms:>14.4f} | {per_tok_us:>14.2f} | {eff:>10}")
    print()

    print("=" * 78)
    print("Kernel 3: quantize_q8_K — grid (M_blocks,)")
    print("=" * 78)
    print(f"{'M_blocks':>10} | {'per_call_ms':>14} | {'per_block_us':>14}")
    print("-" * 78)
    # Representative sizes:
    #   M=1, 16 blocks_in   → 16 blocks
    #   M=6, 16             → 96 blocks
    #   M=128, 16           → 2048 blocks
    #   M=1, 8 blocks_int   → 8 blocks  (mid quantize for 1 token, 1 expert)
    for M_blocks in [8, 16, 96, 256, 1024, 2048]:
        t_ms, info = bench_q8_K_quantize(M_blocks)
        per_blk_us = (t_ms / M_blocks) * 1000
        print(f"{M_blocks:>10} | {t_ms:>14.4f} | {per_blk_us:>14.2f}")
    print()

    print("=" * 78)
    print("Per-layer estimated cost @ T=1 decode (top_k=6):")
    print("=" * 78)
    # Decode: 6 active experts/layer. For each:
    #   iq2 pair-dot @ M=1 + q8 quant of 8 mid blocks + q2_K accum-dot @ M=1
    iq2_m1, _ = bench_iq2_xxs_pair(1)
    q2k_m1, _ = bench_q2_K_accum(1)
    q8_8, _ = bench_q8_K_quantize(8)
    # x quantize once per layer (M=1 input → 16 blocks):
    q8_16, _ = bench_q8_K_quantize(16)
    per_layer_ms = q8_16 + 6 * (iq2_m1 + q8_8 + q2k_m1)
    print(f"  x quantize (1 call, 16 blocks):          {q8_16:.4f} ms")
    print(f"  6 × iq2_xxs_pair_dot @ M=1:              {6*iq2_m1:.4f} ms")
    print(f"  6 × q8_K quantize (mid, 8 blocks):       {6*q8_8:.4f} ms")
    print(f"  6 × q2_K_accum_dot @ M=1:                {6*q2k_m1:.4f} ms")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Total per-layer MoE cost:                {per_layer_ms:.4f} ms")
    print(f"  × 43 layers = {per_layer_ms*43:.2f} ms/token")
    print(f"  → estimated decode t/s (MoE-only): {1000/(per_layer_ms*43):.2f}")
    print()
    print("=" * 78)
    print("Per-layer estimated cost @ T=128 prefill (top_k=6):")
    print("=" * 78)
    iq2_128, _ = bench_iq2_xxs_pair(128)
    q2k_128, _ = bench_q2_K_accum(128)
    q8_2048, _ = bench_q8_K_quantize(128 * 16)
    q8_1024, _ = bench_q8_K_quantize(128 * 8)
    # Prefill: argmax tokens/expert = T*top_k/n_experts = 128*6/256 = 3, but at high T
    # there's variance. Assume avg M_e=3, but worst case M_e=128 for 1-expert pathological.
    # For envelope we use avg M_e=3 across 256 experts (most have M_e>0 at high T).
    # Simplification: total work = 256 experts × M_e=3 per expert × (iq2 + q2k + q8 quant)
    # ≈ equivalently, M=128 batched once across all selected (top_k*T) expert-slots.
    # Use kernel-at-M=128 as upper bound: one big batched per-expert call.
    per_layer_ms_pf = q8_2048 + 256 * (iq2_m1 + q8_8 + q2k_m1)  # if M_e averages 1
    print(f"  (showing decode-style accounting for envelope; real prefill mixes)")
    print(f"  x quantize (128*16=2048 blocks):         {q8_2048:.4f} ms")
    print(f"  256 × iq2 @ avg M_e=1 (lower-bound):     {256*iq2_m1:.4f} ms")
    print(f"  256 × q8_K (mid, 8 blocks each):         {256*q8_8:.4f} ms")
    print(f"  256 × q2_K @ avg M_e=1:                  {256*q2k_m1:.4f} ms")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Per-layer (lower bound):                 {per_layer_ms_pf:.4f} ms")
    print(f"  × 43 layers ≈ {per_layer_ms_pf*43/1000:.2f} s/prefill")


if __name__ == "__main__":
    main()
