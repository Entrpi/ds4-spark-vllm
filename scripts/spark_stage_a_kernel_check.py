#!/usr/bin/env python3
"""Spark Stage A: validate Triton kernels on SM121 with synthetic data.

Runs each of the three ds4 Triton kernels against the numpy block-level
reference (which is itself bit-exact-validated against ds4 C). Uses tiny
synthetic tensors so it can run alongside a serving vLLM container without
material memory pressure.

Exit code 0 on success, non-zero on the first failure.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np


def _section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_q8_K_quantize() -> bool:
    _section("Q8_K quantize Triton kernel")
    import torch
    from ds4_hybrid_quant.block_layouts import QK_K
    from ds4_hybrid_quant.triton_kernels.q8_K_quantize import (
        quantize_q8_K_numpy, quantize_q8_K_triton,
    )

    if not torch.cuda.is_available():
        print("CUDA not available; FAIL")
        return False

    rng = np.random.default_rng(0xA1)
    n_blocks = 8
    x_np = rng.standard_normal((n_blocks, QK_K)).astype(np.float32) * 1.3
    x_t = torch.from_numpy(x_np).cuda()

    qs_t, d_t, bsums_t = quantize_q8_K_triton(x_t)
    qs_n, d_n, bsums_n = quantize_q8_K_numpy(x_np)

    qs_match = np.array_equal(qs_t.cpu().numpy(), qs_n)
    bsums_match = np.array_equal(bsums_t.cpu().numpy(), bsums_n)
    d_match = np.array_equal(d_t.cpu().numpy(), d_n)

    print(f"qs match: {qs_match}, bsums match: {bsums_match}, d match: {d_match}")
    return qs_match and bsums_match and d_match


def check_iq2_xxs_pair_dot() -> bool:
    _section("IQ2_XXS pair dot Triton kernel")
    import torch
    from ds4_hybrid_quant.block_layouts import QK_K
    from ds4_hybrid_quant.reference import quantize_q8_K
    from ds4_hybrid_quant.test_helpers import make_iq2_xxs_blocks
    from ds4_hybrid_quant.triton_kernels.iq2_xxs_pair_dot import (
        iq2_xxs_pair_dot_triton,
        vec_dot_iq2_xxs_pair_kernel_numpy,
    )

    if not torch.cuda.is_available():
        print("CUDA not available; FAIL")
        return False

    rng = np.random.default_rng(0xA2)
    n_rows, n_blocks = 4, 2
    rows_a = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.012) for _ in range(n_rows)]
    rows_b = [make_iq2_xxs_blocks(n_blocks=n_blocks, rng=rng, d_scale=0.011) for _ in range(n_rows)]
    x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
    q8 = quantize_q8_K(x)

    w_a_qs = torch.from_numpy(np.stack([r.qs for r in rows_a])).cuda()
    w_a_d = torch.from_numpy(np.stack([r.d for r in rows_a])).cuda()
    w_b_qs = torch.from_numpy(np.stack([r.qs for r in rows_b])).cuda()
    w_b_d = torch.from_numpy(np.stack([r.d for r in rows_b])).cuda()
    q8_qs_t = torch.from_numpy(q8.qs).cuda()
    q8_d_t = torch.from_numpy(q8.d).cuda()

    out_a, out_b = iq2_xxs_pair_dot_triton(
        w_a_qs, w_a_d, w_b_qs, w_b_d, q8_qs_t, q8_d_t,
    )

    ok = True
    for i in range(n_rows):
        ra, rb = vec_dot_iq2_xxs_pair_kernel_numpy(rows_a[i], rows_b[i], q8)
        delta_a = abs(out_a[i].item() - ra)
        delta_b = abs(out_b[i].item() - rb)
        ok_row = delta_a < 1e-4 and delta_b < 1e-4
        print(f"row {i}: a triton={out_a[i].item():.6f} ref={ra:.6f} d={delta_a:.2e}, "
              f"b triton={out_b[i].item():.6f} ref={rb:.6f} d={delta_b:.2e} -> "
              f"{'PASS' if ok_row else 'FAIL'}")
        ok = ok and ok_row
    return ok


def check_q2_K_accum_dot() -> bool:
    _section("Q2_K accum dot Triton kernel")
    import torch
    from ds4_hybrid_quant.block_layouts import QK_K
    from ds4_hybrid_quant.reference import quantize_q8_K
    from ds4_hybrid_quant.test_helpers import make_q2_K_blocks
    from ds4_hybrid_quant.triton_kernels.q2_K_accum_dot import (
        q2_K_accum_dot_triton,
        vec_dot_q2_K_accum_kernel_numpy,
    )

    if not torch.cuda.is_available():
        print("CUDA not available; FAIL")
        return False

    rng = np.random.default_rng(0xA3)
    n_experts, n_rows, n_blocks = 2, 4, 2

    rows: list[list] = []
    acts = []
    for e in range(n_experts):
        rows_e = [make_q2_K_blocks(n_blocks=n_blocks, rng=rng) for _ in range(n_rows)]
        rows.append(rows_e)
        x = rng.standard_normal(n_blocks * QK_K).astype(np.float32) * 0.4
        acts.append(quantize_q8_K(x))

    w_scales = torch.from_numpy(np.stack([
        np.stack([r.scales for r in row]) for row in rows
    ])).cuda()
    w_qs = torch.from_numpy(np.stack([
        np.stack([r.qs for r in row]) for row in rows
    ])).cuda()
    w_d = torch.from_numpy(np.stack([
        np.stack([r.d for r in row]) for row in rows
    ])).cuda()
    w_dmin = torch.from_numpy(np.stack([
        np.stack([r.dmin for r in row]) for row in rows
    ])).cuda()
    q8_qs = torch.from_numpy(np.stack([a.qs for a in acts])).cuda()
    q8_d = torch.from_numpy(np.stack([a.d for a in acts])).cuda()
    q8_bsums = torch.from_numpy(np.stack([a.bsums for a in acts])).cuda()

    out = q2_K_accum_dot_triton(w_scales, w_qs, w_d, w_dmin, q8_qs, q8_d, q8_bsums)

    ok = True
    for r in range(n_rows):
        expected = vec_dot_q2_K_accum_kernel_numpy(
            [rows[e][r] for e in range(n_experts)], acts,
        )
        delta = abs(out[r].item() - expected)
        ok_row = delta < 1e-4
        print(f"row {r}: triton={out[r].item():.6f} ref={expected:.6f} d={delta:.2e} -> "
              f"{'PASS' if ok_row else 'FAIL'}")
        ok = ok and ok_row
    return ok


def main() -> int:
    print("=== Stage A: ds4 hybrid quant kernel validation on SM121 ===")
    import torch
    print(f"torch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"compute capability: sm_{cap[0]}{cap[1]}")
    try:
        import triton
        print(f"triton: {triton.__version__}")
    except Exception as e:
        print(f"triton import error: {e}")

    results = {}
    for name, fn in (
        ("q8_K_quantize", check_q8_K_quantize),
        ("iq2_xxs_pair_dot", check_iq2_xxs_pair_dot),
        ("q2_K_accum_dot", check_q2_K_accum_dot),
    ):
        try:
            results[name] = fn()
        except Exception:
            print(f"{name}: EXCEPTION")
            traceback.print_exc()
            results[name] = False

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
