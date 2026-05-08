#!/usr/bin/env python3
"""Sanity-check the hybrid checkpoint's quantized sub-tensors.

Verifies that the 6 sub-tensors per layer have the expected byte
distributions, dtypes, and value ranges. Catches converter bugs (wrong
byte layout, swapped scales/qs, dtype mismatches) without needing to
load the full model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def stat(arr: np.ndarray, name: str, *, expected: str = "") -> dict:
    out = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "any_nan": bool(np.isnan(arr.astype(np.float32, copy=False)).any())
                   if arr.dtype != np.uint8 else False,
        "any_inf": bool(np.isinf(arr.astype(np.float32, copy=False)).any())
                   if arr.dtype != np.uint8 else False,
    }
    if expected:
        out["expected"] = expected
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/models/deepseek-v4-flash-ds4-q2",
                   help="Hybrid checkpoint dir")
    p.add_argument("--layer", type=int, default=0,
                   help="Which layer to inspect")
    p.add_argument("--expert", type=int, default=0,
                   help="Which expert (within the layer) to inspect")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    idx_path = ckpt / "model.safetensors.index.json"
    if not idx_path.exists():
        raise SystemExit(f"index not found: {idx_path}")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    prefix = f"model.layers.{args.layer}.mlp.experts"
    targets = [
        ("w13_iq2xxs_qs", "uint8, 64 bytes per block; bytes are grid indices + sign indices + 4-bit scale"),
        ("w13_iq2xxs_d", "float16, per-block d scale; expected magnitudes ~1e-3 to ~1e-1"),
        ("w2_q2k_qs", "uint8, 64 bytes per block; 4 packed 2-bit values per byte"),
        ("w2_q2k_scales", "uint8, 16 bytes per block; low nibble = scale, high nibble = min weight"),
        ("w2_q2k_d", "float16; expected ~1e-3 to ~1e-1"),
        ("w2_q2k_dmin", "float16; expected similar to d"),
    ]

    print(f"=== Layer {args.layer} expert {args.expert} sub-tensor sanity ===\n")
    for sub, expectation in targets:
        full = f"{prefix}.{sub}"
        if full not in weight_map:
            print(f"  MISSING: {full}")
            continue
        shard = weight_map[full]
        with safe_open(str(ckpt / shard), framework="pt") as f:
            t = f.get_tensor(full)
        # Slice to one expert if possible
        if t.dim() >= 1 and t.shape[0] > args.expert:
            sliced = t[args.expert]
        else:
            sliced = t
        arr = sliced.cpu().numpy()
        s = stat(arr, sub, expected=expectation)
        print(f"  {s['name']:18s}  shape={str(s['shape']):28s}  "
              f"dtype={s['dtype']:8s}  "
              f"min={s['min']!r:>10s}  max={s['max']!r:>10s}  "
              f"nan={s['any_nan']} inf={s['any_inf']}")
        print(f"     ↳ expected: {expectation}")

        # Format-specific deeper checks.
        if sub == "w13_iq2xxs_qs" and arr.size > 0:
            # The full block is 64 bytes containing 16 uint32 (8 sub-blocks × 2 uint32).
            # The aux32_1 of each sub-block has scale in top 4 bits (must be 0..15).
            # Let's sample first row's first block.
            first_block = arr.reshape(-1, 64)[0]  # 64 bytes
            aux = first_block.view(np.uint32).reshape(-1, 2)
            scales = aux[:, 1] >> 28
            grid_idx_b0 = aux[:, 0] & 0xFF
            print(f"     ↳ first block sub-scales (top 4 bits, expect 0..15): "
                  f"{scales.tolist()}")
            print(f"     ↳ first block grid indices (byte 0, expect 0..255): "
                  f"{grid_idx_b0.tolist()}")
        elif sub == "w2_q2k_scales" and arr.size > 0:
            first = arr.reshape(-1, 16)[0]
            scale_lo = first & 0x0F
            scale_hi = first >> 4
            print(f"     ↳ first block scale_lo (expect 0..15): {scale_lo.tolist()}")
            print(f"     ↳ first block scale_hi (expect 0..15): {scale_hi.tolist()}")
        elif sub.endswith("_d") or sub.endswith("_dmin"):
            f32 = arr.astype(np.float32, copy=False)
            print(f"     ↳ |d| stats: mean={float(np.abs(f32).mean()):.4e}  "
                  f"median={float(np.median(np.abs(f32))):.4e}  "
                  f"#zero={int((f32 == 0).sum())}")

        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
