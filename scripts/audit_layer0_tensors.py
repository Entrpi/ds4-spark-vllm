#!/usr/bin/env python3
"""Audit layer 0 tensor coverage in our hybrid checkpoint.

Lists all layer-0 tensor names and any scale tensors anywhere, so we can
see whether the converter retained FP8 scales (kv_a_proj_with_mqa.weight_scale_inv,
qkv_proj.weight_scale_inv, etc.) that DSv4-Flash's FP8 attention path needs.
"""
from __future__ import annotations
import json
from pathlib import Path

CKPT = Path("/models/deepseek-v4-flash-ds4-q2")
idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
keys = list(idx.keys())

l0 = sorted(k for k in keys if ".layers.0." in k)
print(f"=== layer 0 tensor count: {len(l0)} ===")
for k in l0:
    print(f"  {k}")
print()

scales = sorted(k for k in keys if "scale" in k.lower())
l0_scales = [k for k in scales if ".layers.0." in k]
print(f"=== total tensors: {len(keys)} ===")
print(f"=== tensors with 'scale' in name: {len(scales)} ===")
print(f"=== layer 0 scale tensors ({len(l0_scales)}): ===")
for k in l0_scales:
    print(f"  {k}")
