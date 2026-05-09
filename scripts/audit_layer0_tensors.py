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

# Per-layer count to see if some layers have full coverage and some don't
import re
import collections
counts = collections.Counter()
for k in keys:
    m = re.search(r"\.layers\.(\d+)\.", k)
    if m:
        counts[int(m.group(1))] += 1
print(f"=== per-layer tensor count (first 5 + last 5) ===")
sorted_layers = sorted(counts.items())
for li, c in sorted_layers[:5] + sorted_layers[-5:]:
    print(f"  layer {li}: {c} tensors")
print()

# Top-level (non-layer) tensor names: embedding, lm_head, final norm, etc.
non_layer = sorted(k for k in keys if not re.search(r"\.layers\.\d+\.", k))
print(f"=== non-layer tensors ({len(non_layer)}) ===")
for k in non_layer[:30]:
    print(f"  {k}")
print()

# Pick a layer with many tensors and dump its names — what does a "complete" layer look like in our ckpt?
target_layer = max(counts, key=counts.get)
print(f"=== layer {target_layer} (max coverage, {counts[target_layer]} tensors) ===")
for k in sorted(k for k in keys if f".layers.{target_layer}." in k):
    print(f"  {k}")
print()

scales = sorted(k for k in keys if "scale" in k.lower() and "q2k_scales" not in k)
print(f"=== scale tensors (excluding our q2k_scales): {len(scales)} ===")
print(f"=== first 20 scale tensor names: ===")
for k in scales[:20]:
    print(f"  {k}")
