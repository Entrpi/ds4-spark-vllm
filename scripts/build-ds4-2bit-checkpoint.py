#!/usr/bin/env python3
"""Build a hybrid IQ2_XXS+Q2_K+FP8 checkpoint for vLLM.

Sources:
    --gguf  : antirez DSv4-Flash GGUF (q2 variant, 86.7 GB on disk)
    --fp8   : Hugging Face repo for the dense FP8 source
              (default: sgl-project/DeepSeek-V4-Flash-FP8)

Output: a directory of safetensors shards + index + config.json + tokenizer
files, ready to be served by a vLLM that has the deepseek_v4_hybrid_iq2
quantization method registered.

The script processes one transformer layer at a time so peak RAM is bounded
by ~one layer of expert tensors (a few GB for DSv4-Flash).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root.
_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.exists() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from ds4_hybrid_quant.builder import (  # noqa: E402
    build_layer_expert_subtensors,
    get_fp8_dense_manifest,
    hf_tensor_name_for,
    iter_layer_expert_tensors,
    open_gguf,
    write_quantization_config,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="Path to antirez q2 GGUF")
    p.add_argument(
        "--fp8", default="sgl-project/DeepSeek-V4-Flash-FP8",
        help="HF repo for dense FP8 source",
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--shard-bytes", type=int, default=5_000_000_000,
        help="Approx target shard size (default 5 GB)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without downloading or writing tensors",
    )
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Open GGUF, identify per-layer expert tensors.
    log = logging.getLogger("build")
    log.info("[1/5] Opening GGUF: %s", args.gguf)
    reader, tensors, metadata = open_gguf(args.gguf)
    by_layer = iter_layer_expert_tensors(tensors)
    log.info("  found %d layers with routed-expert tensors", len(by_layer))

    # Sniff hidden / intermediate / n_experts from GGUF metadata.
    n_experts = int(metadata.get("deepseek4.expert_count",
                                 metadata.get("expert_count", 256)))
    hidden = int(metadata.get("deepseek4.embedding_length",
                              metadata.get("hidden_size", 4096)))
    intermediate = int(metadata.get(
        "deepseek4.expert_feed_forward_length",
        metadata.get("moe_intermediate_size", 2048),
    ))
    log.info("  n_experts=%d  hidden=%d  intermediate=%d",
             n_experts, hidden, intermediate)

    # 2. FP8 dense manifest.
    log.info("[2/5] Fetching FP8 dense manifest from %s", args.fp8)
    fp8_manifest = get_fp8_dense_manifest(args.fp8)
    log.info("  non-expert tensors: %d", len(fp8_manifest))

    if args.dry_run:
        log.info("DRY RUN: stopping before any tensor writes")
        return 0

    # 3. Stream output shards: one shard accumulator, flush when full.
    from safetensors.torch import save_file as save_st
    import torch

    weight_map: dict[str, str] = {}
    current_shard: dict[str, "torch.Tensor"] = {}
    current_bytes = 0
    shard_idx = 0

    def shard_name(i: int) -> str:
        return f"model-{i:05d}.safetensors"

    def flush_shard():
        nonlocal current_shard, current_bytes, shard_idx
        if not current_shard:
            return
        path = out_dir / shard_name(shard_idx)
        save_st(current_shard, str(path))
        log.info("  wrote %s (%d tensors, %.2f GB)",
                 path.name, len(current_shard), current_bytes / 1e9)
        for name in current_shard:
            weight_map[name] = path.name
        current_shard = {}
        current_bytes = 0
        shard_idx += 1

    def add_tensor(name: str, t: "torch.Tensor"):
        nonlocal current_bytes
        current_shard[name] = t
        current_bytes += t.numel() * t.element_size()
        if current_bytes >= args.shard_bytes:
            flush_shard()

    # 4. Per-layer: repack experts and add to shard stream.
    log.info("[3/5] Repacking routed-expert tensors layer-by-layer")
    for layer in sorted(by_layer.keys()):
        log.info("  layer %d", layer)
        sub = build_layer_expert_subtensors(
            by_layer[layer], reader=reader,
            n_experts=n_experts, hidden=hidden, intermediate=intermediate,
        )
        for k, arr in sub.items():
            t = torch.from_numpy(np.ascontiguousarray(arr))
            add_tensor(hf_tensor_name_for(layer, k), t)

    # 5. Pull non-expert FP8 tensors and add them.
    log.info("[4/5] Pulling non-expert FP8 tensors from %s", args.fp8)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    shards_to_files = {}
    for name, shard in fp8_manifest.items():
        shards_to_files.setdefault(shard, []).append(name)

    for shard, names in shards_to_files.items():
        log.info("  downloading %s (%d tensors)", shard, len(names))
        path = hf_hub_download(args.fp8, shard)
        with safe_open(path, framework="pt") as f:
            for name in names:
                add_tensor(name, f.get_tensor(name))

    flush_shard()

    # 6. Index + config.
    log.info("[5/5] Writing index, config, and patching quantization_config")
    total_size = sum(
        t.numel() * t.element_size()
        for shard_path in out_dir.glob("model-*.safetensors")
        for t in []  # we already accumulated total via add_tensor; recompute below
    )
    # Recompute from final files for accuracy.
    total_size = 0
    for sp in out_dir.glob("model-*.safetensors"):
        with safe_open(str(sp), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                total_size += t.numel() * t.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    # Pull config.json + tokenizer files from FP8 source if not already present.
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "chat_template.jinja"):
        if not (out_dir / fname).exists():
            try:
                src = hf_hub_download(args.fp8, fname)
                shutil.copy2(src, out_dir / fname)
            except Exception:
                log.warning("  skipping %s (not in FP8 source)", fname)

    write_quantization_config(out_dir)
    log.info("Done. Output: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
