"""Phase 0.5 attention microbench.

Loads DSv4-Flash via vllm.LLM (offline API), warms up, then runs a short
generation under torch.profiler. Decomposes the per-token decode budget
by op so we can see where the 86 ms residual is actually going.

Output: top-30 CUDA ops by total time. We're looking for:
  - HC TileLang fused kernels (`mhc_*`)
  - Sparse MLA decode kernel (`fp8ds_global_paged_*`)
  - FP8 dense GEMMs (per-shape)
  - Indexer (`fp8_paged_mqa_*` + Python loop overhead)
  - Compressor + RoPE + RMSNorm fused ops
  - Shared expert (per-layer FP8 GEMMs)
  - Our own kernels (`iq2_xxs_pair_dot`, `q2_K_accum_dot`, `q8_K_quantize`)

Run inside the lmxxf container after run-on-lmxxf.sh has been invoked:
    python3 /work/scripts/microbench_attention.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.profiler


def main() -> None:
    # Match the working serve config from PATH_B docs.
    os.environ.setdefault("VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE", "0")

    from vllm import LLM, SamplingParams

    print(f"[microbench_attn] Loading model ({time.strftime('%H:%M:%S')}); ~3 min...", flush=True)
    t_load_start = time.perf_counter()
    llm = LLM(
        model="/models/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target",
        quantization="deepseek_v4_hybrid_iq2",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        dtype="bfloat16",
        block_size=256,
    )
    print(f"[microbench_attn] Loaded in {time.perf_counter() - t_load_start:.1f}s", flush=True)

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    prompts = ["The capital of France is"]

    # Warmup — first call materializes JIT caches.
    print("[microbench_attn] Warmup generation...", flush=True)
    t = time.perf_counter()
    out = llm.generate(prompts, sp)
    print(f"[microbench_attn] Warmup done in {time.perf_counter() - t:.2f}s", flush=True)
    print(f"[microbench_attn] Sample output: {out[0].outputs[0].text!r}", flush=True)

    # Two more warmup runs to fully prime kernel caches.
    for _ in range(2):
        llm.generate(prompts, sp)

    # Profile a longer generation to get many decode steps.
    sp_long = SamplingParams(max_tokens=16, temperature=0.0)
    print("[microbench_attn] Profiling 16-token generation...", flush=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        out = llm.generate(prompts, sp_long)
    print(f"[microbench_attn] Generation: {out[0].outputs[0].text!r}", flush=True)

    # Top ops sorted by CUDA total.
    print()
    print("=" * 110)
    print("Top 40 CUDA ops by total time (across all decode + prefill):")
    print("=" * 110)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

    # Top ops by self-CUDA time (excludes child ops).
    print()
    print("=" * 110)
    print("Top 30 CUDA ops by SELF-CUDA time:")
    print("=" * 110)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
