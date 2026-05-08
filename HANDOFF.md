# Handoff — ds4-spark-vllm

State as of pause for design review. Pure-software side of the project is closed
out and validated; the only remaining gap is upstream DeepGEMM SM12x
HyperConnection kernel compatibility — not specific to our 2-bit work.

## What works

| | |
|---|---|
| ds4 IQ2_XXS / Q2_K / Q8_K Triton kernels | Bit-exact vs ds4 C, validated on SM121 |
| `Ds4HybridIq2Config` + plugin entry-point registration | `--quantization deepseek_v4_hybrid_iq2` available; auto-detected from checkpoint |
| Hybrid checkpoint converter (GGUF → safetensors) | 83 GB output, 17 shards, FP8 dense from sgl-project |
| Modded vLLM image (`vllm-ds4-flash:latest`) | vLLM mainline + DeepGEMM (jasl + cherry-picked SM120 HC) + our mod |
| **Model loads end-to-end on Spark** | 81.4 GiB resident, 35.5 s |
| Custom expert tensor names load via patched `load_weights` | `name_mapped` fallback works |

## What's blocking first inference

A single op: DeepGEMM's HyperConnection kernel CUBIN, when JIT-compiled for
`sm_120f` family arch, is rejected by the SM121 driver with
`CUDA_ERROR_INVALID_IMAGE` at `cuModuleLoad`. The SM120 HC implementation
exists (cherry-picked from jasl tag `checkpoint/sm120-before-paged-mqa-tile`)
and compiles cleanly, but doesn't actually run on SM121 hardware — likely a
shared-memory cap (SM12x = ~101 KB vs SM100 = 232 KB) or TMA-descriptor quirk.

This is a separate ecosystem gap from our 2-bit work. Affects every
DSv4-Flash deployment on consumer Blackwell, not just ours.

## The 12-layer fix stack we built up

Each one was a real defensive issue surfaced during integration; none were
self-inflicted:

| # | Fix | Why |
|---|---|---|
| 1 | `@register_quantization_config` decorator + `vllm.general_plugins` entry point | Brittle string-patching of `QUANTIZATION_METHODS` doesn't work; vLLM has a proper plugin discovery mechanism |
| 2 | Modded image with current vLLM mainline | Image needed ≥ PR #40860 (DSv4-Flash arch) merged April 27 |
| 3 | Phantom-tensor tolerance in converter | sgl-project FP8 index lists `layers.X.attn.wo_a.scale` that doesn't exist in any shard (wo_a stays at BF16) |
| 4 | Hermes stack stopped → 115 GB available | STT + diarizer + gateway hold ~25 GB resident; vLLM at 0.85 utilization needs 101 GB |
| 5 | DeepGEMM (jasl) install: `--no-build-isolation`, non-editable, with submodules | `pip install -e` triggers setuptools `develop` wrapper which recursively `pip install -e . --use-pep517`, ignoring `--no-build-isolation` from the outer call. Submodules required (cutlass) |
| 6 | `--enforce-eager` to bypass the tilelang/flashinfer.comm import chain | `tilelang/lib/libcudart_stub.so` has a missing `cudaDeviceReset` symbol that ctypes resolves to before real libcudart |
| 7 | `deepseek_v4.py:1422` — initialize `name_mapped = None` before inner loop, fall through to default loader | vLLM's `load_weights` raises `UnboundLocalError` on any expert tensor whose name doesn't match the standard w1/w2/w3 mapping. Our `w13_iq2xxs_qs` etc. trip this |
| 8 | Force-reinstall DeepGEMM in `run.sh` on every container start | The image's pre-built deep_gemm needed to be replaced by our patched one each run |
| 9 | Cherry-pick SM120 HC kernel files from jasl tag `checkpoint/sm120-before-paged-mqa-tile` | Mainline has no SM12x HC; without these files the dispatch hits "Unsupported architecture" |
| 10 | Wipe `build/` before `pip install` | setup.py's incremental .o keeps `python_api.o` cached based on mtime; in-place header patches don't trigger recompile, so the .so was stale |
| 11 | `LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so` | TileLang ships its own `libnvrtc_stub.so` that resolves first via dlopen, blocking real NVRTC symbols TileLang then looks up at JIT time |
| 12 | DeepGEMM `get_arch()` returns `sm_120f` family target instead of `sm_121a` | Same shape as eugr issue #143 (`sm120_only` → `sm120_family` for `__CUDA_ARCH__ == 1210`); patch applied **before** `pip install` so the rebuilt binary picks it up |

## Current Spark state (post-pause)

- Qwen container `vllm-qwen35` restarted ✅
- Hermes Agent stack restarted (stt-sidecar :8001, diarize-sidecar :8002, hermes_cli gateway) ✅
- DSv4-Flash hybrid checkpoint preserved at `/home/ent/models/deepseek-v4-flash-ds4-q2/` (83 GB)
- Antirez GGUF preserved at `/home/ent/models/antirez-q2/` (81 GB)
- DeepGEMM source tree at `/home/ent/extras/DeepGEMM/` (with cherry-picks applied; root-owned)
- Modded image `vllm-ds4-flash:latest` retained
- `vllm-ds4` container removed

## How to resume (when HC kernel ready)

```bash
# On Spark, with modded image already built and checkpoint already converted:
docker stop vllm-qwen35
docker run -d --gpus all --name vllm-ds4 --network host \
  -v /home/ent/models:/models -v /home/ent/ds4-spark-vllm:/work \
  -v /home/ent/logs:/logs -v /home/ent/extras:/extras \
  -e DG_LOCAL=/extras/DeepGEMM \
  -e LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so \
  --entrypoint bash vllm-ds4-flash:latest \
  -c "bash /work/eugr_mod/mods/ds4-2bit-deepseek-v4-flash/run.sh > /logs/serve-mod.log 2>&1 && \
      vllm serve /models/deepseek-v4-flash-ds4-q2 \
        --served-model-name dsv4 --quantization deepseek_v4_hybrid_iq2 \
        --port 8000 --host 0.0.0.0 \
        --max-model-len 16384 --gpu-memory-utilization 0.85 \
        --kv-cache-dtype fp8 --attention-backend FLASHINFER \
        --load-format fastsafetensors --enforce-eager 2>&1 | tee /logs/serve.log"
```

Once HC works, the next checks are (a) does it reach "Application startup
complete", (b) does `curl localhost:8000/v1/completions` return coherent
text, (c) logit-parity vs ds4 on antirez's test vectors.

## Next paths (in order of recommendation)

1. **Wait for upstream**. vLLM tracking issue
   [#41063](https://github.com/vllm-project/vllm/issues/41063) tracks SM 12.x
   coverage. DeepGEMM issue
   [#317](https://github.com/deepseek-ai/DeepGEMM/issues/317) tracks the HC
   gap specifically. Path of least engineering work.
2. **Mine `lmxxf/vllm-deepseek-v4-dgx-spark` Docker image** for their TileLang
   HC kernel (their README explicitly mentions a TileLang HC replacement).
   1–2 days if extractable.
3. **Write a Triton/TileLang HC kernel from scratch.** Math is simple
   (per-row sum-of-squares + bf16×fp32 → fp32 GEMM, both reductions over the
   same K). Performance can be 2–4× slower than SM100 — fine for v1.
   Estimated 5–10 days.
4. **Debug DeepGEMM's CUBIN rejection** — `cuobjdump` the failing module,
   identify the offending instruction or shmem allocation, patch source.
   May or may not be tractable.

## Pointers for whoever picks this up

- Investigation report from the first agent (Path 3-lite recommendation): in
  conversation history. Key finding: jasl's `checkpoint/sm120-before-paged-mqa-tile`
  tag has SM120 HC files but they don't actually run on SM121.
- All 12 fixes live in `eugr_mod/mods/ds4-2bit-deepseek-v4-flash/run.sh`.
- The vLLM monkey-patches (load_weights + DeepGEMM get_arch) are applied on
  every container start by the run.sh.
- Tests on Mac side stay green: `pytest tests/` — 34 passed, 3 skipped (those
  3 require CUDA and have been validated on Spark separately).
