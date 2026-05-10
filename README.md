# ds4-spark-vllm

DeepSeek-V4-Flash, 2-bit hybrid, on a single NVIDIA DGX Spark, served by vLLM.

**Status:** Working end-to-end. Validated on one DGX Spark (GB10 / SM121, 128 GiB unified memory) against the [`antirez/ds4`](https://github.com/antirez/ds4) C+Metal reference implementation. Other GB10 units should behave identically; other Blackwell SKUs (B100/B200) are likely to work but unverified.

- **Checkpoint:** [`bleysg/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target) (~85 GiB, public, MIT)
- **Bring-up writeup:** [`docs/DSV4_FLASH_2BIT_SPARK_REPORT.md`](docs/DSV4_FLASH_2BIT_SPARK_REPORT.md) — full layer-by-layer bisection story, including both correctness bugs found and fixed
- **Reference:** [`antirez/ds4`](https://github.com/antirez/ds4) — the C+Metal implementation this conversion was validated against

## Quick start

On a DGX Spark with Docker + the NVIDIA Container Toolkit installed:

```bash
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-spark-vllm/main/install.sh | bash
```

That one command:

1. Verifies the host (aarch64, GB10/SM121, ≥118 GiB RAM, ≥100 GiB disk).
2. Installs the `hf` CLI if missing and prompts for a HuggingFace token (optional — checkpoint is public).
3. Downloads ~85 GiB of safetensors and verifies SHA256SUMS.
4. Pulls the [`lmxxf/vllm-deepseek-v4-dgx-spark`](https://hub.docker.com/r/lmxxf/vllm-deepseek-v4-dgx-spark) base image.
5. Starts `vllm serve` on `:8000` with all correctness-critical flags baked in.
6. Polls `/health`, then runs a first-token smoke test against the canonical `"The capital of France is"` prompt.

To preview the flags before piping into bash:

```bash
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-spark-vllm/main/install.sh | bash -s -- --help
```

Common overrides: `--port`, `--max-model-len`, `--gpu-util`, `--models-dir`, `--skip-download`, `--no-start`, `--non-interactive`, `--force`, `--uninstall`. All defaults are env-overridable.

## Hardware requirements

| | |
|---|---|
| Validated on | NVIDIA DGX Spark (GB10, SM121, 128 GiB unified) |
| Likely to work | other Blackwell + FP8 + Triton (B100/B200, H100) — untested |
| System memory | ≥118 GiB (Spark reports 119 in `/proc/meminfo`; resident during serving ~110 GiB) |
| Free disk | ≥100 GiB on `$MODELS_DIR` |
| OS | aarch64 Linux (Grace) — the base image is aarch64-only |
| Docker | engine + `nvidia-container-toolkit` |

The installer's GB10 detection (`nvidia-smi --query-gpu=name,compute_cap`) flags `compute_cap=12.1 + name~/GB10|Spark/` as the green path. `12.0` (datacenter Blackwell) gets a yellow warning that the SM121-specific decode-kernel workaround may not be needed. Anything older than Hopper hard-fails behind `--force`.

## What's in the checkpoint

| Component | Format | bpw |
|---|---|---|
| Routed experts: gate / up | IQ2_XXS | ~2.06 |
| Routed experts: down | Q2_K | ~2.62 |
| Dense linears + attention | FP8 E4M3, block-128, UE8M0 scales | 8 |
| Embeddings, lm_head, norms, scalars | BF16 | 16 |

Total on-disk: ~85 GiB across 17 safetensors shards. Conversion script: [`scripts/build-ds4-2bit-checkpoint.py`](scripts/build-ds4-2bit-checkpoint.py) (deterministic — re-running on the same GGUF produces a byte-identical safetensors set).

## Validation

Layer-by-layer hidden-state cosine similarity vs the `antirez/ds4` reference, on `"The capital of France is"`:

| Layer | input cos | output cos |
|---|---|---|
| 0 | 0.9991 | 0.9975 |
| 21 | 0.9924 | 0.9909 |
| 42 | 0.9590 | 0.9304 |

Mean input cosine across all 43 layers: **0.9875**. First five generated tokens match the reference exactly:

```
ds4 ref:   ' We are asked: "...'  → tokens [2581, 477, 4869, 28, 582, ...]
this repo: ' We are asked: "The capital of France is'
```

The `scripts/smoke-test.sh` helper checks just the first token by default (fastest signal of a working install) and the full 5-token prefix in `--strict` mode.

## Two correctness-critical knobs

These are the load-bearing pieces of the install. Both are baked into the installer's `docker run` automatically; flagged here for anyone wiring up their own pipeline:

- **`--quantization deepseek_v4_hybrid_iq2`** — registered by this repo's plugin (`Ds4HybridIq2Config` + `Iq2XxsQ2KFusedMoEMethod`). vLLM auto-detects it from the checkpoint's `quant_method` field.
- **`VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0`** — required on SM121. The default Triton compressed-decode kernel (`matmul_sparse_mla_attention_with_sink`) produces wrong output on consumer Blackwell for layers with `compress_ratio≥4`. This env flag switches to the working `fp8ds_global_paged_sparse_mla_attention_with_sink_multihead` path. Without it, the model emits one correct token then degenerates.

The full bring-up story — how those two bugs were located via layer-by-layer hidden-state bisection against the `antirez/ds4` reference, and what didn't work — is in [`docs/DSV4_FLASH_2BIT_SPARK_REPORT.md`](docs/DSV4_FLASH_2BIT_SPARK_REPORT.md).

## Repo layout

```
install.sh                          One-shot installer (curl | bash)
scripts/
  smoke-test.sh                       First-token check vs ds4 reference
  build-ds4-2bit-checkpoint.py        GGUF → safetensors converter (deterministic)
  upload-checkpoint.sh                Publish converted checkpoint to HF
  sanity_check_checkpoint.py          Static integrity check on the safetensors
  patch_deepgemm_sm12.py              Source patch for DeepGEMM SM12x dispatch
  spark_stage_a_kernel_check.py       Synthetic-data Triton kernel validation
src/ds4_hybrid_quant/
  vllm_patches/                       Plugin registration + FusedMoE method
  load_completion.py                  Post-load weight remap (norms, fused Q/KV, etc.)
  triton_kernels/                     IQ2_XXS / Q2_K / Q8_K Triton kernels
  block_layouts.py, dequant.py, ...   Reference dequant + helpers
eugr_mod/mods/ds4-2bit-deepseek-v4-flash/
  run-on-lmxxf.sh                     In-container bootstrap for the lmxxf base image
  run.sh                              Standalone-image bootstrap (older path)
docs/
  DSV4_FLASH_2BIT_SPARK_REPORT.md     Public bring-up writeup
tests/                                Triton kernel tests + reference C harness
```

## Common operations

```bash
# Re-run the smoke test against an already-running container
~/ds4-spark-vllm/scripts/smoke-test.sh --port 8000 --strict --verbose

# Tail serve logs
tail -f ~/logs/serve.log
docker logs -f vllm-ds4

# Verify the SM121 decode-kernel workaround is actually set
docker inspect vllm-ds4 | grep VLLM_TRITON

# Stop and remove the container (keeps the model dir)
~/ds4-spark-vllm/install.sh --uninstall

# Re-install / upgrade (idempotent — skips already-downloaded shards)
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-spark-vllm/main/install.sh | bash
```

## How it fits together

This repo is one of two pieces:

| Piece | Where |
|---|---|
| Triton kernels, plugin, post-load completion, converter, installer | this repo |
| Base vLLM build with SM12x dispatch + DeepGEMM | [`lmxxf/vllm-deepseek-v4-dgx-spark`](https://hub.docker.com/r/lmxxf/vllm-deepseek-v4-dgx-spark) (Docker Hub) |

The installer pulls the lmxxf image, then `pip install -e .` this repo into the container. That overlay registers `--quantization deepseek_v4_hybrid_iq2` and patches `vllm/model_executor/models/deepseek_v4.py:load_weights` to handle our 2-bit expert tensor names — see [`eugr_mod/mods/ds4-2bit-deepseek-v4-flash/run-on-lmxxf.sh`](eugr_mod/mods/ds4-2bit-deepseek-v4-flash/run-on-lmxxf.sh).

The package follows the asymmetric-quantization pattern set by [`rmstxrx/vllm-hybrid-quant`](https://github.com/rmstxrx/vllm-hybrid-quant) (GPTQ-INT4 + FP8 for Qwen).

## License

MIT. Both this repo and the redistributed checkpoint match upstream [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).

## Acknowledgements

- [`antirez/ds4`](https://github.com/antirez/ds4) — the C+Metal reference implementation. The 2-bit recipe, the layer-by-layer bisection methodology, and the canonical validation prompt are all lifted from there.
- [`deepseek-ai`](https://huggingface.co/deepseek-ai) — DeepSeek-V4-Flash upstream weights and architecture.
- [`lmxxf`](https://hub.docker.com/r/lmxxf/vllm-deepseek-v4-dgx-spark) — community-maintained vLLM build with SM12x dispatch, the base image this overlay sits on top of.
- [`sgl-project/DeepSeek-V4-Flash-FP8`](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) — FP8 dense weights used for the non-MoE path of the converter.
