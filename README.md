# vllm-hybrid-quant-ds4

Hybrid 2-bit + FP8 quantization for DeepSeek V4 Flash on a single DGX Spark, served by vLLM.

Recipe (lifted from [antirez/ds4](https://github.com/antirez/ds4)):

- Routed MoE expert **gate/up** projections → **IQ2_XXS** (~2.06 bpw)
- Routed MoE expert **down** projections → **Q2_K** (~2.62 bpw)
- Attention, shared experts, projections, embeddings → **FP8 E4M3 block-128** (sourced from [`sgl-project/DeepSeek-V4-Flash-FP8`](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8))

Total checkpoint ≈ 86 GB → fits the 128 GB Spark unified-memory budget with room for KV cache.

This repo follows the pattern set by [rmstxrx/vllm-hybrid-quant](https://github.com/rmstxrx/vllm-hybrid-quant) (GPTQ-INT4 + FP8). Two pieces:

| Component | Where |
|---|---|
| Triton kernels + checkpoint builder | this repo |
| vLLM dispatch patch | a separate vLLM fork (registered as `deepseek_v4_hybrid_iq2`) |

## Status

Pre-alpha. Bring-up in progress.
