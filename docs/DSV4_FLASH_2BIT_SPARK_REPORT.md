# DeepSeek-V4-Flash 2-bit Hybrid on DGX Spark via vLLM — Setup & Bring-up Report

**Date:** 2026-05-10
**Hardware:** NVIDIA DGX Spark (`gn100-7710.local`, GB10 / SM121, 128 GB unified memory)
**Status:** Working end-to-end. Coherent multi-token generation verified against the ds4 reference implementation.

---

## 1. Goal

Run [antirez/ds4](https://github.com/antirez/ds4)-style 2-bit DeepSeek-V4-Flash inference on a single DGX Spark via vLLM, using a hybrid quantization scheme that fits the model in 128 GB of unified memory while keeping accuracy close to the original DeepSeek-V4 release.

DSv4-Flash at full FP8 is roughly 480 GB. At MXFP4 it's still too large for a single Spark. The 2-bit-routed-experts scheme (described below) brings it under 100 GB, leaving headroom for KV cache, workspace, and the rest of the OS.

## 2. Architecture and quantization recipe

DSv4-Flash specifics:

- 43 transformer layers
- MLA attention with a separate compressor pathway
- Lightning Indexer for sparse SWA selection (ratio-4 layers)
- HyperConnection (HC) residual stream with 4 channels
- Multi-Token Prediction (MTP) head (one extra block-equivalent of params)
- 256 routed experts + 1 shared expert per layer
- `hidden=4096`, attention heads grouped, `n_hc=4`

The hybrid recipe (lifted from antirez/ds4):

| Component | Quantization | bpw |
|---|---|---|
| Routed experts: gate / up | IQ2_XXS | ~2.06 |
| Routed experts: down | Q2_K | ~2.62 |
| Dense layers + attention linears | FP8 E4M3 (block-128, UE8M0 scales) | 8 |
| Embeddings, lm_head, norms, scalars | source dtype (BF16) | 16 |

Routed-expert weights in our checkpoint use the GGUF block-quantization layouts directly (gate+up fused per expert: `(E, 2*intermediate, n_blocks_in, 64)` for IQ2_XXS qs; down-projection per-expert with separate scales, dmin, d for Q2_K).

## 3. Stack

- **Base image:** `lmxxf/vllm-deepseek-v4-dgx-spark:latest` — community-maintained vLLM build with SM12x dispatch for HC, fp8_mqa_logits, fp8_paged_mqa_logits, plus a working DeepGEMM build for SM121.
- **Our overlay:** Repository `Entrpi/ds4-spark-vllm` (this repo). Installed editable into the lmxxf container via `pip install -e`. Provides:
  - `Ds4HybridIq2Config` — registers the `deepseek_v4_hybrid_iq2` quantization method with vLLM. Embeds `DeepseekV4FP8Config` for non-MoE FP8 layers (carries the `is_scale_e8m0` property required for SM12x scale-format interpretation).
  - `Iq2XxsQ2KFusedMoEMethod` — vLLM `FusedMoEMethodBase` subclass implementing the routed-MoE forward via Triton kernels (`quantize_q8_K_triton`, `iq2_xxs_pair_dot_triton`, `q2_K_accum_dot_triton`).
  - `load_completion.complete_load` — post-`load_weights` step that direct-loads tensors stock vLLM's pipeline can't match (ds4-style flat names, fused Q/KV, fused gate/up shared-expert) and default-inits anything genuinely absent from disk.
  - In-container patches to `vllm/model_executor/models/deepseek_v4.py`'s `load_weights` to gracefully handle our 2-bit expert names.

- **Model checkpoint:** `/home/ent/models/deepseek-v4-flash-ds4-q2/` — converted from antirez's GGUF (`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`, ~85 GB) into vLLM-loadable safetensors (~85 GB across 17 shards). Mixed naming on disk: most tensors use the ds4 flat naming `layers.K.X`, while our 2-bit MoE weights use `model.layers.K.mlp.experts.X`.

## 4. The bring-up journey, condensed

Earlier sessions had brought the project to "model loads, no NaNs, but output is gibberish." This session closed the correctness gap.

### 4.1 First-pass diagnostics (didn't move the needle)

- Confirmed our quant method is properly registered, all 43 FusedMoE instances are constructed, and our `apply()` fires for every layer in real inference (not just warmup).
- Verified our routed-MoE output reaches the residual stream by toggling between NORMAL / NO-OP (return zeros) / PASSTHROUGH (return x) modes via `/logs/ds4_moe_*` file flags. All three produced different gibberish — confirming our compute *contributes*, but not localizing the bug.
- Switched from stock `Fp8Config` to `DeepseekV4FP8Config` (the SM12x-aware subclass with `is_scale_e8m0`) for non-MoE FP8 layers. Output bit-identical → not the bug.
- Set `expert_dtype="fp8"` in the model config (it had been defaulting to "fp4"). Output bit-identical → not the bug.

The triple-test was inconclusive because DSv4 needs a functional 256-expert routed MoE to produce coherent text — disabling it just produces a different kind of gibberish.

### 4.2 Reference-model setup for comparison

Compiled `antirez/ds4` on Spark CPU (`-DDS4_NO_METAL`, `CFLAGS="-D_GNU_SOURCE"`), patched `ds4_alloc_guard_check` to warn-not-die so CPU decode actually completes, and confirmed it produces coherent text from the same GGUF used for our conversion:

```
prompt: "The capital of France is"
ds4 internal prefill: [0, 3476, 477, 260, 11502, 22896, 128803, 671,
                       6102, 294, 8760, 344, 128804, 128821]   (14 tokens)
ds4 generated:        [2581, 477, 4869, 28, 582]
                    = ' We are asked: "'
```

We use this same 14-token list as the input to vLLM (sent directly via `/v1/completions` as a token ID array, bypassing tokenization) so the comparison is apples-to-apples.

### 4.3 Layer-by-layer hidden-state bisection

The breakthrough was committing to a real bisection rather than guessing.

**Instrumentation:**

- *ds4 side*: patched `layer_forward_raw_swa_one` in `ds4.c` to dump `scratch->ffn_norm` (post-norm MoE input) and `scratch->ffn_moe` (routed-MoE output) per layer per position to `${DS4_HDUMP_DIR}/ds4_{in,out}_L{K}_P{pos}.f32` — fp32 raw bytes, 4096 floats per file. Run with `DS4_NO_BATCHED_ATTN=1` to force the per-token path that matches our hook.
- *vLLM side*: `Iq2XxsQ2KFusedMoEMethod.apply()` dumps `x[T-1, :]` and `out[T-1, :]` to `/logs/ds4_hdump/vllm_{in,out}_L{K}_C{n}.f32` when `/logs/ds4_dump_arm` exists. C0 = first armed call (prefill), C1+ = subsequent decode steps.
- Layer index resolution: a class-level `_ds4_layer_seq` registry populated in `create_weights` — `layer.prefix` and `layer.layer_idx` are both unset on FusedMoE in our build, but `create_weights` is called sequentially in layer order, so the registration sequence corresponds to layer index 0..42.

**First compare (prefill, position 13 = last prefill token):**

| L | in_cos | in_norm ds4 / vLLM |
|---|---|---|
| 0 | **0.2534** | 15.06 / **64.00** ← √4096 |
| 1 | 0.0027 | 16.40 / 64.00 |
| 2 | 0.0513 | 20.36 / 64.00 |

vLLM's input to the layer-0 MoE was bit-for-bit consistent with **unweighted** RMS-norm output (`‖x‖ ≈ √hidden`). That's the signature of identity-norm-weight: `RMSNorm(x) * 1.0`. The trained `attn_norm.weight` and `ffn_norm.weight` weren't being loaded.

### 4.4 Bug 1 — norm-loading

Root cause in `src/ds4_hybrid_quant/load_completion.py`: the original phase order was

1. **Phase 1**: walk unloaded params, default-init anything matching `*_norm.weight`, `*_scale_inv`, `*_scale`, `*.bias`.
2. **Phase 2**: remap rules direct-load remaining params from disk.

The trained norm weights *were* on disk at `layers.K.attn_norm.weight`, `layers.K.ffn_norm.weight`, etc., but Phase 1 always claimed them first via `_default_for("*_norm.weight") = 1.0`, putting them in `loaded_params` before Phase 2 had a chance.

**Fix:**

- Reordered: Phase 1 = direct-load via remap rules; Phase 2 = default-init what's still unloaded (truly-not-on-disk params like FP8 `weight_scale_inv` for layers without scales, or biases that don't exist).
- Added explicit remap rules for `attn_norm.weight`, `ffn_norm.weight`, `attn.kv_norm.weight`, `attn.q_norm.weight`. Each is identity (live name suffix matches disk name suffix) so the existing `_get_tensor(prefix + src_suffix)` machinery picks them up directly.

**Verification after Bug 1 fix:**

| L | in_cos | in_norm ds4 / vLLM |
|---|---|---|
| 0 | **0.9991** | 15.06 / **15.05** |
| 1 | 0.9978 | 16.40 / 16.40 |
| 42 | 0.9590 | 35.12 / 35.07 |

Layer-by-layer prefill cos sim ≥ 0.95 throughout 43 layers. **First emitted token is now `'We'`** — exactly matches the ds4 reference token 2581.

### 4.5 The decode-time bug surfaces

`max_tokens=10` on the original prompt produced `'We當前而那アメリ而非而非而非...'` — first token correct, then degenerates into a repeating-token attractor.

To rule out drift accumulation: ran an iterative-prefill experiment (each new token is generated by a fresh prefill of the accumulating token list, `max_tokens=1` per call). Result:

```
step 1: token_id=2581 text='We'        ✓
step 2: token_id=477  text=' are'      ✓
step 3: token_id=4869 text=' asked'    ✓
step 4: token_id=28   text=':'         ✓
step 5: token_id=582  text=' "'        ✓
[20 steps total — fully coherent: 'We are asked: "The capital of
 France is". This is a simple factual question. The capital'   ]
```

5/5 match against ds4 reference, then continues coherently. **Prefill is correct; decode is the bug.**

### 4.6 Bug 2 — decode kernel on SM12x

Extended the bisection to the decode path: ran one curl with `max_tokens=4` (giving prefill + 3 decodes = `C0..C3` dumps per layer) and another with a 15-token prompt + `max_tokens=1` (giving a clean prefill at position 14 — the equivalent state to the first decode at position 14, no decode kernel involved). vllm-internal compare (no ds4 reference needed):

| L | in_cos (decode-C1 vs prefill-C4) |
|---|---|
| 0 | **1.00000** ← bit-identical |
| 1 | 0.99943 |
| 2 | **0.91123** ← first divergence |
| 3 | 0.78790 |
| 4 | 0.56343 |

Per-layer architecture audit (from the safetensors index):

```
L 0: compressor=False indexer=False sink=True    ← compress_ratio=1
L 1: compressor=False indexer=False sink=True    ← compress_ratio=1
L 2: compressor=True  indexer=True  sink=True    ← compress_ratio=4  *** bug starts here
L 3: compressor=True  indexer=False sink=True    ← compress_ratio=128
L 4: compressor=True  indexer=True  sink=True    ← compress_ratio=4
...
```

The first divergence is exactly the first layer with `compress_ratio≥4`. SWA-only decode (layers 0, 1) works; compressed-decode (layers 2+) is broken.

Inside `_forward_sparse_mla_compressed_decode_triton` (`vllm/model_executor/layers/deepseek_v4_attention.py`), the path for SM12x with `triton_sparse_mla_matmul_decode_enabled()` calls:

```python
dequantize_combined_sparse_mla_decode_kv(combined_kv, compressed_k_cache, ...)
build_combined_sparse_mla_decode_valid_mask(valid_tokens, ...)
matmul_sparse_mla_attention_with_sink(q=q, kv=combined_kv, ...)
```

The `matmul_sparse_mla_attention_with_sink` path produces wrong output on SM12x. The fallback path (used when matmul-decode is disabled) is `fp8ds_global_paged_sparse_mla_attention_with_sink_multihead`, which works correctly.

**Fix (workaround):** set `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` in the container env. This switches the compressed-decode path to the working fallback while leaving everything else (including SWA-only decode) untouched.

This is a workaround — the underlying bug is in either `matmul_sparse_mla_attention_with_sink` or `dequantize_combined_sparse_mla_decode_kv`. Worth filing upstream once we have a minimal reproducer.

## 5. Verification

After both fixes, with `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0`:

| Test | Output |
|---|---|
| `prompt=14-token list, max_tokens=10` | `'We are asked: "The capital of France is'` (matches ds4 ref) |
| `prompt="The capital of France is", max_tokens=30` | `'We are asked: "The capital of France is". This is a simple factual question. The capital of France is Paris. So the answer should be'` |
| `prompt="What is 12 times 7?", max_tokens=40` | Computes 12 × 7 = 84 with reasoning |
| `prompt="def fibonacci(n):\n    if n <"` | Sensible Python: returns a working-shape recursive Fibonacci stub |

Single-call decode now matches iterative-prefill, which matches ds4 reference. End-to-end correctness achieved.

## 6. Working configuration (verbatim)

```bash
# On Spark, with modded image already built and checkpoint already converted:
docker run -d --gpus all --name vllm-ds4 --network host \
  -v /home/ent/models:/models -v /home/ent/ds4-spark-vllm:/work \
  -v /home/ent/logs:/logs -v /home/ent/extras:/extras \
  -e DG_LOCAL=/extras/DeepGEMM \
  -e LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so \
  -e VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 \
  --entrypoint bash lmxxf/vllm-deepseek-v4-dgx-spark:latest \
  -c "bash /work/eugr_mod/mods/ds4-2bit-deepseek-v4-flash/run-on-lmxxf.sh > /logs/serve-mod.log 2>&1 && \
      vllm serve /models/deepseek-v4-flash-ds4-q2 \
        --served-model-name dsv4 --quantization deepseek_v4_hybrid_iq2 \
        --port 8000 --host 0.0.0.0 \
        --max-model-len 4096 --gpu-memory-utilization 0.85 \
        --kv-cache-dtype fp8 --attention-backend FLASHINFER \
        --enforce-eager 2>&1 | tee /logs/serve.log"
```

Critical knobs:

- `lmxxf/vllm-deepseek-v4-dgx-spark:latest` is the right base image. Earlier in the session I mistakenly tried `vllm-ds4-flash:latest` (an older locally-tagged variant) and got a much newer vLLM that requires DeepGEMM the lmxxf base doesn't ship — both broke. The lmxxf image runs `v0.1.dev1` of vLLM which is what the rest of this stack is calibrated against.
- `--kv-cache-dtype fp8` is **required** — DSv4 in lmxxf has an `assert kv_cache_dtype.startswith("fp8")`. `auto` is rejected.
- `--attention-backend FLASHINFER` is what we use. `FLASHINFER_MLA_SPARSE` requires DeepGEMM that lmxxf doesn't have on this path.
- `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` is required for correctness on SM121 — see Bug 2.
- `--enforce-eager` keeps things simple (no torch.compile / CUDA graphs). Performance optimization is a separate track.

## 7. Performance

Not measured in this session. Quick observations during testing:

- Container startup: ~25-35 minutes (model load + DeepGEMM JIT cache warmup + flashinfer autotuning).
- Generation speed: not benchmarked. The MoE path is per-token Python looping over selected experts — correct but slow. Replacing this with a fused grouped-GEMM kernel is the obvious next perf step.

## 8. Known limitations / next steps

1. **MoE grouped-GEMM kernel.** `Iq2XxsQ2KFusedMoEMethod.apply()` currently does Python loops over `(t, k)` pairs, calling `iq2_xxs_pair_dot_triton` and `q2_K_accum_dot_triton` once per (token, expert). A single fused launch per layer would be 10-100× faster. The kernel math is straightforward; the SM121 shmem budget is ~99 KB/SM (consumer Blackwell, **not** the 228 KB on datacenter Blackwell), which constrains tile sizes.

2. **Upstream fix for compressed-decode triton path.** `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` is a workaround. Filing a minimal reproducer for the broken `matmul_sparse_mla_attention_with_sink` (or `dequantize_combined_sparse_mla_decode_kv`) on SM12x would be valuable.

3. **Wider correctness checks.** We've verified ~30-token coherent generation against ds4 on one prompt and three other prompts qualitatively. Logit-parity tests on antirez's standard test vectors would give a stronger correctness signal across the full output distribution.

4. **MTP head.** The container's second `complete_load` instance (1629 params, 190 stock-loaded, 265 unloaded with no remap rule) appears to be the MTP draft model. We aren't using speculative decoding (no `--num-speculative-tokens`), so its broken state doesn't affect serving — but if MTP becomes interesting later, it'll need its own naming-convention remap.

5. **Long-context.** `--max-model-len 4096` is what we ran. DSv4-Flash is designed for much longer contexts; the indexer/compressor stack is exactly the apparatus for scaling. Worth re-validating once the decode kernel issue is fixed upstream rather than worked around.

## 9. Reference: useful telemetry while debugging

Hot-toggle file flags read by `apply()`:

| File | Effect |
|---|---|
| `/logs/ds4_moe_noop` | `apply()` returns zeros — observe what the model does without routed contribution. |
| `/logs/ds4_moe_passthrough` | `apply()` returns `x.clone()` — pass input through unchanged. |
| `/logs/ds4_route_arm` | Per-call routing entropy and top-1 expert IDs printed for `T<=16` calls. Touch before curl, remove after. |
| `/logs/ds4_dump_arm` | Per-layer x-in / out hidden-state dump to `/logs/ds4_hdump/vllm_{in,out}_L{K}_C{n}.f32`. |

ds4 side env (`DS4_HDUMP_DIR=/path`) plus `DS4_NO_BATCHED_ATTN=1` makes ds4 dump the matching files.

Compare scripts: `local/scripts/ds4_compare.py` (prefill last-pos compare), `local/scripts/ds4_compare_decode.py` (prefill + N decode steps).

---

**Bottom line:** DSv4-Flash 2-bit hybrid is now serving correctly on a single Spark via vLLM. Two concrete bugs were the entire correctness gap; both are fixed, one as a clean source change and one as a single-env-var workaround.
