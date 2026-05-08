"""Hybrid checkpoint builder: antirez GGUF + sgl-project FP8 -> safetensors.

Produces a checkpoint that vLLM can load with
``--quantization deepseek_v4_hybrid_iq2``. Memory-efficient (mmap-based) so
the 86 GB GGUF and the (selectively-downloaded) FP8 dense shards never live
fully in RAM.

Output naming matches ``Iq2XxsQ2KFusedMoEMethod.create_weights``:

    model.layers.{N}.mlp.experts.w13_iq2xxs_qs   (E, 2*I, in_blk, 64)  uint8
    model.layers.{N}.mlp.experts.w13_iq2xxs_d    (E, 2*I, in_blk)      f16
    model.layers.{N}.mlp.experts.w2_q2k_qs       (E, H,   int_blk, 64) uint8
    model.layers.{N}.mlp.experts.w2_q2k_scales   (E, H,   int_blk, 16) uint8
    model.layers.{N}.mlp.experts.w2_q2k_d        (E, H,   int_blk)     f16
    model.layers.{N}.mlp.experts.w2_q2k_dmin     (E, H,   int_blk)     f16

Plus all dense / shared-expert / attention tensors copied from
sgl-project/DeepSeek-V4-Flash-FP8 (UE8M0 scales, block-128 weights).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .lookup_tables import QK_K

logger = logging.getLogger(__name__)


# GGUF tensor naming follows ``blk.{layer}.<role>.weight`` for DeepSeek-style
# models. Routed-expert weights typically appear as fused per-layer tensors:
#     blk.{N}.ffn_gate_exps.weight   (E, intermediate, hidden)  IQ2_XXS
#     blk.{N}.ffn_up_exps.weight     (E, intermediate, hidden)  IQ2_XXS
#     blk.{N}.ffn_down_exps.weight   (E, hidden,       intermediate)  Q2_K
GGUF_GATE_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")
GGUF_UP_RE = re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$")
GGUF_DOWN_RE = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")


# ---------------------------------------------------------------------------
# Block byte-byte repackers (GGUF layout -> our safetensors sub-tensors)
# ---------------------------------------------------------------------------


def repack_iq2_xxs_rows(raw: bytes, n_rows: int, in_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert GGUF IQ2_XXS row-major bytes into (qs, d) sub-tensors.

    GGUF layout per row of length ``in_dim``: ``in_dim/256`` blocks of 66
    bytes each (2 bytes f16 d + 64 bytes qs).

    Returns:
        qs: uint8 (n_rows, n_blocks, 64)
        d:  float16 (n_rows, n_blocks)
    """
    if in_dim % QK_K != 0:
        raise ValueError(f"in_dim {in_dim} is not a multiple of {QK_K}")
    n_blocks = in_dim // QK_K
    expected = n_rows * n_blocks * 66
    if len(raw) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(raw)}")

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, n_blocks, 66)
    d = arr[:, :, :2].view(np.float16).reshape(n_rows, n_blocks).copy()
    qs = arr[:, :, 2:].copy()
    return qs, d


def repack_q2_K_rows(
    raw: bytes, n_rows: int, in_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert GGUF Q2_K row-major bytes into (qs, scales, d, dmin) sub-tensors.

    GGUF block layout: 16 scale bytes + 64 qs bytes + 2 byte f16 d + 2 byte
    f16 dmin = 84 bytes per block.

    Returns:
        qs:     uint8 (n_rows, n_blocks, 64)
        scales: uint8 (n_rows, n_blocks, 16)
        d:      float16 (n_rows, n_blocks)
        dmin:   float16 (n_rows, n_blocks)
    """
    if in_dim % QK_K != 0:
        raise ValueError(f"in_dim {in_dim} is not a multiple of {QK_K}")
    n_blocks = in_dim // QK_K
    expected = n_rows * n_blocks * 84
    if len(raw) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(raw)}")

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, n_blocks, 84)
    scales = arr[:, :, :16].copy()
    qs = arr[:, :, 16:80].copy()
    d_dmin = arr[:, :, 80:84].view(np.float16).reshape(n_rows, n_blocks, 2)
    d = d_dmin[:, :, 0].copy()
    dmin = d_dmin[:, :, 1].copy()
    return qs, scales, d, dmin


# ---------------------------------------------------------------------------
# GGUF file reader (minimal, mmap-based)
# ---------------------------------------------------------------------------


@dataclass
class GgufTensor:
    name: str
    dtype: int           # GGUF type code
    shape: tuple[int, ...]
    offset: int          # absolute file offset to the start of tensor data
    nbytes: int          # raw byte count


# GGUF type codes we care about. Full list in
# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
_GGUF_TYPE_Q2_K = 10
_GGUF_TYPE_IQ2_XXS = 16
_GGUF_TYPE_Q8_0 = 8
_GGUF_TYPE_F16 = 1
_GGUF_TYPE_F32 = 0
_GGUF_TYPE_BF16 = 30


def open_gguf(path: str | Path) -> tuple[Any, list[GgufTensor], dict[str, Any]]:
    """Open a GGUF file via the ``gguf`` Python package.

    Returns ``(reader, tensors, metadata)``. The reader is a ``GGUFReader``
    instance whose mmap stays alive as long as the reference is held; do
    not close it while iterating tensors.
    """
    import gguf  # type: ignore[import-not-found]

    reader = gguf.GGUFReader(str(path), "r")

    tensors: list[GgufTensor] = []
    for t in reader.tensors:
        # ``t.tensor_type`` is an enum; get the int value for portable
        # comparison.
        tensors.append(
            GgufTensor(
                name=t.name,
                dtype=int(t.tensor_type),
                shape=tuple(int(d) for d in t.shape),
                # ``data_offset`` is the relative offset from the file's
                # tensor_data start; the reader exposes raw bytes via
                # ``t.data`` (a numpy view into the mmap).
                offset=int(t.data_offset),
                nbytes=int(t.n_bytes),
            )
        )

    metadata: dict[str, Any] = {}
    for k, f in reader.fields.items():
        try:
            vals = [f.parts[i] for i in f.data]
            if len(vals) == 1 and vals[0].size == 1:
                metadata[k] = vals[0].item()
            elif len(vals) == 1:
                metadata[k] = vals[0]
            else:
                metadata[k] = vals
        except Exception:
            pass
    return reader, tensors, metadata


def gguf_tensor_bytes(reader: Any, name: str) -> bytes:
    """Get raw block bytes for a named GGUF tensor."""
    for t in reader.tensors:
        if t.name == name:
            arr = np.asarray(t.data)
            return arr.tobytes()
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Layer iteration
# ---------------------------------------------------------------------------


def iter_layer_expert_tensors(
    gguf_tensors: Iterable[GgufTensor],
) -> dict[int, dict[str, GgufTensor]]:
    """Group routed-expert tensors by layer index."""
    by_layer: dict[int, dict[str, GgufTensor]] = {}
    for t in gguf_tensors:
        for role, regex in (
            ("gate", GGUF_GATE_RE),
            ("up", GGUF_UP_RE),
            ("down", GGUF_DOWN_RE),
        ):
            m = regex.match(t.name)
            if m:
                layer = int(m.group(1))
                by_layer.setdefault(layer, {})[role] = t
                break
    return by_layer


def expected_role_dtype() -> dict[str, int]:
    return {
        "gate": _GGUF_TYPE_IQ2_XXS,
        "up": _GGUF_TYPE_IQ2_XXS,
        "down": _GGUF_TYPE_Q2_K,
    }


def hf_tensor_name_for(layer: int, sub: str) -> str:
    """HF-style name for one of our six per-layer expert sub-tensors.

    ``sub`` is one of: w13_iq2xxs_qs, w13_iq2xxs_d, w2_q2k_qs,
    w2_q2k_scales, w2_q2k_d, w2_q2k_dmin.
    """
    return f"model.layers.{layer}.mlp.experts.{sub}"


# ---------------------------------------------------------------------------
# Repack one layer's experts into sub-tensors
# ---------------------------------------------------------------------------


def build_layer_expert_subtensors(
    layer_tensors: dict[str, GgufTensor],
    *,
    reader: Any,
    n_experts: int,
    hidden: int,
    intermediate: int,
) -> dict[str, np.ndarray]:
    """Return the six sub-tensors for one layer's routed experts."""
    expected = expected_role_dtype()
    for role in ("gate", "up", "down"):
        if role not in layer_tensors:
            raise KeyError(f"missing GGUF tensor for role {role!r}")
        if layer_tensors[role].dtype != expected[role]:
            raise ValueError(
                f"role {role}: expected GGUF dtype {expected[role]}, "
                f"got {layer_tensors[role].dtype}"
            )

    gate_raw = gguf_tensor_bytes(reader, layer_tensors["gate"].name)
    up_raw = gguf_tensor_bytes(reader, layer_tensors["up"].name)
    down_raw = gguf_tensor_bytes(reader, layer_tensors["down"].name)

    # Gate / up: (E, intermediate, hidden) -> repack each expert independently,
    # then stack on dim 0.
    gate_per_expert_bytes = len(gate_raw) // n_experts
    up_per_expert_bytes = len(up_raw) // n_experts
    if gate_per_expert_bytes * n_experts != len(gate_raw):
        raise ValueError("gate raw bytes not divisible by n_experts")
    if up_per_expert_bytes * n_experts != len(up_raw):
        raise ValueError("up raw bytes not divisible by n_experts")

    n_blocks_in = hidden // QK_K

    gate_qs_list = []
    gate_d_list = []
    up_qs_list = []
    up_d_list = []
    for e in range(n_experts):
        g_qs, g_d = repack_iq2_xxs_rows(
            gate_raw[e * gate_per_expert_bytes:(e + 1) * gate_per_expert_bytes],
            n_rows=intermediate, in_dim=hidden,
        )
        u_qs, u_d = repack_iq2_xxs_rows(
            up_raw[e * up_per_expert_bytes:(e + 1) * up_per_expert_bytes],
            n_rows=intermediate, in_dim=hidden,
        )
        gate_qs_list.append(g_qs)
        gate_d_list.append(g_d)
        up_qs_list.append(u_qs)
        up_d_list.append(u_d)

    gate_qs = np.stack(gate_qs_list)  # (E, intermediate, n_blocks_in, 64)
    gate_d = np.stack(gate_d_list)
    up_qs = np.stack(up_qs_list)
    up_d = np.stack(up_d_list)

    # Pack [gate | up] along the row axis so dim 1 == 2 * intermediate.
    w13_qs = np.concatenate([gate_qs, up_qs], axis=1)  # (E, 2I, n_blocks_in, 64)
    w13_d = np.concatenate([gate_d, up_d], axis=1)

    # Down: (E, hidden, intermediate) at Q2_K.
    n_blocks_int = intermediate // QK_K
    down_per_expert_bytes = len(down_raw) // n_experts
    if down_per_expert_bytes * n_experts != len(down_raw):
        raise ValueError("down raw bytes not divisible by n_experts")

    d_qs_list, d_sc_list, d_d_list, d_dmin_list = [], [], [], []
    for e in range(n_experts):
        qs, sc, d, dmin = repack_q2_K_rows(
            down_raw[e * down_per_expert_bytes:(e + 1) * down_per_expert_bytes],
            n_rows=hidden, in_dim=intermediate,
        )
        d_qs_list.append(qs)
        d_sc_list.append(sc)
        d_d_list.append(d)
        d_dmin_list.append(dmin)

    return {
        "w13_iq2xxs_qs": w13_qs,
        "w13_iq2xxs_d": w13_d,
        "w2_q2k_qs": np.stack(d_qs_list),
        "w2_q2k_scales": np.stack(d_sc_list),
        "w2_q2k_d": np.stack(d_d_list),
        "w2_q2k_dmin": np.stack(d_dmin_list),
    }


# ---------------------------------------------------------------------------
# FP8 dense source (sgl-project)
# ---------------------------------------------------------------------------


def get_fp8_dense_manifest(repo: str) -> dict[str, str]:
    """Read the FP8 source's safetensors index and return non-expert weight map."""
    from huggingface_hub import hf_hub_download
    idx_path = hf_hub_download(repo, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)
    return {k: v for k, v in idx["weight_map"].items() if ".experts." not in k}


def write_quantization_config(out_dir: Path, *, scale_fmt: str = "ue8m0") -> None:
    """Patch the model's config.json with our quantization_config block."""
    cfg_path = out_dir / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "quant_method": "deepseek_v4_hybrid_iq2",
        "scale_fmt": scale_fmt,
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
        "moe_experts": "IQ2_XXS gate/up + Q2_K down (ds4 recipe)",
        "dense_layers": "FP8 E4M3 block-128 with UE8M0 scales (sgl-project)",
        "source_gguf": "antirez/deepseek-v4-gguf",
        "source_fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
        "converter": "ds4_hybrid_quant.builder",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
