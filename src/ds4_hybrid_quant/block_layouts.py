"""GGUF block layouts used by the ds4 recipe.

Layouts and constants match antirez/ds4 (which lifts them from llama.cpp/ggml).
We store IQ2_XXS and Q2_K blocks by carving GGUF block bytes into the standard
sub-tensors so a safetensors checkpoint can hold them in vanilla uint8/float16
dtypes without needing a custom safetensors codec.
"""

from dataclasses import dataclass
import numpy as np

from .lookup_tables import QK_K


# IQ2_XXS: 256 quants per block, 2.0625 bits per weight.
# Block bytes:
#   d  : float16     (2 bytes)
#   qs : uint16[32]  (64 bytes)  -> we store as uint8[64] for portability
# Total: 66 bytes/block.
IQ2_XXS_BLOCK_BYTES = 66
IQ2_XXS_QS_BYTES = 64
IQ2_XXS_N_SUBBLOCKS = QK_K // 32  # 8 sub-blocks of 32 quants per block


# Q2_K: 256 quants per block, 2.625 bits per weight.
# Block bytes:
#   scales : uint8[16] (16 bytes, low nibble = scale, high nibble = min)
#   qs     : uint8[64] (64 bytes, 2-bit packed: 4 quants per byte)
#   d      : float16   (2 bytes)
#   dmin   : float16   (2 bytes)
# Total: 84 bytes/block.
Q2_K_BLOCK_BYTES = 84
Q2_K_SCALES_BYTES = 16
Q2_K_QS_BYTES = 64


# Q8_K: 256 quants per block. Activation quantization format.
# Layout:
#   d     : float32   (4 bytes)
#   qs    : int8[256]
#   bsums : int16[16]  (per-16-quant sums, used by Q2_K dot product fast path)
# Total: 4 + 256 + 32 = 292 bytes/block.
Q8_K_BLOCK_BYTES = 292
Q8_K_QS_LEN = 256
Q8_K_BSUMS_LEN = 16


@dataclass
class IQ2XXSTensors:
    """Packed IQ2_XXS routed-expert weight tensor.

    Stored as separate sub-tensors so safetensors can hold them in standard
    dtypes. ``d`` is the per-block scale; ``qs`` are the packed 64-byte
    payloads (32 uint16s reinterpreted as uint8).

    Shapes (per expert):
        d  : float16[n_rows, n_blocks]
        qs : uint8  [n_rows, n_blocks, 64]
    """

    d: np.ndarray   # float16
    qs: np.ndarray  # uint8


@dataclass
class Q2KTensors:
    """Packed Q2_K routed-expert weight tensor.

    Shapes (per expert):
        d      : float16[n_rows, n_blocks]
        dmin   : float16[n_rows, n_blocks]
        scales : uint8  [n_rows, n_blocks, 16]
        qs     : uint8  [n_rows, n_blocks, 64]
    """

    d: np.ndarray
    dmin: np.ndarray
    scales: np.ndarray
    qs: np.ndarray


@dataclass
class Q8KActivation:
    """Quantized activation in Q8_K format.

    Shapes (per token):
        d     : float32[n_blocks]
        qs    : int8   [n_blocks, 256]
        bsums : int16  [n_blocks, 16]
    """

    d: np.ndarray
    qs: np.ndarray
    bsums: np.ndarray
