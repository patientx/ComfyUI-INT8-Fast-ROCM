"""
Asymmetric W4A8 support for INT8-Fast-ROCM (comfy-kitchen PR#90, format="asym_w4a8_int8").

Verified tensor layout (from actual checkpoint header):
    <layer>.weight              int8   [N, K//2]   packed int4 codes (unsigned 0-15,
                                                     low nibble=K_even, high nibble=K_odd)
    <layer>.weight_codebook     float32 [16]        Lloyd-Max levels, indexed by code
    <layer>.weight_s_channel    float32 [N]         per-output-channel scale
    <layer>.weight_s_rel        float8_e4m3 [N, K//group_size]  per-group relative scale

    dequant_float[n,k] = codebook[code[n,k]] * s_channel[n] * s_rel[n, k // group_size]

group_size comes from the per-layer entry in the safetensors header's
_quantization_metadata (e.g. 16) -- NOT the same as convrot_groupsize (e.g. 256),
which is the unrelated activation-rotation group size already handled elsewhere
in Int8TensorwiseOps._load_from_state_dict.

dequant_float is fp32 [N, K] full precision. Callers should re-quantize it with
the existing quantize_int8_axiswise(x, dim=1) to get an int8 weight + [N,1] scale
in the same format the rest of the W8A8 pipeline already expects.
"""

import json
import logging
import torch
from torch import Tensor


def unpack_int4_unsigned(packed: Tensor) -> Tensor:
    """[N, K//2] int8 (packed nibbles) -> [N, K] int32 codes in [0, 15] (unsigned)."""
    assert packed.ndim == 2, f"Expected 2D tensor, got {packed.ndim}D"
    N, K_packed = packed.shape
    K = K_packed * 2

    # Mask to raw byte value first so sign of the underlying int8 storage doesn't matter.
    p32 = packed.to(torch.int32) & 0xFF
    lo = p32 & 0x0F          # K_even
    hi = (p32 >> 4) & 0x0F   # K_odd

    codes = torch.zeros((N, K), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    return codes


def dequant_asym_w4a8_to_float(
    packed_codes: Tensor,
    codebook: Tensor,
    s_channel: Tensor,
    s_rel: Tensor,
    group_size: int,
) -> Tensor:
    """
    Returns fp32 [N, K] dequantized weight. Caller re-quantizes to int8 with
    quantize_int8_axiswise() for the existing Triton int8 GEMM path.
    """
    N, K_packed = packed_codes.shape
    K = K_packed * 2

    codes = unpack_int4_unsigned(packed_codes)             # [N, K] int32, 0-15
    codebook_f32 = codebook.to(torch.float32)               # [16]
    vals = codebook_f32[codes.long()]                        # [N, K] float32

    s_channel_f32 = s_channel.to(torch.float32).reshape(N, 1)  # [N, 1]
    s_rel_f32 = s_rel.to(torch.float32)                         # [N, num_groups]

    num_groups = s_rel_f32.shape[1]
    if num_groups * group_size != K:
        raise ValueError(
            f"W4A8: group mismatch, num_groups({num_groups}) * group_size({group_size}) != K({K})"
        )

    s_rel_expanded = s_rel_f32.repeat_interleave(group_size, dim=1)  # [N, K]

    return vals * s_channel_f32 * s_rel_expanded


def bridge_w4a8_metadata_to_comfy_quant(state_dict: dict, metadata) -> int:
    """Bridge per-layer entries from the safetensors header's _quantization_metadata
    into synthetic <layer>.comfy_quant marker tensors, same pattern as INT4 ConvRot."""
    if not metadata or "_quantization_metadata" not in metadata:
        return 0

    try:
        quant_meta = json.loads(metadata["_quantization_metadata"])
    except Exception as e:
        logging.warning(f"W4A8: failed to parse _quantization_metadata: {e}")
        return 0

    bridged = 0
    for layer_name, quant_conf in quant_meta.get("layers", {}).items():
        if quant_conf.get("format") != "asym_w4a8_int8":
            continue

        candidates = (
            layer_name + ".weight",
            "model." + layer_name + ".weight",
            "diffusion_model." + layer_name + ".weight",
            "model.diffusion_model." + layer_name + ".weight",
        )
        weight_key = next((k for k in candidates if k in state_dict), None)
        if weight_key is None:
            suffix = layer_name + ".weight"
            matches = [k for k in state_dict if k == suffix or k.endswith("." + suffix)]
            if len(matches) == 1:
                weight_key = matches[0]

        if weight_key is None:
            logging.warning(f"W4A8: metadata bridge could not find weight for layer '{layer_name}'")
            continue

        base_key = weight_key[:-len(".weight")]
        quant_key = base_key + ".comfy_quant"
        if quant_key not in state_dict:
            payload = json.dumps(quant_conf, separators=(",", ":")).encode("utf-8")
            state_dict[quant_key] = torch.tensor(list(payload), dtype=torch.uint8)
        bridged += 1

    if bridged:
        logging.info(f"W4A8: bridged safetensors header metadata for {bridged} asym_w4a8_int8 layers")
    return bridged
