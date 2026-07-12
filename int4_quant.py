"""
INT4 ConvRot W4A4 support, without depending on comfy_kitchen.

Comfy's native int4 format ("convrot_w4a4") hands off entirely to the
closed comfy_kitchen package, which needs Triton or hipBLASLt. This module
reimplements the same math with an experimental packed-INT4 Triton DP4A path (verified bit-identical against
comfy_kitchen 0.2.17's own eager reference), using Triton for the packed W4A4 GEMM.

On-disk format (per Linear layer), confirmed against a real checkpoint:
  <prefix>weight        int8, shape (out_features, in_features // 2) --
                          two signed int4 nibbles packed per byte
                          (low=even col, high=odd col).
  <prefix>weight_scale   fp32, shape (out_features,) -- per-row scale.
  <prefix>comfy_quant    JSON: {"format": "convrot_w4a4", "convrot_groupsize": 256}

Layers can instead be tagged "int8_tensorwise" (a handful of precision-
sensitive layers comfy's quantizer keeps at plain int8) or have no
comfy_quant tag at all (kept high precision) -- handled per-layer, so a
checkpoint mixing int4 / int8 / bf16 in one file works without special-casing.

Hadamard rotation is delegated to this repo's convrot.py, which already
matches comfy_kitchen's exact construction.

v1 scope: loading + on-the-fly quantization + LoRA (pre_lora bake-in at load
time, plus post-load bake-in/Stochastic/Dynamic via Int4ModelPatcher --
see that class below). Still missing: Aimdo dynamic/lowvram deferred-patch
machinery (see INT8ModelPatcher in int8_quant.py for what that would take
to add -- deferring quantization itself, not just LoRA, to first-use).
"""

import json
import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.lora
import comfy.utils

from .convrot import build_hadamard, rotate_weight, rotate_activation
from .triton_int4_mm import triton_int4_mm

# ConvRot W4A4 is Hadamard rotation applied in blocks of this size along the
# input (K) dimension. Must be a power of 4 -- comfy's own default is 256.
CONVROT_W4A4_GROUP_SIZE = 256

# Symmetric int4 quantizer emission range: signed nibble field can hold
# [-8, 7], but (matching comfy_kitchen exactly) only [-7, 7] is emitted so
# the dequant range stays symmetric about zero. scale = absmax / 7.
_INT4_MAX = 7


# =============================================================================
# Pack / unpack -- byte-for-byte compatible with comfy_kitchen's
# _pack_int4_row_major / _unpack_int4_row_major (backends/eager/svdquant.py).
# Confirmed by a real skeleton-load test that this IS the on-disk weight
# convention (see module docstring for the full story of getting this wrong,
# then right, then wrong again, then right).
# =============================================================================

def pack_int4_row_major(values: Tensor) -> Tensor:
    """Pack (..., K) int8-valued nibbles into (..., K // 2) int8 storage.

    Low nibble = even column, high nibble = odd column. Caller must ensure
    values fit in a signed 4-bit field ([-8, 7]); use quantize_signed_int4_rowwise
    for the actual quantizer (which clamps to [-7, 7]).
    """
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def unpack_int4_row_major(packed: Tensor) -> Tensor:
    """Inverse of pack_int4_row_major, signed-nibble interpretation."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


# =============================================================================
# Quantize -- weight side is packed 2-per-byte (matches the real on-disk
# format). Activation side (quantize_signed_int4_rowwise, used per forward
# pass on the rotated activation) is NEVER written to disk -- it's purely an
# internal scratch tensor for this one matmul, so there is no compatibility
# requirement on its layout. Kept unpacked (one element per int8 byte) for
# simplicity; this was never the bug, only the on-disk WEIGHT layout was.
# =============================================================================

def quantize_signed_int4_rowwise(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row symmetric int4-range quantization, full-width int8 storage
    (one element per value, no bit packing) -- used for the ephemeral
    per-forward activation quantization only."""
    rows = x.shape[0]
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = (x / scales).round().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return q, scales.reshape(rows).to(torch.float32)


def quantize_signed_int4_rowwise_packed(x: Tensor) -> tuple[Tensor, Tensor]:
    """Same as quantize_signed_int4_rowwise but packs the result 2-per-byte.
    Used for the WEIGHT side, to match the real on-disk format."""
    rows = x.shape[0]
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = (x / scales).round().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return pack_int4_row_major(q), scales.reshape(rows).to(torch.float32)


def quantize_signed_int4_packed_with_scale(x: Tensor, scale: Tensor) -> Tensor:
    """Pack a (rotated, full-precision) weight to signed int4, 2-per-byte,
    REUSING a caller-supplied per-row scale instead of recomputing absmax.

    Used for LoRA re-patching: mirrors int8_quant.py's
    ``quantize_int8(patched_weight_float, scale)`` -- the weight_scale
    buffer that was already on disk/computed at first load stays fixed
    across repeated LoRA (re-)patches instead of drifting with every call,
    which keeps behavior deterministic and matches what int8's equivalent
    path does. NOTE: if a LoRA delta pushes a value outside the original
    absmax-derived range, it silently clamps at +/-7 same as any other
    int4 quantization -- same tradeoff int8 already accepts.
    """
    scale = scale.reshape(-1, 1).float()
    q = (x.float() / scale).round().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return pack_int4_row_major(q)


def stochastic_round_int4_packed_delta(x: Tensor, scale: Tensor, seed: int = 0) -> Tensor:
    """Stochastic-rounding counterpart to quantize_signed_int4_packed_with_scale.
    Same idea as int8_quant.py's stochastic_round_int8_delta, adapted to the
    narrower [-7, 7] int4 emission range and 2-per-byte packing."""
    scale = scale.reshape(-1, 1).float()
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    x_scaled = x.float() / scale
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor
    del x_scaled

    random_vals = torch.rand(x_floor.shape, generator=generator, device=x.device, dtype=x_floor.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    del random_vals, fraction, x_floor

    q = torch.clamp(x_rounded, -_INT4_MAX, _INT4_MAX).to(torch.int8)
    return pack_int4_row_major(q)


# =============================================================================
# Quantize / dequantize a whole weight matrix (offline, at load/convert time)
# =============================================================================

def quantize_convrot_w4a4_weight(
    weight: Tensor,
    convrot_groupsize: int = CONVROT_W4A4_GROUP_SIZE,
) -> tuple[Tensor, Tensor]:
    """Rotate a weight matrix with ConvRot and quantize+pack it to signed
    int4, 2-per-byte (matches the real on-disk format)."""
    if weight.dim() != 2:
        raise ValueError(f"ConvRot W4A4 expects a 2D weight, got shape {tuple(weight.shape)}")
    in_f = weight.shape[-1]
    if in_f % convrot_groupsize != 0:
        raise ValueError(f"in_features {in_f} not divisible by convrot_groupsize {convrot_groupsize}")
    H = build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    w_rot = rotate_weight(weight, H, group_size=convrot_groupsize)
    return quantize_signed_int4_rowwise_packed(w_rot)


def dequantize_simple_int8_rowwise(
    weight: Tensor,
    scale: Tensor,
    out_features: int,
    convrot: bool = False,
    convrot_groupsize: int = CONVROT_W4A4_GROUP_SIZE,
) -> Tensor:
    """Fallback dequantizer for stray layers tagged 'int8_tensorwise' (plain
    full-width int8, not bit-packed) that can show up mixed into an
    otherwise int4 checkpoint -- comfy's own quantizer sometimes keeps a
    handful of precision-sensitive layers at plain int8 instead of int4.

    IMPORTANT: 'int8_tensorwise' is just the storage format name -- whether
    the weight was ConvRot-rotated is a SEPARATE 'convrot' boolean in the
    same comfy_quant JSON (confirmed from int8_save.py's own extra_keys
    schema: quant_conf = {"format": "int8_tensorwise", "convrot": use_convrot}).
    A rotated layer stored this way still needs the rotation undone before
    the dequantized weight means anything -- skipping that (as an earlier
    version of this function did) would silently corrupt this layer's
    output, and if that layer happens to be the network's input/patchify
    projection, it poisons everything downstream regardless of how correct
    every other layer's math is.

    Not performance critical either way (typically only a couple of layers
    per model), so we just dequantize immediately to float and treat the
    layer as ordinary high precision from here on."""
    scale = scale.float()
    if scale.numel() == out_features:
        scale = scale.reshape(-1, 1)
    elif scale.numel() == 1:
        scale = scale.reshape(1, 1)
    else:
        logging.warning(
            f"INT4 ConvRot: unrecognized weight_scale shape {tuple(scale.shape)} for a "
            f"non-convrot_w4a4 layer (out_features={out_features}); dequantizing without "
            f"scale correction, this layer's output will be wrong."
        )
        return weight.float()

    deq = weight.float() * scale

    if convrot:
        in_f = deq.shape[-1]
        if in_f % convrot_groupsize != 0:
            logging.warning(
                f"INT4 ConvRot: int8_tensorwise layer has convrot=True but in_features={in_f} "
                f"not divisible by convrot_groupsize={convrot_groupsize}; cannot un-rotate, "
                f"this layer's output will be wrong."
            )
            return deq
        H = build_hadamard(convrot_groupsize, device=deq.device, dtype=deq.dtype)
        deq = rotate_weight(deq, H, group_size=convrot_groupsize)

    return deq


# =============================================================================
# Forward -- pure PyTorch, no hipBLASLt, no Triton, no comfy_kitchen import.
# Rotate activation online, quantize it per-row to int4 (unpacked scratch
# tensor), unpack the WEIGHT (which is packed 2-per-byte on disk), matmul,
# rescale.
# =============================================================================

def dequantize_convrot_w4a4_weight_full(
    qweight: Tensor,
    wscales: Tensor,
    convrot_groupsize: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Fully reconstruct the ORIGINAL (non-rotated) weight: unpack, rescale
    per-row, then rotate back (Hadamard is self-inverse/orthogonal, so
    applying it again undoes the original rotation). Used only by the
    debug_weight_only_reference path to isolate whether weight decode is
    correct independent of the per-forward activation quantization."""
    w_int = unpack_int4_row_major(qweight).to(dtype=dtype)
    w_deq = w_int * wscales.reshape(-1, 1).to(dtype=dtype)
    H = build_hadamard(convrot_groupsize, device=w_deq.device, dtype=dtype)
    return rotate_weight(w_deq, H, group_size=convrot_groupsize)


def convrot_w4a4_forward_weight_only_reference(
    x: Tensor,
    qweight: Tensor,
    wscales: Tensor,
    bias: Tensor | None,
    convrot_groupsize: int,
    compute_dtype: torch.dtype,
) -> Tensor:
    """DIAGNOSTIC ONLY: reconstructs the full (un-rotated) dequantized
    weight once, then runs an ordinary plain float linear -- NO activation
    rotation, NO activation quantization at all. Isolates whether the WEIGHT
    decode (packing/scale/rotation) is correct independent of the per-
    forward dynamic activation quantization step. Not meant for production
    use (recomputes the full-precision weight every call, no memory savings,
    and doesn't test the real W4A4 compute path)."""
    w_full = dequantize_convrot_w4a4_weight_full(qweight, wscales, convrot_groupsize, dtype=compute_dtype)
    bias_c = bias.to(dtype=compute_dtype) if bias is not None else None
    return F.linear(x.to(dtype=compute_dtype), w_full, bias_c)



def prepare_rotated_activation(
    x: Tensor,
    convrot_groupsize: int,
    compute_dtype: torch.dtype,
    skip_rotation: bool = False,
) -> tuple[Tensor, torch.Size]:
    """Flatten + cast + (optionally) Hadamard-rotate activations.

    Split out of convrot_w4a4_forward so the dynamic-LoRA additive branch
    (Int4ModelPatcher / Linear.forward) can compute this exactly once per
    forward call and reuse it for both the W4A4 matmul and the LoRA
    down-projection, instead of rotating the same activation twice.
    """
    orig_shape = x.shape
    in_f = orig_shape[-1]
    if in_f % convrot_groupsize != 0:
        raise ValueError(
            f"Input K={in_f} not divisible by convrot_groupsize "
            f"{convrot_groupsize}"
        )

    x2d = x.reshape(-1, in_f).to(compute_dtype).contiguous()

    if skip_rotation:
        return x2d, orig_shape

    H = build_hadamard(
        convrot_groupsize,
        device=x2d.device,
        dtype=compute_dtype,
    )
    x_rot = rotate_activation(
        x2d, H, group_size=convrot_groupsize
    ).contiguous()
    return x_rot, orig_shape


def convrot_w4a4_forward(
    x: Tensor,
    qweight: Tensor,
    wscales: Tensor,
    bias: Tensor | None,
    convrot_groupsize: int,
    compute_dtype: torch.dtype,
    skip_rotation: bool = False,
    qweight_unpacked: Tensor | None = None,
    x_rot: Tensor | None = None,
    orig_shape: torch.Size | None = None,
) -> Tensor:
    """Experimental Triton packed-INT4 DP4A path.

    qweight stays packed [N, K//2]. The Triton kernel decodes each nibble
    while loading the current weight tile and feeds the resulting INT8 tile
    directly to tl.dot with an INT32 accumulator.

    If x_rot/orig_shape are supplied (already-rotated activation from a
    prior prepare_rotated_activation call, e.g. shared with a dynamic LoRA
    branch), that is used as-is instead of recomputing the rotation.
    """
    if x_rot is None:
        x_rot, orig_shape = prepare_rotated_activation(
            x, convrot_groupsize, compute_dtype, skip_rotation=skip_rotation
        )
    in_f = orig_shape[-1]
    if in_f != qweight.shape[-1] * 2:
        raise ValueError(
            f"Input K={in_f} does not match packed qweight "
            f"K={qweight.shape[-1] * 2}"
        )

    qact, x_scale = quantize_signed_int4_rowwise(x_rot)
    qact = qact.contiguous()

    # qweight_unpacked intentionally ignored in this test path. Keeping the
    # argument preserves call compatibility with earlier test files.
    out = triton_int4_mm(
        qact,
        qweight,
        out_dtype=torch.int32,
    )

    # Match the original eager math:
    #   int32_dot * activation_row_scale * weight_row_scale + bias
    out = out.to(dtype=torch.float32)
    out.mul_(x_scale.reshape(-1, 1).to(
        device=out.device, dtype=torch.float32
    ))
    out.mul_(wscales.reshape(1, -1).to(
        device=out.device, dtype=torch.float32
    ))

    if bias is not None:
        out.add_(bias.to(
            device=out.device, dtype=torch.float32
        ).reshape(1, -1))

    return out.to(dtype=compute_dtype).reshape(
        *orig_shape[:-1], wscales.shape[0]
    )


# =============================================================================
# Int4ConvRotOps -- ComfyUI custom_operations, same slot Int8TensorwiseOps
# plugs into (model_options={"custom_operations": Int4ConvRotOps}).
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


# Same per-architecture exclusion lists as Int8TensorwiseOps (int8_unet_loader.py),
# kept in sync here so int4 on-the-fly quantization skips the same
# precision-sensitive layers (embeddings, modulation, final projection, etc).
MODEL_TYPE_EXCLUSIONS = {
    "flux2": ['img_in', 'time_in', 'guidance_in', 'txt_in',
              'double_stream_modulation_img', 'double_stream_modulation_txt',
              'single_stream_modulation'],
    "z-image": ['cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner',
                'final_layer', 'noise_refiner', 'adaLN', 'x_pad_token', 'layers.0.'],
    "chroma": ['distilled_guidance_layer', 'final_layer', 'img_in', 'txt_in', 'nerf_image_embedder',
               'nerf_blocks', 'nerf_final_layer_conv', '__x0__'],
    "qwen": ['time_text_embed', 'img_in', 'norm_out', 'proj_out', 'txt_in'],
    "ernie": ['time', 'x_embedder', 'text_proj', 'adaLN'],
    "anima": ['embed', 'llm'],
    "krea2": ['first', 'last', 'tmlp', 'tproj', 'txtfusion', 'txtmlp'],
    "hidream o1": ['embed', 'language_model.layers.35.mlp'],
    "boogu": ['embed', 'refine', 'norm_out'],
    "ideogram4": ['embed_image_indicator', 't_embedding', 'proj'],
    "wan": ['patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head',
            'img_emb', 'face_adapter', 'face_encoder', 'motion_encoder', 'pose_patch_embedding'],
    "ltx2": ['adaln', 'embedding', 'patchify', 'to_gate_logits', 'proj_out',
             'model.audio', 'model.video', 'model.av', 'model.patch', 'model.proj', 'shift'],
}


if _COMFY_OPS_AVAILABLE:
    class Int4ConvRotOps(manual_cast):
        """Custom ComfyUI operations for INT4 ConvRot W4A4, eager-only
        (no comfy_kitchen, no Triton, no hipBLASLt)."""

        excluded_names: list[str] = []
        dynamic_quantize = False  # on-the-fly bf16/fp16 -> int4 quantization toggle
        compute_dtype = None      # optional override; default follows activation dtype
        convrot_groupsize = CONVROT_W4A4_GROUP_SIZE
        _is_prequantized = False
        _logged_otf = False
        debug_nan_check = False   # set True to log every layer whose output contains NaN/Inf
        debug_skip_rotation = False  # A/B test: skip Hadamard rotation on activations (weight stays as-loaded, i.e. already rotated on disk -- this deliberately breaks correctness, it's purely diagnostic to isolate whether rotation-handling is the source of an accuracy bug)
        debug_weight_only_reference = False  # A/B test: fully dequantize+un-rotate weight, plain float linear, NO activation quantization/rotation at all. Isolates weight-decode correctness from activation-quantization correctness.
        _dtype_guard_count = 0
        _load_time_dtype_logged = False
        _debug_nan_count = 0
        _debug_nan_max_reports = 25

        # --- LoRA support (mirrors Int8TensorwiseOps in int8_quant.py) ---
        lora_mode = "None"        # None/Stochastic bake LoRA into the int4 weight at (re-)patch time; Dynamic applies it additively at inference
        dynamic_lora = False      # True iff lora_mode == "Dynamic"
        lora_patches: dict = {}   # normalized_key -> [(value, offset, function, strength), ...], consumed at LOAD time (pre_lora) regardless of lora_mode
        lora_strength = 1.0
        skeleton_meta_init = False  # Temporary mode for LoRA key-map discovery (meta-device skeleton model, no real weights)

        @staticmethod
        def _default_compute_dtype(x: Tensor) -> torch.dtype:
            if x.dtype in (torch.float16, torch.bfloat16):
                return x.dtype
            return torch.float16

        class Linear(manual_cast.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                if getattr(Int4ConvRotOps, "skeleton_meta_init", False):
                    # Fast meta-device skeleton used only to discover LoRA
                    # key-mapping (comfy.lora.model_lora_keys_unet) without
                    # allocating/loading real weights. Same trick as
                    # Int8TensorwiseOps.Linear.__init__ in int8_quant.py.
                    nn.Module.__init__(self)
                    self.in_features = in_features
                    self.out_features = out_features
                    tensor_kwargs = {"device": "meta"}
                    if dtype is not None:
                        tensor_kwargs["dtype"] = dtype
                    self.weight = nn.Parameter(torch.empty((out_features, in_features), **tensor_kwargs), requires_grad=False)
                    self.bias = nn.Parameter(torch.empty((out_features,), **tensor_kwargs), requires_grad=False) if bias else None
                    self.weight_comfy_model_dtype = dtype
                    self.bias_comfy_model_dtype = dtype
                else:
                    super().__init__(in_features, out_features, bias, device, dtype)
                self.register_buffer('weight_scale', None)
                self._is_quantized = False
                self._convrot_groupsize = Int4ConvRotOps.convrot_groupsize
                self._linear_dtype = "int4"
                self.comfy_cast_weights = False
                self._debug_name = "<unnamed>"
                self.lora_patches = []  # [(down_scaled, up, start, size), ...] set by Int4ModelPatcher for lora_mode == "Dynamic"

            def reset_parameters(self):
                return None

            @staticmethod
            def _normalize_lora_key(key):
                if not isinstance(key, str):
                    return key
                for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                    if key.startswith(p):
                        return key[len(p):]
                return key

            @staticmethod
            def _is_bias_key(key):
                return isinstance(key, str) and key.endswith(".bias")

            @staticmethod
            def _format_lora_patches(patches):
                formatted = []
                for patch in patches or []:
                    if len(patch) == 4:
                        v, offset, function, strength = patch
                    else:
                        v, offset, function = patch
                        strength = getattr(Int4ConvRotOps, "lora_strength", 1.0)
                    formatted.append((strength, v, 1.0, offset, function))
                return formatted

            def _apply_int4_lora_patches_float(self, tensor, key, patches, device):
                """Apply LoRA patches to a FLOAT (non-packed, non-rotated)
                weight/bias tensor. Used for: (a) the on-the-fly quantize
                branch, before rotation+quantization; (b) bias, which is
                never quantized; (c) the 'int8_tensorwise' stray-layer
                dequant fallback, which is already float by the time it's
                stored."""
                if not patches or tensor.dtype == torch.int8:
                    return tensor
                temp_dtype = comfy.model_management.lora_compute_dtype(device)
                tensor_temp = tensor.to(device=device, non_blocking=True).to(dtype=temp_dtype)
                return comfy.lora.calculate_weight(self._format_lora_patches(patches), tensor_temp, key)

            def _apply_int4_lora_patches_prequantized(self, qweight, wscale, key, patches, device):
                """Apply LoRA patches to an already-packed convrot_w4a4
                weight: unpack -> rescale -> un-rotate -> patch in float ->
                re-rotate -> re-quantize -> re-pack. Recomputes the per-row
                scale from the patched weight (unlike the Int4ModelPatcher
                dynamic-repatch path, which reuses the original scale) since
                this only runs once, at load time."""
                if not patches:
                    return qweight, wscale
                temp_dtype = comfy.model_management.lora_compute_dtype(device)
                qweight_dev = qweight.to(device, non_blocking=True)
                wscale_dev = wscale.to(device, non_blocking=True)
                weight_float = dequantize_convrot_w4a4_weight_full(
                    qweight_dev, wscale_dev, self._convrot_groupsize, dtype=temp_dtype
                )
                patched = comfy.lora.calculate_weight(self._format_lora_patches(patches), weight_float, key)
                new_qweight, new_wscale = quantize_convrot_w4a4_weight(
                    patched.float(), convrot_groupsize=self._convrot_groupsize
                )
                return new_qweight.cpu(), new_wscale.cpu()

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs):
                self._debug_name = prefix
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                quant_key = prefix + "comfy_quant"
                bias_key = prefix + "bias"

                def pop_metadata(sd, p, k):
                    v = sd.pop(p + k, None)
                    if v is not None:
                        return v
                    v = sd.pop("model." + p + k, None)
                    if v is not None:
                        return v
                    if p.startswith("model."):
                        v = sd.pop(p[6:] + k, None)
                        if v is not None:
                            return v
                    if p.startswith("diffusion_model."):
                        v = sd.pop("diffusion_model." + p + k, None)
                        if v is not None:
                            return v
                    return None

                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)
                weight_scale = pop_metadata(state_dict, prefix, "weight_scale")
                comfy_quant_tensor = pop_metadata(state_dict, prefix, "comfy_quant")
                # Not our format -- ignore and leave for whatever loader owns it.
                _ = state_dict.pop(prefix + "input_scale", None)

                quant_conf = None
                if comfy_quant_tensor is not None:
                    try:
                        quant_conf = json.loads(bytes(comfy_quant_tensor.tolist()).decode("utf-8"))
                    except Exception:
                        quant_conf = None

                is_convrot_w4a4 = isinstance(quant_conf, dict) and quant_conf.get("format") == "convrot_w4a4"
                is_other_quant_format = (
                    isinstance(quant_conf, dict)
                    and quant_conf.get("format") not in (None, "convrot_w4a4")
                )

                # --- pre_lora bake-in lookup (applied at LOAD time regardless
                # of lora_mode/dynamic_lora -- those only govern LoRAs added
                # LATER via a grouped-LoRA node through Int4ModelPatcher).
                weight_lora_patches = Int4ConvRotOps.lora_patches.get(self._normalize_lora_key(weight_key))
                bias_lora_patches = Int4ConvRotOps.lora_patches.get(self._normalize_lora_key(bias_key))
                if weight_lora_patches or bias_lora_patches:
                    if not hasattr(Int4ConvRotOps, 'applied_lora_patches'):
                        Int4ConvRotOps.applied_lora_patches = set()
                    if weight_lora_patches:
                        Int4ConvRotOps.applied_lora_patches.add(self._normalize_lora_key(weight_key))
                    if bias_lora_patches:
                        Int4ConvRotOps.applied_lora_patches.add(self._normalize_lora_key(bias_key))
                lora_device = comfy.model_management.get_torch_device()

                if weight_tensor is not None and weight_tensor.dtype == torch.int8 and is_convrot_w4a4:
                    if weight_scale is None:
                        error_msgs.append(f"{weight_key}: convrot_w4a4 weight found but weight_scale is missing")
                    else:
                        expected_in = weight_tensor.shape[-1] * 2
                        if expected_in != self.in_features:
                            error_msgs.append(
                                f"{weight_key}: int4 weight implies in_features={expected_in}, "
                                f"but this Linear expects {self.in_features}"
                            )
                        self._is_quantized = True
                        self._convrot_groupsize = int(quant_conf.get("convrot_groupsize", CONVROT_W4A4_GROUP_SIZE))
                        self._linear_dtype = quant_conf.get("linear_dtype", "int4")

                        if weight_lora_patches:
                            logging.info(f"INT4 ConvRot: baking pre_lora into prequantized layer '{prefix}' (dequant -> patch -> requant)")
                            weight_tensor, weight_scale = self._apply_int4_lora_patches_prequantized(
                                weight_tensor, weight_scale.float(), weight_key, weight_lora_patches, lora_device
                            )

                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        self.register_buffer('weight_scale', weight_scale.float())
                        Int4ConvRotOps._is_prequantized = True
                        if not Int4ConvRotOps._load_time_dtype_logged:
                            Int4ConvRotOps._load_time_dtype_logged = True
                            logging.info(
                                f"INT4 ConvRot LOAD-TIME CHECK: layer '{prefix}' weight stored as "
                                f"{self.weight.dtype} immediately after loading (should be torch.int8)."
                            )

                elif weight_tensor is not None and weight_tensor.dtype == torch.int8 and is_other_quant_format:
                    # Stray layer tagged with a different quant format (e.g.
                    # "int8_tensorwise") mixed into an otherwise int4
                    # checkpoint -- comfy's own quantizer sometimes keeps a
                    # handful of precision-sensitive layers at plain int8.
                    # Not performance critical (a couple of layers per model
                    # typically), so just dequantize immediately to float
                    # rather than implement a second quant kernel here.
                    if weight_scale is None:
                        logging.warning(
                            f"INT4 ConvRot: {weight_key} tagged '{quant_conf.get('format')}' but no "
                            f"weight_scale found; loading raw int8 values as float (will be wrong)."
                        )
                        dequant = weight_tensor.float()
                    else:
                        use_convrot = bool(quant_conf.get("convrot", False))
                        conv_gs = int(quant_conf.get("convrot_groupsize", CONVROT_W4A4_GROUP_SIZE))
                        dequant = dequantize_simple_int8_rowwise(
                            weight_tensor, weight_scale, self.out_features,
                            convrot=use_convrot, convrot_groupsize=conv_gs,
                        )
                        logging.info(
                            f"INT4 ConvRot: {prefix} int8_tensorwise dequant -- convrot={use_convrot}"
                            + (f" groupsize={conv_gs}" if use_convrot else "")
                        )
                    if weight_lora_patches:
                        dequant = self._apply_int4_lora_patches_float(
                            dequant, weight_key, weight_lora_patches, lora_device
                        ).to(dequant.dtype).cpu()
                    self._is_quantized = False
                    self.weight = nn.Parameter(dequant, requires_grad=False)
                    logging.info(
                        f"INT4 ConvRot: {prefix} uses format '{quant_conf.get('format')}', not convrot_w4a4 -- "
                        f"loaded as dequantized high precision (not performance critical, a handful of layers only)."
                    )

                elif weight_tensor is not None and weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    if weight_lora_patches:
                        orig_dtype = weight_tensor.dtype
                        weight_tensor = self._apply_int4_lora_patches_float(
                            weight_tensor, weight_key, weight_lora_patches, lora_device
                        ).to(orig_dtype).cpu()
                    is_excluded = any(ex in prefix for ex in Int4ConvRotOps.excluded_names)
                    is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                    in_f = weight_tensor.shape[-1]
                    fits_groupsize = in_f % Int4ConvRotOps.convrot_groupsize == 0
                    should_quantize = (
                        Int4ConvRotOps.dynamic_quantize
                        and not is_excluded
                        and not is_dim1
                        and fits_groupsize
                    )

                    if not should_quantize:
                        self._is_quantized = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        if Int4ConvRotOps.dynamic_quantize and not is_excluded and not is_dim1 and not fits_groupsize:
                            logging.warning(
                                f"INT4 ConvRot: {prefix} in_features={in_f} not divisible by "
                                f"convrot_groupsize={Int4ConvRotOps.convrot_groupsize}, keeping high precision."
                            )
                    else:
                        if not Int4ConvRotOps._logged_otf:
                            logging.info("INT4 ConvRot: quantizing on-the-fly (eager, no comfy_kitchen)")
                            Int4ConvRotOps._logged_otf = True

                        device = comfy.model_management.get_torch_device()
                        w_gpu = weight_tensor.to(device, non_blocking=True).float()
                        try:
                            q_weight, q_scale = quantize_convrot_w4a4_weight(
                                w_gpu, convrot_groupsize=Int4ConvRotOps.convrot_groupsize
                            )
                        finally:
                            del w_gpu

                        self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                        self.register_buffer('weight_scale', q_scale.cpu())
                        self._is_quantized = True
                        self._convrot_groupsize = Int4ConvRotOps.convrot_groupsize
                        self._linear_dtype = "int4"
                        del q_weight, q_scale
                else:
                    self._is_quantized = False
                    if weight_tensor is not None:
                        if weight_tensor.dtype == torch.int8:
                            logging.warning(
                                f"INT4 ConvRot: {weight_key} is int8 with no recognizable comfy_quant "
                                f"metadata (format={quant_conf.get('format') if isinstance(quant_conf, dict) else None!r}); "
                                f"storing as-is, this layer's output will likely be wrong."
                            )
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                    else:
                        missing_keys.append(weight_key)

                if bias_tensor is not None:
                    if bias_lora_patches:
                        bias_tensor = self._apply_int4_lora_patches_float(
                            bias_tensor, bias_key, bias_lora_patches, lora_device
                        ).to(bias_tensor.dtype).cpu()
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

                if self.weight is not None:
                    self.weight_comfy_model_dtype = self.weight.dtype
                if self.weight_scale is not None:
                    self.weight_scale_comfy_model_dtype = self.weight_scale.dtype
                if self.bias is not None:
                    self.bias_comfy_model_dtype = self.bias.dtype

            def convert_weight(self, _weight, inplace=False):
                if not self._is_quantized:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                elif out_weight.dtype == torch.int8:
                    new_weight = out_weight
                else:
                    # Fallback re-quantize path (e.g. a patch produced a float delta).
                    q_weight, q_scale = quantize_convrot_w4a4_weight(
                        out_weight.float(), convrot_groupsize=self._convrot_groupsize
                    )
                    new_weight = q_weight
                    self.register_buffer('weight_scale', q_scale.to(self.weight_scale.device))

                if return_weight:
                    return new_weight
                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                if return_weight:
                    return out_bias
                if inplace_update and self.bias is not None:
                    self.bias.data.copy_(out_bias)
                else:
                    self.bias = nn.Parameter(out_bias, requires_grad=False)

            def forward(self, x: Tensor) -> Tensor:
                need_cast = self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0

                if not self._is_quantized:
                    if need_cast:
                        weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        result = out
                    elif x.device != self.weight.device or x.dtype != self.weight.dtype:
                        weight = self.weight.to(device=x.device, dtype=x.dtype)
                        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
                        result = F.linear(x, weight, bias)
                    else:
                        result = F.linear(x, self.weight, self.bias)

                    if Int4ConvRotOps.debug_nan_check and Int4ConvRotOps._debug_nan_count < Int4ConvRotOps._debug_nan_max_reports:
                        if result.is_cuda:
                            torch.cuda.synchronize(result.device)
                        bad = not torch.isfinite(result).all().item()
                        if bad:
                            Int4ConvRotOps._debug_nan_count += 1
                            logging.error(
                                f"INT4 ConvRot DEBUG [{Int4ConvRotOps._debug_nan_count}]: NaN/Inf output at "
                                f"high-precision/dequant layer '{self._debug_name}' "
                                f"(input already had NaN/Inf: {not torch.isfinite(x).all().item()}, "
                                f"weight_function count: {len(self.weight_function)}, "
                                f"bias_function count: {len(self.bias_function)}, "
                                f"comfy_cast_weights: {self.comfy_cast_weights}, need_cast: {need_cast}). "
                                f"weight dtype={self.weight.dtype} shape={tuple(self.weight.shape)}"
                            )
                    return result

                # INT4 ConvRot path
                if need_cast:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True,
                    )
                    if weight is not None and weight.device != x.device:
                        weight = weight.to(x.device, non_blocking=True)
                    if bias is not None and bias.device != x.device:
                        bias = bias.to(x.device, non_blocking=True)
                else:
                    weight = self.weight
                    bias = self.bias
                    offload_stream = None
                    if weight is not None and weight.device != x.device:
                        weight = weight.to(x.device, non_blocking=True)
                    if bias is not None and bias.device != x.device:
                        bias = bias.to(x.device, non_blocking=True)

                if weight.dtype != torch.int8 and Int4ConvRotOps._dtype_guard_count < 5:
                    Int4ConvRotOps._dtype_guard_count += 1
                    logging.error(
                        f"INT4 ConvRot DTYPE GUARD: layer '{self._debug_name}' weight is "
                        f"{weight.dtype}, NOT torch.int8, at forward time (self.weight.dtype="
                        f"{self.weight.dtype}, need_cast={need_cast}). This layer was loaded as "
                        f"packed int4 -- if the dtype changed before forward, unpack_int4_row_major's "
                        f"bitwise ops will silently compute garbage from what looks like valid small "
                        f"integer values (no crash, no NaN, just wrong numbers)."
                    )

                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = Int4ConvRotOps.compute_dtype or Int4ConvRotOps._default_compute_dtype(x)

                # If a "Dynamic" lora_mode LoRA is active on this layer, prepare
                # the rotated activation once and reuse it both for the W4A4
                # matmul and the LoRA down-projection (dynamic LoRA additive
                # branch, applies the delta in the SAME rotated-activation
                # basis the main path uses -- see Int4ModelPatcher, which
                # pre-rotates lora_down through the same Hadamard for exactly
                # this reason). Diagnostic debug_weight_only_reference path is
                # intentionally excluded -- it doesn't rotate/quantize activations
                # at all, so a rotated LoRA delta would be inconsistent with it.
                x_rot = None
                orig_shape = None
                if self.lora_patches and not Int4ConvRotOps.debug_weight_only_reference:
                    x_rot, orig_shape = prepare_rotated_activation(
                        x, self._convrot_groupsize, compute_dtype,
                        skip_rotation=Int4ConvRotOps.debug_skip_rotation,
                    )

                if Int4ConvRotOps.debug_weight_only_reference:
                    y = convrot_w4a4_forward_weight_only_reference(
                        x, weight, w_scale, bias,
                        convrot_groupsize=self._convrot_groupsize,
                        compute_dtype=compute_dtype,
                    )
                else:
                    y = convrot_w4a4_forward(
                        x, weight, w_scale, bias,
                        convrot_groupsize=self._convrot_groupsize,
                        compute_dtype=compute_dtype,
                        skip_rotation=Int4ConvRotOps.debug_skip_rotation,
                        x_rot=x_rot,
                        orig_shape=orig_shape,
                    )

                if self.lora_patches and x_rot is not None:
                    y_2d = y.reshape(-1, y.shape[-1])
                    for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                        lD = lora_down.to(x.device, non_blocking=True)
                        lU = lora_up.to(x.device, non_blocking=True)
                        lora_x = F.linear(x_rot.to(lD.dtype), lD)
                        lora_y = F.linear(lora_x, lU)  # [batch, slice_size or full_out]
                        if lora_start is not None:
                            y_2d[:, lora_start:lora_start + lora_size] = (
                                y_2d[:, lora_start:lora_start + lora_size] + lora_y.to(y_2d.dtype)
                            )
                        else:
                            y_2d = y_2d + lora_y.to(y_2d.dtype)
                    y = y_2d.reshape(y.shape)

                if Int4ConvRotOps.debug_nan_check and Int4ConvRotOps._debug_nan_count < Int4ConvRotOps._debug_nan_max_reports:
                    if y.is_cuda:
                        torch.cuda.synchronize(y.device)
                    output_bad = not torch.isfinite(y).all().item()
                    if output_bad:
                        Int4ConvRotOps._debug_nan_count += 1
                        input_bad = not torch.isfinite(x).all().item()
                        logging.error(
                            f"INT4 ConvRot DEBUG [{Int4ConvRotOps._debug_nan_count}]: NaN/Inf output at layer "
                            f"'{self._debug_name}' (input already had NaN/Inf: {input_bad}, "
                            f"weight_function count: {len(self.weight_function)}, "
                            f"bias_function count: {len(self.bias_function)}, "
                            f"comfy_cast_weights: {self.comfy_cast_weights}, need_cast: {need_cast}). "
                            f"x: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                            f"absmax={'N/A (has inf/nan)' if input_bad else x.abs().max().item()} | "
                            f"weight: shape={tuple(weight.shape)} dtype={weight.dtype} | "
                            f"weight_scale: shape={tuple(w_scale.shape) if isinstance(w_scale, torch.Tensor) else None} "
                            f"range=[{w_scale.min().item():.6g}, {w_scale.max().item():.6g}] | "
                            f"convrot_groupsize={self._convrot_groupsize} compute_dtype={compute_dtype} | "
                            f"y finite_frac (post-sync recheck)={torch.isfinite(y).float().mean().item():.4f}"
                        )

                if need_cast:
                    uncast_bias_weight(self, weight, bias, offload_stream)
                return y

        # Pass-through for other layer types.
        class GroupNorm(manual_cast.GroupNorm): pass
        class LayerNorm(manual_cast.LayerNorm): pass
        class Conv2d(manual_cast.Conv2d): pass
        class Conv3d(manual_cast.Conv3d): pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding): pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            raise ValueError(f"unsupported dimensions: {dims}")


# =============================================================================
# Int4ModelPatcher -- Unified LoRA Handling for INT4 ConvRot W4A4
#
# Mirrors INT8ModelPatcher/INT8LowVramPatch in int8_quant.py, adapted for the
# packed-2-per-byte, always-ConvRot-rotated storage format:
#   - "None"/"Stochastic" lora_mode: BAKE the patch into the int4 weight
#     (unpack -> rescale -> un-rotate -> comfy.lora.calculate_weight ->
#     re-rotate -> re-quantize -> re-pack), reusing the ORIGINAL per-row
#     weight_scale (same reuse-not-recompute behavior as int8's
#     quantize_int8(patched_weight_float, scale)).
#   - "Dynamic" lora_mode: leaves the base int4 weight untouched and instead
#     builds a per-module list of (rotated_down, up, start, size) additive
#     branches, applied in Linear.forward on the SAME rotated activation used
#     for the W4A4 matmul (see prepare_rotated_activation / lora_patches).
#
# Aimdo/lowvram deferred-quantization is intentionally NOT handled here yet
# (still out of v1 scope, per the module docstring) -- this patcher assumes
# weights are already materialized on a real device when patched.
# =============================================================================

import inspect
try:
    _int4_prefetch_sig = inspect.signature(comfy.lora.prefetch_prepared_value)
    _INT4_USE_NEW_PREFETCH = len(_int4_prefetch_sig.parameters) == 5
except Exception:
    _INT4_USE_NEW_PREFETCH = False


class Int4LowVramPatch:
    """Lowvram-path callable: dequant -> patch -> re-quantize on the fly as
    ComfyUI streams this layer's weight to the compute device. Only used for
    lora_mode in {"None", "Stochastic"}; "Dynamic" never installs this."""
    is_lowvram_patch = True

    def __init__(self, key, patches, module, lora_mode):
        self.key = key
        self.patches = patches
        self.module = module
        self.lora_mode = lora_mode
        self.prepared_patches = None

    def memory_required(self):
        if not _INT4_USE_NEW_PREFETCH:
            return 0
        counter = [0]
        for patch in self.patches[self.key]:
            comfy.lora.prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, *args, **kwargs):
        if _INT4_USE_NEW_PREFETCH:
            destination = args[0] if len(args) > 0 else kwargs.get("destination")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")
            copy = args[2] if len(args) > 2 else kwargs.get("copy", True)
            commit = args[3] if len(args) > 3 else kwargs.get("commit", True)

            counter = [0]
            prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            if commit:
                self.prepared_patches = prepared_patches
            return prepared_patches
        else:
            allocate_buffer = args[0] if len(args) > 0 else kwargs.get("allocate_buffer")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")

            self.prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], allocate_buffer, stream), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            return self.prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        """weight is the packed int4 [N, K//2] tensor for this layer."""
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
        groupsize = getattr(self.module, "_convrot_groupsize", CONVROT_W4A4_GROUP_SIZE)
        wscale = self.module.weight_scale
        if isinstance(wscale, torch.Tensor):
            wscale = wscale.to(weight.device)

        weight_float = dequantize_convrot_w4a4_weight_full(weight, wscale, groupsize, dtype=torch.float32)

        patched_weight_float = comfy.lora.calculate_weight(
            patches, weight_float, self.key, intermediate_dtype=weight_float.dtype,
        )

        # Re-rotate happens INSIDE quantize_convrot_w4a4_weight for the
        # "recompute scale" path -- but here we reuse the ORIGINAL scale
        # (mirrors int8's LowVramPatch), so we rotate explicitly and pack
        # with the fixed scale instead of calling quantize_convrot_w4a4_weight
        # (which would both re-rotate AND recompute the scale).
        H = build_hadamard(groupsize, device=patched_weight_float.device, dtype=patched_weight_float.dtype)
        patched_rotated = rotate_weight(patched_weight_float, H, group_size=groupsize)

        if self.lora_mode == "Stochastic":
            return stochastic_round_int4_packed_delta(
                patched_rotated, wscale, seed=comfy.utils.string_to_seed(self.key),
            )
        return quantize_signed_int4_packed_with_scale(patched_rotated, wscale)


class Int4ModelPatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher that intercepts patching for INT4 ConvRot W4A4
    layers, routing through a bake-in path (dequant/un-rotate -> patch ->
    re-rotate/re-quantize) or a dynamic additive path, per
    Int4ConvRotOps.dynamic_lora -- same structure as INT8ModelPatcher."""

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches and not force_cast:
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

        module_path = key.rsplit('.', 1)[0]
        try:
            module = comfy.utils.get_attr(self.model, module_path)
        except AttributeError:
            module = None

        is_int4_module = hasattr(module, "_is_quantized") and module._is_quantized and getattr(module, "_linear_dtype", None) == "int4"
        patches = self.patches.get(key, [])

        if is_int4_module and Int4ConvRotOps.Linear._is_bias_key(key):
            return comfy.utils.get_attr(self.model, key) if return_weight else None

        if is_int4_module:
            groupsize = getattr(module, "_convrot_groupsize", CONVROT_W4A4_GROUP_SIZE)

            if not Int4ConvRotOps.dynamic_lora:
                # --- BAKE-IN LORA PATH ---
                current_weight = comfy.utils.get_attr(self.model, key)
                wscale = module.weight_scale

                if device_to is None:
                    device_to = current_weight.device

                if key not in self.backup:
                    import collections
                    BackupEntry = collections.namedtuple('Dimension', ['weight', 'inplace_update'])
                    self.backup[key] = BackupEntry(
                        weight=current_weight.to(device=self.offload_device, copy=inplace_update),
                        inplace_update=inplace_update,
                    )
                    source_weight = current_weight
                else:
                    source_weight = self.backup[key].weight

                wscale_dev = wscale.to(device_to) if isinstance(wscale, torch.Tensor) else wscale
                weight_float = dequantize_convrot_w4a4_weight_full(
                    source_weight.to(device_to), wscale_dev, groupsize, dtype=torch.float32
                )

                patches_list = self.patches.get(key, [])
                patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

                H = build_hadamard(groupsize, device=device_to, dtype=patched_weight_float.dtype)
                patched_rotated = rotate_weight(patched_weight_float, H, group_size=groupsize)

                if getattr(Int4ConvRotOps, "lora_mode", "None") == "Stochastic":
                    patched_weight_int4 = stochastic_round_int4_packed_delta(patched_rotated, wscale_dev)
                else:
                    patched_weight_int4 = quantize_signed_int4_packed_with_scale(patched_rotated, wscale_dev)

                patched_weight_int4 = patched_weight_int4.to(current_weight.device)

                if return_weight:
                    return patched_weight_int4

                if inplace_update:
                    current_weight.data.copy_(patched_weight_int4)
                else:
                    comfy.utils.set_attr(self.model, key, nn.Parameter(patched_weight_int4, requires_grad=False))
                return

            else:
                # --- DYNAMIC LORA PATH ---
                weight = comfy.utils.get_attr(self.model, key)
                device = weight.device if weight is not None else self.offload_device

                cache_key = tuple(id(p[1]) for p in patches)
                if getattr(module, "_lora_cache_key", None) == cache_key:
                    if return_weight:
                        return comfy.utils.get_attr(self.model, key)
                    return

                lora_patches = []
                for p in patches:
                    strength_patch = p[0]
                    adapter = p[1]
                    strength_model = p[2]
                    offset = p[3] if len(p) > 3 else None

                    if not hasattr(adapter, "weights"):
                        continue

                    strength = strength_patch * strength_model
                    weights = adapter.weights
                    if len(weights) == 6:
                        up, down, alpha, mid, dora, reshape = weights
                        rank = down.shape[0] if down.ndim >= 2 else 1
                        scale = (alpha / rank) * strength if alpha is not None else strength

                        down_scaled = down.flatten(1) * scale
                        if mid is not None:
                            down_scaled = torch.mm(mid.flatten(1), down.flatten(1)) * scale

                        # Int4 ConvRot layers are ALWAYS rotated (that's the
                        # whole point of the format), so unlike int8 (which
                        # gates this on a per-layer _use_convrot flag) this
                        # rotation is unconditional here -- keeps the LoRA
                        # delta coherent with the rotated activation basis
                        # Linear.forward feeds it (see prepare_rotated_activation):
                        #   W_rot = W @ H^T  =>  ΔW_rot = ΔW @ H^T  =>  rotate down only
                        if down_scaled.shape[1] % groupsize == 0:
                            H = build_hadamard(groupsize, device=down_scaled.device, dtype=down_scaled.dtype)
                            down_scaled = rotate_weight(down_scaled, H, group_size=groupsize)
                        else:
                            logging.warning(
                                f"INT4 ConvRot: dynamic LoRA on '{key}' has rank/shape not divisible by "
                                f"convrot_groupsize={groupsize}; skipping rotation, this delta will be wrong."
                            )

                        start, size = None, None
                        if offset is not None:
                            _dim, start, size = offset

                        lora_patches.append((down_scaled.to(device), up.flatten(1).to(device), start, size))

                module.lora_patches = lora_patches
                module._lora_cache_key = cache_key
                if return_weight:
                    return weight
                return

        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

    def load(self, *args, **kwargs):
        if not Int4ConvRotOps.dynamic_lora:
            for k in list(self.backup):
                if k in self.patches:
                    try:
                        module = comfy.utils.get_attr(self.model, k.rsplit('.', 1)[0])
                    except AttributeError:
                        module = None
                    if hasattr(module, "_is_quantized") and module._is_quantized and getattr(module, "_linear_dtype", None) == "int4":
                        bk = self.backup.pop(k)
                        if bk.inplace_update:
                            dest = comfy.utils.get_attr(self.model, k)
                            dest.data.copy_(bk.weight)
                        else:
                            comfy.utils.set_attr(self.model, k, bk.weight)

        stale_keys = [k for k in self.backup if k not in self.patches]
        for k in stale_keys:
            bk = self.backup.pop(k)
            if bk.inplace_update:
                dest = comfy.utils.get_attr(self.model, k)
                dest.data.copy_(bk.weight)
            else:
                comfy.utils.set_attr(self.model, k, bk.weight)

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_patches") and module.lora_patches:
                if not Int4ConvRotOps.dynamic_lora or (name + ".weight") not in self.patches:
                    module.lora_patches = []

        res = super().load(*args, **kwargs) if hasattr(super(), "load") else None

        device_to = kwargs.get("device_to", args[0] if len(args) > 0 else self.model.device)

        for name, module in self.model.named_modules():
            if hasattr(module, "_is_quantized") and module._is_quantized and getattr(module, "_linear_dtype", None) == "int4":
                weight_key = name + ".weight"

                if weight_key in self.patches:
                    if Int4ConvRotOps.dynamic_lora:
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = None
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]
                        self.patch_weight_to_device(weight_key, device_to=device_to)
                    else:
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]

                        lowvram_patch = Int4LowVramPatch(
                            weight_key,
                            self.patches,
                            module,
                            getattr(Int4ConvRotOps, "lora_mode", "None"),
                        )
                        module.weight_lowvram_function = lowvram_patch

        return res

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        src_cls = self.__class__

        if src_cls is Int4ModelPatcher:
            return super().clone(*args, **kwargs)

        if not issubclass(src_cls, Int4ModelPatcher):
            name = f"Int4_{src_cls.__name__}"
            dynamic_cls = type(name, (Int4ModelPatcher, src_cls), {})
        else:
            dynamic_cls = src_cls

        self.__class__ = dynamic_cls

        if not self.is_dynamic() and getattr(self, "cached_patcher_init", None) is None:
            self.cached_patcher_init = (lambda *a, **kw: self, ())

        n = super().clone(*args, **kwargs)

        disable_dyn = kwargs.get("disable_dynamic", False)
        if len(args) > 0:
            disable_dyn = args[0]

        if disable_dyn and not issubclass(n.__class__, Int4ModelPatcher):
            new_cls = type(f"Int4_{n.__class__.__name__}", (Int4ModelPatcher, n.__class__), {})
            n.__class__ = new_cls

        self.__class__ = src_cls
        return n
