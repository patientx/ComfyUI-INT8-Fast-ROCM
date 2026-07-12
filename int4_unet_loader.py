import json
import logging

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_detection
import comfy.lora
import comfy.lora_convert

from .int4_quant import Int4ConvRotOps, Int4ModelPatcher, MODEL_TYPE_EXCLUSIONS, CONVROT_W4A4_GROUP_SIZE


class UNetLoaderINT4ConvRot:
    """
    Load (or on-the-fly quantize into) INT4 ConvRot W4A4 diffusion models,
    using a pure-PyTorch eager fallback -- no comfy_kitchen, no Triton, no
    hipBLASLt. See int4_quant.py module docstring for the full rationale.

    LoRA support (see Int4ModelPatcher / Int4LowVramPatch in int4_quant.py):
      - `pre_lora` (optional PRE_LORA input, same node/type as the INT8
        loader's PreLoraLoader) is baked in AT LOAD TIME, before/alongside
        quantization -- works regardless of `lora_mode`.
      - `lora_mode` controls how LoRAs applied LATER via a grouped-LoRA node
        (add_patches on the returned MODEL) are handled: "None"/"Stochastic"
        bake into the int4 weight (dequant -> patch -> re-quantize, reusing
        the original scale), "Dynamic" applies the LoRA additively at
        inference time instead of touching the base weight.

    v1 scope note: Aimdo dynamic/lowvram deferred-quantization is still not
    implemented (see int4_quant.py docstring) -- LoRA baking here always
    happens against fully-materialized weights.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp16", "bf16", "fp32"], {"tooltip": "INT4 compute dtype for the activation-rotation/quantization/dequantization math. 'default' follows the activation dtype (fp16/bf16), or fp16 for anything else."}),
                "model_type": (["flux2", "z-image", "ideogram4", "chroma", "krea2", "wan", "ltx2",
                                 "qwen", "ernie", "anima", "hidream o1", "boogu"],
                                {"tooltip": "Only used for on-the-fly quantization, to filter precision-sensitive layers."}),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False, "tooltip": "Quantize a higher precision model to INT4 ConvRot W4A4. If the selected model is already convrot_w4a4, keep unchecked."}),
                "convrot_groupsize": ("INT", {"default": CONVROT_W4A4_GROUP_SIZE, "min": 4, "max": 4096, "step": 4, "tooltip": "Only affects on-the-fly quantization. Loaded checkpoints ignore this entirely -- each layer's groupsize is read from its own comfy_quant metadata. Leave at default if you're loading a pre-quantized file."}),
                "lora_mode": (["None", "Stochastic", "Dynamic"], {"default": "None", "tooltip": "Governs LoRAs added later via a grouped-LoRA node. None bakes with normal rounding (default). Stochastic bakes with stochastic int4 rounding, which can occasionally be closer to the full-precision+lora baseline. Dynamic applies LoRA at inference time instead of touching the base weight, which is slower but avoids repeated re-quantization error."}),
                "debug_nan_check": ("BOOLEAN", {"default": False, "tooltip": "Log the first layer whose output contains NaN/Inf, then stop checking. Use this to localize a black/garbage output; leave off otherwise (adds a small per-layer overhead)."}),
                "debug_skip_rotation": ("BOOLEAN", {"default": False, "tooltip": "DIAGNOSTIC ONLY: skip Hadamard rotation on activations. Deliberately produces wrong output if the checkpoint really is rotated -- only useful as an A/B test to isolate whether rotation-handling is the source of an accuracy bug. Leave off for normal use."}),
                "debug_weight_only_reference": ("BOOLEAN", {"default": False, "tooltip": "DIAGNOSTIC ONLY: fully dequantize+un-rotate weight once, then run a plain float linear with NO activation quantization/rotation. Much slower, but isolates whether weight decode itself is correct, independent of the per-forward activation quantization step. Leave off for normal use."}),
            },
            "optional": {
                "pre_lora": ("PRE_LORA",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load INT4 ConvRot W4A4 models with a pure-PyTorch eager fallback (no comfy_kitchen / Triton / hipBLASLt required)."

    def load_unet(self, unet_name, model_type, on_the_fly_quantization, weight_dtype="default", convrot_groupsize=CONVROT_W4A4_GROUP_SIZE,
                  lora_mode="None", debug_nan_check=False, debug_skip_rotation=False, debug_weight_only_reference=False,
                  pre_lora=None):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        # Backward compatibility for workflows saved before lora_mode existed.
        if isinstance(lora_mode, bool):
            lora_mode = "Dynamic" if lora_mode else "None"
        lora_mode = str(lora_mode)
        if lora_mode not in {"None", "Stochastic", "Dynamic"}:
            lora_mode = "None"

        if pre_lora is not None:
            loras_to_load = pre_lora if isinstance(pre_lora, list) else [pre_lora]
        else:
            loras_to_load = []

        model_options = {"custom_operations": Int4ConvRotOps}

        Int4ConvRotOps.excluded_names = MODEL_TYPE_EXCLUSIONS.get(model_type, [])
        Int4ConvRotOps.dynamic_quantize = on_the_fly_quantization
        Int4ConvRotOps.convrot_groupsize = int(convrot_groupsize)
        _dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        Int4ConvRotOps.compute_dtype = _dtype_map.get(str(weight_dtype), None)
        Int4ConvRotOps._is_prequantized = False
        Int4ConvRotOps._logged_otf = False
        Int4ConvRotOps.debug_nan_check = bool(debug_nan_check)
        Int4ConvRotOps._debug_nan_count = 0
        Int4ConvRotOps.debug_skip_rotation = bool(debug_skip_rotation)
        Int4ConvRotOps.debug_weight_only_reference = bool(debug_weight_only_reference)
        Int4ConvRotOps._dtype_guard_count = 0
        Int4ConvRotOps._load_time_dtype_logged = False
        Int4ConvRotOps.lora_mode = lora_mode
        Int4ConvRotOps.dynamic_lora = lora_mode == "Dynamic"
        if hasattr(Int4ConvRotOps, "_logged_otf"):
            delattr(Int4ConvRotOps, "_logged_otf")

        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        # Bridge safetensors header quant metadata into the per-layer
        # .comfy_quant tensors consumed by Int4ConvRotOps.
        bridged_quant_layers = 0
        if metadata and "_quantization_metadata" in metadata:
            try:
                quant_meta = json.loads(metadata["_quantization_metadata"])
                for layer_name, quant_conf in quant_meta.get("layers", {}).items():
                    if quant_conf.get("format") != "convrot_w4a4":
                        continue

                    candidates = (
                        layer_name + ".weight",
                        "model." + layer_name + ".weight",
                        "diffusion_model." + layer_name + ".weight",
                        "model.diffusion_model." + layer_name + ".weight",
                    )
                    weight_key = next((k for k in candidates if k in sd), None)

                    if weight_key is None:
                        suffix = layer_name + ".weight"
                        matches = [k for k in sd if k == suffix or k.endswith("." + suffix)]
                        if len(matches) == 1:
                            weight_key = matches[0]

                    if weight_key is None:
                        logging.warning(
                            f"INT4 ConvRot: metadata bridge could not find weight "
                            f"for quantized layer '{layer_name}'"
                        )
                        continue

                    base_key = weight_key[:-len(".weight")]
                    quant_key = base_key + ".comfy_quant"
                    if quant_key not in sd:
                        payload = json.dumps(
                            quant_conf, separators=(",", ":")
                        ).encode("utf-8")
                        sd[quant_key] = torch.tensor(
                            list(payload), dtype=torch.uint8
                        )
                    bridged_quant_layers += 1

                if bridged_quant_layers:
                    logging.info(
                        f"INT4 ConvRot: bridged safetensors header metadata for "
                        f"{bridged_quant_layers} convrot_w4a4 layers"
                    )
            except Exception as e:
                logging.warning(
                    f"INT4 ConvRot: failed to parse/bridge "
                    f"_quantization_metadata: {e}"
                )

        # Pre-load LoRA(s) so they can be baked in during quantization/load,
        # same key-map-discovery-via-meta-skeleton trick as int8_unet_loader.py.
        Int4ConvRotOps.lora_patches = {}
        if len(loras_to_load) > 0:
            grouped_patches = {}
            for lora in loras_to_load:
                lora_name = lora.get("lora_name", "None")
                lora_strength = lora.get("lora_strength", 1.0)
                if lora_name == "None" or lora_strength == 0:
                    continue

                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                lora_data = comfy.lora_convert.convert_lora(lora_data)

                unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
                m_config = comfy.model_detection.model_config_from_unet(sd, unet_prefix, metadata=metadata)

                if m_config is None and unet_prefix != "":
                    m_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
                    if m_config is not None:
                        unet_prefix = ""

                if m_config is not None:
                    m_config.custom_operations = Int4ConvRotOps
                    Int4ConvRotOps.skeleton_meta_init = True
                    try:
                        skeleton_model = m_config.get_model(sd, unet_prefix)
                    finally:
                        Int4ConvRotOps.skeleton_meta_init = False
                    key_map = comfy.lora.model_lora_keys_unet(skeleton_model, {})

                    patch_dict = comfy.lora.load_lora(lora_data, key_map)

                    def normalize_key(key):
                        if not isinstance(key, str):
                            return key
                        for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                            if key.startswith(p):
                                return key[len(p):]
                        return key

                    for k, v in patch_dict.items():
                        target_key = k
                        offset = None
                        function = None
                        if isinstance(k, tuple):
                            target_key = k[0]
                            if len(k) > 1: offset = k[1]
                            if len(k) > 2: function = k[2]

                        nk = normalize_key(target_key)
                        grouped_patches.setdefault(nk, []).append((v, offset, function, lora_strength))
                else:
                    logging.warning("INT4 ConvRot: could not detect model type for LoRA mapping.")

                del lora_data

            if grouped_patches:
                Int4ConvRotOps.lora_patches = grouped_patches
                logging.info(f"INT4 ConvRot: prepared {len(grouped_patches)} layer patches for baking.")

        try:
            Int4ConvRotOps.applied_lora_patches = set()
            model = comfy.sd.load_diffusion_model_state_dict(
                sd, model_options=model_options, metadata=metadata
            )

            if Int4ConvRotOps.lora_patches:
                unmatched = set(Int4ConvRotOps.lora_patches.keys()) - Int4ConvRotOps.applied_lora_patches
                if unmatched:
                    logging.warning(f"INT4 ConvRot: {len(unmatched)} pre_lora keys were NOT matched:")
                    for k in sorted(unmatched):
                        logging.warning(f"  unmatched: {k}")
        finally:
            Int4ConvRotOps.lora_patches = {}
            if hasattr(Int4ConvRotOps, 'applied_lora_patches'):
                delattr(Int4ConvRotOps, 'applied_lora_patches')

        if on_the_fly_quantization and not Int4ConvRotOps._is_prequantized:
            pass  # quantization happened inline during state_dict loading above
        elif not on_the_fly_quantization and not Int4ConvRotOps._is_prequantized:
            logging.warning(
                "INT4 ConvRot: on_the_fly_quantization is off and no convrot_w4a4 "
                "layers were detected in this checkpoint -- the model loaded at its "
                "original precision instead of INT4."
            )

        # Wrap in the LoRA-aware patcher so a downstream grouped-LoRA node
        # (add_patches) routes through Int4ModelPatcher.patch_weight_to_device
        # instead of ComfyUI's default float-weight-merging path.
        model = Int4ModelPatcher.clone(model)

        # Stash metadata so it survives ModelPatcher.clone() (e.g. an
        # INT4 grouped-LoRA node cloning the patcher downstream) -- the
        # inner model.model object is shared by reference across clones even
        # though clone() builds a fresh patcher, mirroring int8_save.py's
        # _resolve_source_metadata walk.
        model._safetensors_metadata = metadata
        try:
            if model.model is not None:
                model.model._int4_source_metadata = metadata
        except Exception:
            pass

        return (model,)
