import torch
import folder_paths
import comfy.utils
import comfy.lora
import logging

class INT4GroupedLora:
    """
    Stack multiple LoRAs onto an INT4 ConvRot W4A4 model.

    Mirrors INT8GroupedLora (int8_lora.py) exactly -- this node only calls
    model.clone() + add_patches(); the actual LoRA application (bake-in vs
    dynamic, dequant/un-rotate/patch/re-rotate/re-quantize for int4) is
    intercepted by Int4ModelPatcher.patch_weight_to_device in int4_quant.py,
    the same way INT8ModelPatcher intercepts it for the INT8 loader. The
    `lora_mode` chosen on UNetLoaderINT4ConvRot ("None"/"Stochastic"/"Dynamic")
    governs how patches added here get applied.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {}
        }
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_loras"
    CATEGORY = "loaders"
    DESCRIPTION = "Stacks multiple LoRAs onto an INT4 ConvRot W4A4 model. Actual patching is handled by Int4ModelPatcher."

    def apply_loras(self, model, **kwargs):
        model_patcher = model.clone()

        # ComfyUI's ModelPatcher.clone() builds a fresh patcher object and does
        # NOT carry over arbitrary attributes set on the source patcher (e.g.
        # the _safetensors_metadata stash from UNetLoaderINT4ConvRot). Without
        # this, downstream save/export nodes lose the source safetensors
        # metadata (convrot_w4a4 flags, model_type, etc.) and produce a
        # corrupted/unloadable checkpoint. Same fix as INT8GroupedLora.
        for attr in ("_safetensors_metadata", "_int4_source_metadata"):
            if hasattr(model, attr) and not hasattr(model_patcher, attr):
                try:
                    setattr(model_patcher, attr, getattr(model, attr))
                except Exception:
                    pass

        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        applied_loras = []
        for i in range(1, 11):
            name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)

            if name and name != "None" and strength != 0:
                lora_path = folder_paths.get_full_path("loras", name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                patch_dict = comfy.lora.load_lora(lora_data, key_map)
                model_patcher.add_patches(patch_dict, strength)
                applied_loras.append(name)
                del lora_data

        if applied_loras:
            logging.info(f"INT4 ConvRot Grouped LoRA: Stacked {len(applied_loras)} LoRAs: {', '.join(applied_loras)}")

        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT4GroupedLora": INT4GroupedLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT4GroupedLora": "INT4 ConvRot Grouped LoRA",
}
