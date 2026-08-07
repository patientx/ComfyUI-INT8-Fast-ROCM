### AMD SPECIFIC INFO ###

((( This node requires triton package on linux or triton-windows (pip install triton-windows) on windows 10-11. )))

* int4-a4w8 support added via kitchen hijacker. now can use these special int4 quants from Kijai [https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental) even with full lora support, just like int8 models. 

* IMPORTANT "ROCm INT8 compatibility patch"

ComfyUI ships a component (comfy-kitchen) that provides fast INT8-quantized model support. It works great on Nvidia GPUs and on newer AMD GPUs (RDNA3/RDNA4), but on RDNA2 cards (RX 6800/6900 series and similar) and older AMD hardware, its INT8 math path relies on a GPU library feature that isn't implemented for that hardware. The practical result: any INT8-quantized model or VAE (including things like MiniMax-H3's INT8 VAE) fails to load with a HIPBLAS_STATUS_INVALID_VALUE error, even though the rest of the model works fine.

This node pack fixes that by swapping in its own INT8 math kernel — the same one this pack already uses for its INT8 UNet loaders — as a replacement for just that one broken piece. Everything else (loading the model, applying quantization scales, rotation-based quantization, etc.) still runs exactly as ComfyUI normally does it; only the actual GPU multiplication step gets rerouted to hardware that supports it.

This happens automatically in the background as soon as the node pack loads — there's nothing to configure. It detects your GPU and only steps in on hardware known to have the broken path; on Nvidia and on RDNA3/RDNA4 AMD cards it stays out of the way entirely and lets ComfyUI use its normal, native path, which is expected to be faster and better-tested there.

If you ever need to override this behavior — say, a newer or older card behaves unexpectedly — set the environment variable ROCM_INT8_KITCHEN_PATCH before launching ComfyUI:

ROCM_INT8_KITCHEN_PATCH=off — never apply the patch
ROCM_INT8_KITCHEN_PATCH=force — always apply it, regardless of detected GPU

* int4-a4w8 quantized model loading + generation speed improved, DESPITE IT BEING AN INT4 , LOAD THIS WITH INT8 LOADER
* initial int4-a4w8 support (https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors)
* if you get "NaN" or basically black output with a model try forcing weight type to fp32 or bf16.
* int4 model loading support. Works with comfyui's example krea2 model, models made with [Starnodes-ModelConverter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter) and a few other's I came across. Partial lora support. On ltxvideo same error as int8 persist, couldn't solve for now.(no lora support there) On other int4 models USE 2X OR MORE LORA STRENGTH you would use normally with int8 or other models.

* int8 clip saving now available

<img width="550" height="166" alt="image" src="https://github.com/user-attachments/assets/c6bf6dd4-e8f2-4a86-8b33-8900d1e2a781" />

Move the converted clip into models\clip ; it can be loaded via the same "Load Clip int8" node AND comfyui's native "Load Clip" node.

* int8 clip support added
* As of latest patch, models converted and saved with this node would work with both comfyui & this (and bob's original) node.
* The `convert_to_comfy.py` script works, and you can use models created with it in this node as well. So, for AMD users, the short version is: use this node to quantize, load, and run the models. You can also convert those models to a Comfy-native compatible format, allowing you to use them both ways.

<details>
<summary><strong> * Original README * </strong> </summary>
# 🎉 INT8 is now officially supported in ComfyUI 🎉
https://github.com/Comfy-Org/ComfyUI/commit/1a510f04234e5a213d3985a1a54f65652623f4bc

No, I did not help at all with this and had no involvement. My **existing quants are likely to not work** due to a quant naming missmatch, but [silveroxides](https://huggingface.co/silveroxides) are likely to work as they were quite involved in the process of making this happen.

Existing INT8 fast quants can be converted to the proper native format via this script https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/convert_to_comfy.py

```
python convert_comfy_quant.py I8Fast.safetensors I8Comfy.safetensors
or
python convert_comfy_quant.py I8Fast.safetensors --inplace
```

I am glad to retire with a Piña colada in my hands, on the beach. Might slim this node down to an exclusively pre-lora focused node in the future, if that does not become a default comfy feature.

# Comfy INT8 Acceleration

This node speeds up Flux2, Ideogram4, Chroma, Z-Image, Ernie Image in ComfyUI by using INT8 quantization, delivering between 1.5~2x faster inference on my 3090 depending on the model. It should work on any NVIDIA GPU with enough INT8 TOPS. It appears to be faster than FP8 on 40-Series and above as well. 
Works with lora, torch compile.

Further Reading:

[Quality Metrics comparing against MXFP8, FP8, GGUF, etc.](Metrics.md)

[Speed](Speed.md)

[List of Prequantized Checkpoints](Models.md)

---

Updates:

2026-06-06

Fixes for 20-series GPUs

Ensuring proper handling of static weights when dynamic is deactivated

2026-24-05:

RAM usage for lora loading is fixed and on par with base comfy.

RAM usage for model loading is fixed.

Only thing that remains is on the fly quantization will create an extra int8 copy in memory, but it is too much of a hassle to work around. Please rely on swap or pre converted models if this is an issue.

Fixed an issue with loading loras on models that include .bias layers (WAN, LTX2.X) which would cause a OOM error.

2026-15-05:

Bringing back stochastic lora. Some loras appear to need it, others don't, try it if your lora is not working and you don't like pre-lora. TLDR is "sometimes it really helps, sometimes its a little worse". See our measurements [here](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/RAMExp/Metrics.md#some-loras-require-stochastic-lora-to-work).

Attempt at reducing RAM usage

Fixed an issue with Pre-Lora crashing on windows

2026-10-05:

Overhauled the entire lora system. Normal lora loader node works now, no need for specialized lora loaders.

Converted QuaRot to ConvRot, which is a small but free quality gain.

Added Pre-Lora node, which you can connect to the INT8 Model loader to merge loras before utilizing on the fly quantization. 

For more info on quality of convrot, lora approaches see the [Metrics](Metrics.md)

---

# Common GPU related issues:

RTX 20-Series will require you to either use Triton-Windows on windows, triton==3.2.0 or compile triton yourself with SM75 support which was dropped in 3.3.0.

A100 has no possible INT8 Speed-up https://github.com/BobJohnson24/ComfyUI-INT8-Fast/issues/71


## FAQ:

Q: How do I quantize myself?

A: It is not recommended to quantize the human existence. If you would like to quantize a model, see example_workflows/int8_save_convrot_model.json

Q: What is ConvRot?

A: ConvRot is a variant of QuaRot. It basically rotates model weights and activations to eliminate outliers before quantization. This has some inference overhead, but is generally a large quality boost.

Q: What is Pre-Lora?

A: Pre-Lora is a way to merge the lora weights to a BF16 checkpoint within ComfyUI before you quantize the model. This requires an unquantized base model, and enabling on-the-fly quantization. It is generally a higher quality way to apply a lora.

Q: Torch compile takes forever and I hate it

A: Use the torch compile node from [KJ Nodes](https://github.com/kijai/ComfyUI-KJNodes) and ensure you set the disable dynamic VRAM toggle.


# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code during the very early versions of this node, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!

## newgrit1004 for the base ConvRot code we modified into proper ConvRot 
https://github.com/newgrit1004/ComfyUI-ZImage-Triton

## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of multiple months now
</details>
