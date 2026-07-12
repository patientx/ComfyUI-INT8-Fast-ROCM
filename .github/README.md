### AMD SPECIFIC INFO ###
The official support is problematic with AMD gpu's atm. It uses either "eager" or "triton" (you can enable this with `--enable-triton-backend` startup parameter on comfyui). Eager doesn't work for RDNA2 unless a patch is made (works for rdna4 at least dunno about rdna3) and same situation with triton backend. And even if you make them work , the performance is worse compared to using this node. Also, the `convert_comfy_quant script.py` works and you can also use models made with that in this node as well, so for AMD in short, use this node to quantize and load and use the models. You can convert those to comfy native compatible format to be able to use in both ways. 

Long story short, at least for AMD this is still relevant until they fix the performance in the comfyui-kitchen for us. (doubt)

* the int4 branch has int4 model loading available every model I've tried works, except there is no lora support atm.
* int8 clip saving now available

<img width="550" height="166" alt="image" src="https://github.com/user-attachments/assets/c6bf6dd4-e8f2-4a86-8b33-8900d1e2a781" />

Move the converted clip into models\clip ; it can be loaded via the same "Load Clip int8" node AND comfyui's native "Load Clip" node.

* int8 clip support added
* As of latest patch, models converted and saved with this node would work with both comfyui & this (and bob's original) node.

-
-
-
-
-
-
-
-
::::: ORIGINAL README ::::

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
