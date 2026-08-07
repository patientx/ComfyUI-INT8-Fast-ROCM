"""
test_rocm_int8_patch.py

Standalone smoke test for rocm_int8_kitchen_patch.py -- does NOT require
starting ComfyUI. It just needs your portable install's python_env, which
already has torch/triton/comfy_kitchen installed.

SETUP:
    1. Place this script in the SAME folder as rocm_int8_kitchen_patch.py
       and int8_fused_kernel.py (i.e. your ComfyUI-INT8-Fast-ROCM node
       pack folder).

RUN (from that folder, using your portable python_env):
    D:\\comfyui-rocm\\python_env\\python.exe test_rocm_int8_patch.py

    or, if you use the `py` launcher convention:
    py -3 test_rocm_int8_patch.py
    (only works if that resolves to the same python_env -- if unsure,
    use the full python_env\\python.exe path above to be certain you're
    testing against the right interpreter/site-packages.)
"""

import sys
import os

# Make sure this folder's modules (int8_fused_kernel.py, rocm_int8_kitchen_patch.py)
# are importable as plain top-level modules, without needing ComfyUI's
# custom_nodes package machinery.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False in this python_env. "
          "Make sure you're running the portable install's python.exe, "
          "not a system Python.")
    sys.exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")

import rocm_int8_kitchen_patch

rocm_int8_kitchen_patch.smoke_test(device="cuda")
