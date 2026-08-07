"""
rocm_int8_kitchen_patch.py

Makes int8_convrot / int8_tensorwise checkpoints (e.g. MiniMax-H3's INT8
VAE, or any model loaded through mixed_precision_ops) work on hardware
where comfy-kitchen's INT8 GEMM path is broken, by replacing it with this
pack's own working Triton kernel. Kitchen's path ends in
torch._int_mm -> hipblasLt, confirmed broken on RDNA2 gfx1030
(HIPBLAS_STATUS_INVALID_VALUE). By default this patch applies to
everything EXCEPT RDNA3/RDNA4 (which have working WMMA/hipBLASLt INT8
support and should keep using kitchen's native path) -- see
_should_patch() / _KNOWN_GOOD_ARCH_PREFIXES below for the exact logic,
and ROCM_INT8_KITCHEN_PATCH env var for manual override.

HOW DISPATCH ACTUALLY WORKS (confirmed by reading the actual installed
comfy_kitchen files, not just PR diffs -- this took two attempts to get
right, see git history / conversation this was built from):

    Path A -- plain int8_tensorwise (UNet, embeddings, int8_convrot VAE
    without W4A8), via _op_int8_linear (a torch.library.custom_op):
        _op_int8_linear
            -> impl = registry.get_implementation("int8_linear", kwargs=kwargs)
            -> impl(**kwargs)
    where `registry` is a singleton imported via
        from comfy_kitchen.registry import registry
    The module-level `int8_linear` function that lives in quantization.py
    is NOT used by this path -- an earlier version of this patch only
    overwrote that function directly and was a silent no-op for this path
    because of it.

    Path B -- W4A8 (comfy_kitchen/backends/eager/w4a8_int8.py):
        w4a8_int8_linear() does
            from .quantization import int8_linear
        at the top of the file, then calls int8_linear(...) as a plain
        function at the bottom. This COMPLETELY BYPASSES the registry --
        it's a direct reference captured at w4a8_int8.py's own import
        time. Reassigning quantization.int8_linear later does not reach
        back and update that already-captured reference; the w4a8_int8
        module's own namespace has to be patched directly too. This
        matches a known upstream bug: comfy-kitchen issue #92, "eager
        backend advertises int8_linear on MPS but dispatches to
        CUDA-only torch._int_mm".

    This patch handles BOTH paths: it wraps registry.get_implementation
    for Path A, and directly overwrites the `int8_linear` name inside
    the already-imported w4a8_int8 module's namespace for Path B. Every
    other kitchen op name (dequantize_int8_simple,
    quantize_int8_convrot_weight, etc.) passes through untouched in both
    cases.

Kitchen's original int8_linear (the one being replaced) does, in order:
    1. _apply_input_act(x, input_act)          -- elementwise, fine on ROCm
    2. shape check
    3. weight = weight.to(device).contiguous()
    4. weight_scale validated + reshaped
    5. if convrot: rotate x via kitchen's own Hadamard rotation -- fine on
       ROCm, it's a normal fp16/bf16 matmul, not int8
    6. quantize x -> int8                      -- REPLACED (fused into our kernel)
    7. x_int8 @ weight.T via torch._int_mm     -- REPLACED (this is what crashes)
    8. scale back, add bias                    -- REPLACED (fused into our kernel)

rocm_int8_linear() below keeps steps 1-5 exactly as kitchen implements
them (by calling kitchen's own helper functions directly, so behavior for
those steps doesn't change), and replaces steps 6-8 with a single call
into int8_fused_kernel.triton_int8_linear / triton_int8_linear_per_row.

USAGE:
    Import this module once, early, from your node pack's __init__.py
    (after `import torch` is fine, as you already have it):

        from . import rocm_int8_kitchen_patch  # noqa: F401

    Must run before any workflow actually calls int8_linear -- doesn't
    need to run before the checkpoint is *loaded*, just before it's
    *used* (i.e. before the first decode/forward pass), since the patch
    only affects the registry lookup, not weight loading.
"""

import logging
import os

import torch

import comfy_kitchen.backends.eager.quantization as ck_quant
from comfy_kitchen.registry import registry

# The W4A8 eager backend (comfy_kitchen.backends.eager.w4a8_int8) does
# `from .quantization import int8_linear` and calls it as a plain function
# at the bottom of w4a8_int8_linear(), completely bypassing
# registry.get_implementation(). This is a SEPARATE bypass from the one
# registry patching fixes -- reassigning quantization.int8_linear later
# does not change the reference w4a8_int8.py already captured at its own
# import time, so we have to reach into that module's namespace directly
# too. See: https://github.com/Comfy-Org/comfy-kitchen/issues/92
# (eager backend dispatches to CUDA-only torch._int_mm directly).
try:
    import comfy_kitchen.backends.eager.w4a8_int8 as ck_w4a8_eager
except ImportError:
    ck_w4a8_eager = None  # older comfy_kitchen without W4A8 support

try:
    # Normal case: loaded as part of the custom_nodes package by ComfyUI
    from .int8_fused_kernel import triton_int8_linear, triton_int8_linear_per_row
except ImportError:
    # Standalone case: script run directly for testing, with this file's
    # own folder added to sys.path (no relative package context available)
    from int8_fused_kernel import triton_int8_linear, triton_int8_linear_per_row

log = logging.getLogger("comfyui-int8-fast-rocm")

_original_get_implementation = registry.get_implementation


def rocm_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    # Step 1: input activation function, same as kitchen
    x = ck_quant._apply_input_act(x, input_act)

    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    # Step 3-4: weight / weight_scale prep, same validation as kitchen
    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    # Step 5: ConvRot rotation -- reuse kitchen's own implementation untouched.
    # This is a plain float matmul against a fixed Hadamard matrix, not int8,
    # so it already works fine on ROCm and there's no reason to reimplement it.
    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = ck_quant._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = ck_quant._rotate_activation(x, h, convrot_groupsize)

    # Steps 6-8, fused: quantize activation, INT8 GEMM, dequantize, bias.
    # Dispatch to the per-row kernel if weight_scale is per-output-channel,
    # otherwise the scalar (tensorwise) kernel.
    if weight_scale.numel() == 1:
        return triton_int8_linear(
            x, weight, weight_scale, bias=bias, compute_dtype=out_dtype
        )
    else:
        return triton_int8_linear_per_row(
            x, weight, weight_scale, bias=bias, compute_dtype=out_dtype
        )


def _patched_get_implementation(name, *args, **kwargs):
    if name == "int8_linear":
        return rocm_int8_linear
    return _original_get_implementation(name, *args, **kwargs)


def _is_rocm_torch() -> bool:
    # torch.version.hip is a version string on ROCm builds of torch, and
    # None on CUDA builds. torch.cuda.is_available() is True on BOTH,
    # since ROCm builds expose themselves through torch.cuda.* for
    # compatibility, so that alone can't distinguish them.
    return getattr(torch.version, "hip", None) is not None


# RDNA3 (gfx11xx: Navi 31/32/33 dGPUs, Phoenix/Strix APUs) and RDNA4
# (gfx12xx: Navi 44/48) have working hipBLASLt INT8 GEMM and/or kitchen's
# Triton WMMA route, so they should keep using kitchen's native
# implementation rather than this pack's kernel.
#
# Everything else -- RDNA2 (confirmed broken), RDNA1, older GCN, and any
# future/unknown arch string -- defaults to patched, on the assumption
# that hardware without WMMA support has the same or worse gap as RDNA2
# until proven otherwise. Update this list as new hardware gets tested.
_KNOWN_GOOD_ARCH_PREFIXES = (
    "gfx11",  # RDNA3
    "gfx12",  # RDNA4
)


def _detect_gpu_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        raw = torch.cuda.get_device_properties(0).gcnArchName
        # raw looks like "gfx1030:sramecc+:xnack-" -- strip the feature suffix
        return raw.split(":")[0]
    except Exception:
        return None


def _should_patch() -> tuple[bool, str]:
    """Returns (should_patch, reason_string)."""
    override = os.environ.get("ROCM_INT8_KITCHEN_PATCH", "auto").strip().lower()

    if override == "off":
        return False, "ROCM_INT8_KITCHEN_PATCH=off -- patch explicitly disabled"
    if override == "force":
        return True, "ROCM_INT8_KITCHEN_PATCH=force -- patch explicitly forced on, ignoring arch detection"
    if override not in ("auto", ""):
        log.warning(f"[comfyui-int8-fast-rocm] unrecognized ROCM_INT8_KITCHEN_PATCH value {override!r}, "
                     f"falling back to 'auto'")

    if not _is_rocm_torch():
        return False, "non-ROCm torch build (torch.version.hip is None) -- CUDA has working hipBLASLt-equivalent, no patch needed"

    arch = _detect_gpu_arch()
    if arch is None:
        # Can't tell what we're on -- default to patched, same reasoning as
        # the general "unknown arch" case below (safer to assume it needs
        # the patch than to assume it doesn't).
        return True, "could not detect GPU architecture (gcnArchName unavailable) -- defaulting to patched"

    if arch.startswith(_KNOWN_GOOD_ARCH_PREFIXES):
        return False, (f"detected {arch} (RDNA3/RDNA4), which has working WMMA/hipBLASLt INT8 GEMM -- "
                        f"leaving kitchen's native int8_linear untouched. If int8_convrot/int8_tensorwise "
                        f"still fails on this card, set ROCM_INT8_KITCHEN_PATCH=force to try this pack's kernel.")

    return True, f"detected {arch} (not RDNA3/RDNA4) -- patching, this arch is assumed to lack working INT8 GEMM until confirmed otherwise"


def _apply_patch():
    should_patch, reason = _should_patch()
    if not should_patch:
        log.info(f"[comfyui-int8-fast-rocm] {reason}")
        return

    # Patch point 1: the registry-based dispatch path, used by
    # int8_tensorwise / mixed_precision_ops.Linear (UNet, embeddings,
    # int8_convrot VAE without W4A8).
    registry.get_implementation = _patched_get_implementation

    # Patch point 2: the W4A8 eager backend's direct function call, which
    # bypasses the registry entirely (see comment above the ck_w4a8_eager
    # import at the top of this file). Also overwrite the reference on the
    # origin quantization module itself for completeness / any other code
    # that might import it directly the same way in the future.
    ck_quant.int8_linear = rocm_int8_linear
    if ck_w4a8_eager is not None:
        ck_w4a8_eager.int8_linear = rocm_int8_linear

    log.info(f"[comfyui-int8-fast-rocm] {reason} -- patched registry.get_implementation('int8_linear') "
             f"and the W4A8 eager backend's direct int8_linear reference -> rocm_int8_linear (Triton GEMM)")


_apply_patch()


# -----------------------------------------------------------------------
# Smoke test -- run this manually to confirm the patch is actually hit,
# via the SAME resolution path real code uses (registry.get_implementation),
# not just calling rocm_int8_linear directly (that would prove nothing
# about whether the patch is wired in correctly).
# -----------------------------------------------------------------------
def smoke_test(device="cuda"):
    should_patch, reason = _should_patch()
    if not should_patch:
        print(f"[comfyui-int8-fast-rocm] smoke_test skipped -- patch is inactive on this hardware. "
              f"Reason: {reason}")
        print("[comfyui-int8-fast-rocm] Set ROCM_INT8_KITCHEN_PATCH=force before running this script "
              "if you want to test the kernel anyway on this card.")
        return

    torch.manual_seed(0)
    M, K, N = 37, 256, 512  # deliberately not multiples of block sizes

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    weight_int8 = torch.randint(-127, 127, (N, K), device=device, dtype=torch.int8)
    weight_scale = torch.tensor(0.01, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    test_kwargs = dict(
        x=x, weight=weight_int8, weight_scale=weight_scale, bias=bias,
        out_dtype=torch.float16, convrot=False, convrot_groupsize=256,
        input_act=None,
    )
    impl = registry.get_implementation("int8_linear", kwargs=test_kwargs)
    assert impl is rocm_int8_linear, (
        f"registry.get_implementation('int8_linear') returned {impl!r}, "
        f"NOT rocm_int8_linear -- patch did not take effect."
    )
    out = impl(**test_kwargs)
    assert out.shape == (M, N), f"unexpected output shape {out.shape}"
    assert not torch.isnan(out).any(), "NaNs in output -- something's wrong"
    print(f"[comfyui-int8-fast-rocm] smoke_test OK (dispatch confirmed via registry), "
          f"output shape {tuple(out.shape)}, sample values: {out.flatten()[:5].tolist()}")

    test_kwargs["convrot"] = True
    out_rot = impl(**test_kwargs)
    assert out_rot.shape == (M, N)
    assert not torch.isnan(out_rot).any()
    print(f"[comfyui-int8-fast-rocm] smoke_test (convrot=True) OK, output shape {tuple(out_rot.shape)}")

    # --- Path B: W4A8 eager backend, exercised end-to-end (quantize -> linear) ---
    if ck_w4a8_eager is None:
        print("[comfyui-int8-fast-rocm] W4A8 smoke_test skipped -- comfy_kitchen.backends.eager.w4a8_int8 "
              "not present in this comfy_kitchen install (older version without W4A8 support).")
        return

    assert ck_w4a8_eager.int8_linear is rocm_int8_linear, (
        "comfy_kitchen.backends.eager.w4a8_int8.int8_linear was NOT patched -- "
        "Path B (W4A8) is still calling the original, broken implementation."
    )

    group_size, convrot_groupsize = 16, 256
    Kw = 512  # must be divisible by 16, group_size, and convrot_groupsize
    fp_weight = torch.randn(N, Kw, device=device, dtype=torch.float16)
    qdata, s_rel, s_channel, correction, codebook = ck_w4a8_eager.quantize_w4a8_int8_weight(
        fp_weight, group_size=group_size, convrot_groupsize=convrot_groupsize,
    )
    x_w4a8 = torch.randn(M, Kw, device=device, dtype=torch.float16)
    out_w4a8 = ck_w4a8_eager.w4a8_int8_linear(
        x_w4a8, qdata, s_rel, s_channel,
        codebook=codebook, correction=correction, bias=None,
        group_size=group_size, convrot_groupsize=convrot_groupsize,
        out_dtype=torch.float16,
    )
    assert out_w4a8.shape == (M, N), f"unexpected W4A8 output shape {out_w4a8.shape}"
    assert not torch.isnan(out_w4a8).any(), "NaNs in W4A8 output -- something's wrong"
    print(f"[comfyui-int8-fast-rocm] smoke_test W4A8 (Path B) OK, output shape {tuple(out_w4a8.shape)}, "
          f"sample values: {out_w4a8.flatten()[:5].tolist()}")
