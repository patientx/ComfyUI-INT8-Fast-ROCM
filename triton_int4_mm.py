"""
Experimental packed-INT4 x INT8 DP4A-style Triton GEMM.

A: int8 [M, K] activations (values normally in [-7, 7])
W: int8 [N, K // 2] packed signed INT4 weights
   low nibble = even K column, high nibble = odd K column

Computes int32 C = A @ unpack(W).T without materializing unpack(W).
The packed weight tile is decoded inside the Triton kernel.
"""

import os

import torch
import triton
import triton.language as tl


# Fixed first-pass correctness/performance config.
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def triton_int4_mm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wkp: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # BLOCK_SIZE_K is logical K. Load only BLOCK_SIZE_K/2 packed bytes.
    offs_kp = tl.arange(0, BLOCK_SIZE_K // 2)

    accumulator = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32
    )

    for k0 in range(0, K, BLOCK_SIZE_K):
        kp = (k0 // 2) + offs_kp
        k_even = k0 + offs_kp * 2
        k_odd = k_even + 1

        a_even = tl.load(
            a_ptr
            + offs_m[:, None] * stride_am
            + k_even[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_even[None, :] < K),
            other=0,
        ).to(tl.int8)

        a_odd = tl.load(
            a_ptr
            + offs_m[:, None] * stride_am
            + k_odd[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_odd[None, :] < K),
            other=0,
        ).to(tl.int8)

        # Restore one contiguous logical-K tile for a single DP4A dot.
        a = tl.interleave(a_even, a_odd)

        packed = tl.load(
            w_ptr
            + offs_n[None, :] * stride_wn
            + kp[:, None] * stride_wkp,
            mask=(offs_n[None, :] < N) & (kp[:, None] < (K // 2)),
            other=0,
        ).to(tl.int32)

        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        w_lo = tl.where(lo >= 8, lo - 16, lo).to(tl.int8)
        w_hi = tl.where(hi >= 8, hi - 16, hi).to(tl.int8)

        # interleave() works on the last dimension. Transpose so packed-K is
        # last, interleave low/high nibbles, then transpose back to [K, N].
        w = tl.trans(
            tl.interleave(tl.trans(w_lo), tl.trans(w_hi))
        )

        accumulator = tl.dot(
            a, w, accumulator, out_dtype=tl.int32
        )

    tl.store(
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_int4_mm(
    a: torch.Tensor,
    packed_weight: torch.Tensor,
    out_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """a[M,K] @ unpack(packed_weight[N,K/2]).T -> [M,N]."""
    assert a.ndim == 2
    assert packed_weight.ndim == 2
    assert a.dtype == torch.int8
    assert packed_weight.dtype == torch.int8
    assert a.device == packed_weight.device
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    N, K_packed = packed_weight.shape
    assert K == K_packed * 2, (
        f"K mismatch: activation K={K}, packed weight represents {K_packed * 2}"
    )
    assert K % 2 == 0

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M)
        * triton.cdiv(N, BLOCK_SIZE_N),
    )

    triton_int4_mm_kernel[grid](
        a,
        packed_weight,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return c
