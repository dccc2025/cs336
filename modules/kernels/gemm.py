"""
gemm: matmul + linear function
"""


from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    """
    Use the Tiling ideas, C = A @ B, as M*K * K*N = M*N

    But, BLOCK_M * BLOCK_N here, each program outputs; 

    offs_m, offs_n means the subscript

    a_ptrs means BLOCK_M * BLOCK_K

    stride_*L find address based on the real memory (non-continue saving)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K) # [0,1,...,BLOCK_K-1]  , current square block `K`

    # when line_num + 1, the mem skips `K` elements, and stride_ak is therefore = 1
    a_ptrs = a_ptr +  offs_m[:, None] * stride_am +  # (BLOCK_M, 1), line offset
                offs_k[None, :] * stride_ak # (1, BLOCK_K);  col offset

    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0
        )

        b = tl.load(
            b_ptrs,
            mask=(offs_k[:,None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0
        )

        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    
    c = acc.to(a_ptr.dtype.element_ty) # ptr -> element
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    tl.store(
        c_ptrs,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N) 
    )

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    c = a @ b, both are contiguous 
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dim mismatch: {a.shape} @ {b.shape}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("matmul requires CUDA tensors")

    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    assert K == b.shape[0], f"matrix a and b have different shape and cannot directly multiply"
    _, N = b.shape

    c = torch.empty((M, N), dtype = a.dtype, device = a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
        BLOCK_K = BLOCK_K
    )

    return c


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    y = x @ weight.T
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2D (d_out, d_in)")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"expected last dim {weight.shape[1]}, got {x.shape[-1]}"
        )
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("linear requires CUDA tensors")


    x_2d = x.contiguous().reshape(-1, x.shape[-1])

    y_2d = matmul(x_2d, weight.contiguous().T)

    return y_2d.reshape(*x.shape[:-1], weight.shape[0])

  