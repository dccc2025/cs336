"""
vec.py is elementwise,
reduce.py is concat along a dim first, and then output results
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_rows_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # one program = one row; exp(x - max) / \sum exp(x - row_max) ---> stable implementation of softmax
    row = tl.program_id(axis=0)

    if row >= n_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    offsets = row * n_cols + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))

    x = x.to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = tl.exp(x - row_max)

    x = tl.where(mask, x, 0.0)

    row_sum = tl.sum(x, axis=0)
    out = x / row_sum

    tl.store(out_ptr + offsets, out, mask=mask)


def softmax_rows(x: torch.Tensor) -> torch.Tensor:

    if x.device.type != "cuda":
        raise ValueError("softmax_rows requires CUDA tensors")
    
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")

    x = x.contiguous()
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    BLOCK_SIZE = triton.next_power_of_2(n_cols) # 768 -> 1024

    if BLOCK_SIZE > 2048:
        raise ValueError(
            f"n_cols={n_cols} too large for this kernels; need blocked softmax"
        )
    
    out = torch.empty_like(x)

    _softmax_rows_kernel[(n_rows,)](
        x,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # still, one row = one program

    row = tl.program_id(axis=0)

    if row >= n_rows:
        return
    
    cols = tl.arange(0, BLOCK_SIZE)

    mask = cols < n_cols

    offsets = row * n_cols + cols

    x = tl.load(x_ptr + offsets, mask = mask, other = 0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other = 0.0).to(tl.float32)

    x2 = tl.where(mask, x*x, 0.0)
    mean_sq = tl.sum(x2, axis=0) / n_cols

    inv_rms = tl.rsqrt(mean_sq + eps)

    outputs = x * inv_rms * w

    tl.store(out_ptr + offsets, outputs, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError(
            "x and weight need to be CUDA Tensors"
        )
    
    if x.shape[-1] != weight.numel():
        raise ValueError(
            f"weight size: {weight.numel()} != x cols: {x.shape[-1]}"
        )
    
    if eps <= 0:
        raise ValueError(
            "eps must be positive"
        )
    
    x = x.contiguous()

    weight = weight.contiguous().view(-1) # (1, 768) -> (768, )

    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if BLOCK_SIZE > 2048:
        raise ValueError(f"n_cols={n_cols} too large for this kernel")
    
    out = torch.empty_like(x)
    _rmsnorm_kernel[(n_rows,)](
        x,
        weight, 
        out,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out