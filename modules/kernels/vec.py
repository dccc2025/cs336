""" Elementwise Triton kernels """

from __future__ import annotations

import torch
import triton # allocation in a more high-level (compiling, start, chunking)
import triton.language as tl  # grammar in the single gpu

"""
Note:
triton.jit: compile python functions to GPU kernel
tl: appears in @triton.jit, means what each program do on the GPU
    index: tl.program_id, tl.arnage
    visiting: tl.load, tl.store (ptx)
    cal: tl.sum, tl.exp
    compiling constant: tl.constexpr (like BLOCK_SIZE)
"""

@triton.jit
def _add_kernel(
    x_ptr,  # start ptr
    y_ptr,
    out_ptr,
    n_elements,
    BLOCKSIZE: tl.constexpr # the num each program dealing with the idx
):
    pid = tl.program_id(axis=0) # index of current parallel task
    offsets = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offsets < n_elements # protect the boundary, valid boundary

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    out = x + y; Contiguous CUDA tensors, same shape/dtype
    """

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("add requires CUDA tensors")
    
    x = x.contiguous()
    y = y.contiguous()

    out = torch.empty_like(x)

    n = x.numel() # number of elements
    BLOCKSIZE = 1024

    grid = (triton.cdiv(n, BLOCKSIZE),)

    _add_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCKSIZE = BLOCKSIZE
    )
    return out


@triton.jit
def _mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCKSIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0) 
    offsets = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(out_ptr + offsets, x * y, mask=mask)


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("mul requires CUDA tensors")
    
    x = x.contiguous()
    y = y.contiguous()

    out = torch.empty_like(x)

    n = x.numel() # number of elements
    BLOCKSIZE = 1024

    grid = (triton.cdiv(n, BLOCKSIZE),)

    _mul_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCKSIZE = BLOCKSIZE
    )
    return out


@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCKSIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    out = x * tl.sigmoid(x)

    tl.store(out_ptr + offsets, out, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError("silu requires CUDA tensors")

    x = x.contiguous()
    out = torch.empty_like(x)

    n = x.numel()
    BLOCKSIZE = 1024

    grid = (triton.cdiv(n, BLOCKSIZE),)

    _silu_kernel[grid](
        x,
        out,
        n,
        BLOCKSIZE = BLOCKSIZE
    )

    return out