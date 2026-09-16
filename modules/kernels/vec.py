""" Elementwise Triton kernels """

from __future__ import annoations

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
    x_ptr, 
    y_ptr,
    out_ptr,
    n_elements,
    BLOCKSIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    