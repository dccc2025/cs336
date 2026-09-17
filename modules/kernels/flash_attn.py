"""
FlashAttn forward  (causal version)
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B,
    H,
    M, # seq_len for Q
    N, # seq_len for K/V
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # one program = one (batch, head, Q-block)
    