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
    sm_scale, # rsqrt(d_{head})
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
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh

    q_ptrs = q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # scale once; accumulate in fp32
    q = (q * sm_scale).to(tl.float32)

    # for the current Q (line-block)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) # num
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32) #den

    # last visible key index for this Q block (causal) — used only for clarity;
    # the loop still walks all K blocks and relies on causal_mask.
    # split K/V cols into BLOCK_N size block
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = v_ptr + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0).to(tl.float32)

        # q:[BLOCK_M, D] * k: [D, BLOCK_N] -> scores [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)

        # causal + bounds
        q_idx = offs_m[:, None]
        k_idx = offs_n[None, :]

        causal_mask = q_idx >= k_idx
        bound_mask = (q_idx < M) & (k_idx < N)

        qk = tl.where(causal_mask & bound_mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1)) # max logit per line

        p = tl.exp(qk - m_ij[:, None]) # for stable softmax
        p = tl.where(bound_mask & causal_mask, p, 0.0)

        alpha = tl.exp(m_i - m_ij)
        # rows that were all -inf stay at -inf; alpha would be nan — fix them
        alpha = tl.where(m_ij > float("-inf"), alpha, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    l_i = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_i[:, None]

    out_ptrs = out_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=q_mask)


def flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """
    FlashAttention forward.
    q,k,v: (B, H, S, D), same shape, CUDA, D power-of-2 friendly (e.g. 64).
    Only causal=True is implemented in this first kernel.
    """
    if not causal:
        raise NotImplementedError("non-causal flash_attn_fwd comes later")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    if q.ndim != 4:
        raise ValueError("expected (B, H, S, D)")
    if q.device.type != "cuda":
        raise ValueError("flash_attn_fwd requires CUDA")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, H, S, D = q.shape
    if D > 128:
        raise ValueError(f"head_dim={D} too large for this teaching kernel")
    if triton.next_power_of_2(D) != D:
        raise ValueError(f"head_dim={D} must be a power of 2 for this kernel")
    sm_scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    BLOCK_D = D
    grid = (B, H, triton.cdiv(S, BLOCK_M))
    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H,
        S,
        S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


@triton.jit
def _flash_attn_bwd_preprocess_kernel(
    out_ptr, # [B, H, S, D]
    dout_ptr, # dfrac{partial{L}}{partial O]}
    delta_ptr,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_deltab,
    stride_deltah,
    stride_deltam,
    M,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # delta[b,h,m] = sum_d out * dout   (needed for dS = P * (dP - delta))
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_off = pid_b * stride_ob + pid_h * stride_oh
    do_off = pid_b * stride_dob + pid_h * stride_doh

    o_ptrs = out_ptr + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_ptrs = dout_ptr + do_off + offs_m[:, None] * stride_dom +  offs_d[None, :] * stride_dod

    mask = offs_m[:, None] < M

    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    d_ptrs = (
        delta_ptr
        + pid_b * stride_deltab
        + pid_h * stride_deltah
        + offs_m * stride_deltam
    )

    tl.store(d_ptrs, delta, mask=offs_m < M)


@triton.jit
def _flash_attn_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    dout_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
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
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_deltab,
    stride_deltah,
    stride_deltam,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    B,
    H,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """
    One program = one (b, h, k/v-block)

    Wlak Q blocks; recompute P (attn of the small block); accumulate dK, dV; non-repeated dQ
    """

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_off = pid_b * stride_qb + pid_h * stride_qh
    k_off = pid_b * stride_kb + pid_h * stride_kh
    v_off = pid_b * stride_vb + pid_h * stride_vh
    do_off = pid_b * stride_dob + pid_h * stride_doh
    dq_off = pid_b * stride_dqb + pid_h * stride_dqh
    dk_off = pid_b * stride_dkb + pid_h * stride_dkh
    dv_off = pid_b * stride_dvb + pid_h * stride_dvh
    delta_off = pid_b * stride_deltab + pid_h * stride_deltah

    k_ptrs = k_ptr + k_off + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    v_ptrs = v_ptr + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    n_mask = offs_n < N 

    k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32) # [BLOCK_D, BLOCK_N]
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32) # [BLOCK_N, BLOCK_D]

    dk = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # casual: Q rows with m >= start_n can see this K block

    for start_m in range(0, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        m_mask = offs_m < M

        q_ptrs = q_ptr + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = dout_ptr + do_off + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        
        delta_ptrs = delta_ptr + delta_off + offs_m * stride_deltam # linear add

        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        delta = tl.load(delta_ptrs, mask=m_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q * sm_scale, k) # [BLOCK_M, BLOCK_N]
        casual = offs_m[:, None] >= offs_n[None, :]
        bound = m_mask[:, None] & n_mask[None, :]

        qk = tl.where(casual & bound, qk, -float("inf"))


        # recompute P (fwd softmax on this tile) : should save  in the forward process

        p = tl.exp(qk - tl.max(qk, axis=1)[:, None])
        p = tl.where(casual & bound, p, 0.0)
        p = p / tl.maximum(tl.sum(p, axis=1)[:, None], 1e-6)

        dp = tl.dot(do, tl.trans(v))  # [BLOCK_M, BLOCK_N], do is the attn weight after softmax
        ds = p * (dp - delta[:, None])
        ds = tl.where(casual & bound, ds, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * sm_scale

        dq = tl.dot(ds.to(k.dtype), tl.trans(k)) * sm_scale

        dq_ptrs = dq_ptr + dq_off + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
        tl.atomic_add(dq_ptrs, dq, mask=m_mask[:, None]) # add on the original dq addr

    dk_ptrs = dk_ptr + dk_off + offs_n[None, :] * stride_dkn + offs_d[:, None] * stride_dkd
    dv_ptrs = dv_ptr + dv_off + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=n_mask[None, :])
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=n_mask[:, None])



def flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # dq, dk, dv
        
    """
        FlashAttn backward (casual)

        return (dq, dk, dv), same shape as (q, k, v) as [B, H, S, D]

        This dcc hand-writtened version does not implement the tiling, and use O(S^2) recompute P

        So, in the future, replace the Triton kernel...            
    """

    if q.shape != k.shape or q.shape != v.shape or q.shape != out.shape:
        raise ValueError("q, k, v, out must have the same shape")
    if dout.shape != out.shape:
        raise ValueError("dout must match out")
    if q.device.type != "cuda":
        raise ValueError("flash_attn_bwd requires CUDA")
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = out.contiguous()
    dout = dout.contiguous()

    b, h, s, d = q.shape

    scale = 1.0 / math.sqrt(d) # 1/ \sqrt d_head

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    casual = torch.triu(
        torch.ones(s, s, device = q.device, dtype=torch.bool), diagonal = 1
    )

    scores = scores.masked_fill(casual, float("-inf"))  # good writing grammar
    p = torch.softmax(scores, dim=-1)

    # dV = P^T @ dO
    dv = torch.matmul(p.transpose(-2, -1), dout)

    # dP = dO @ V^T
    dp = torch.matmul(dout, v.transpose(-2, -1))

    # dS = P * (dP - sum_row(P * dP))
    delta = (p * dp).sum(dim=-1, keepdim=True)
    ds = p * (dp - delta)

    # dO = dS @ K * scale; dK = dS^T @ Q * scale
    dq = torch.matmul(ds, k) * scale
    dk = torch.matmul(ds.transpose(-2, -1), q) * scale

    return dq, dk, dv


class FlashAttn(torch.autograd.Function): # autograd will graph-calculation and backward
    """
    fwd: Triton flash
    bwd: exact formula
    """

    @staticmethod
    def forward(ctx, q, k, v):
        out = flash_attn_fwd(q, k, v, causal=True)
        ctx.save_for_backward(q, k, v, out) # usage as the name
        return out
    
    @staticmethod
    def backward(ctx, dout):
        q, k, v, out = ctx.saved_tensors
        dq, dk, dv = flash_attn_bwd(q, k, v, out, dout)
        return dq, dk, dv


# encode into the function, equal to class.apply
def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    return FlashAttn.apply(q, k, v)


