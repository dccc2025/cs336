"""
_rope_kernel + rope 

cos/sin use pytorch to pre-calculate, and triton only used to rotate

Split head_dim and rotate (for the adajacent elements)

Usage:

if self.rope is not None:
    positions = torch.arange(seq_len, device=x.device).view(1, 1, seq_len)
    positions = positions.expand(batch_size, self.num_heads, seq_len)
    q = self.rope(q, positions)
    k = self.rope(k, positions)

"""


from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_tokens,
    n_heads,
    head_dim,
    stride_xt,
    stride_xh,
    stride_xd, 
    stride_cos_t,
    stride_cos_d,
    stride_sin_t,
    stride_sin_d,
    stride_ot, 
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr # head dim
):
    # one (token, head) = one program
    pid = tl.program_id(axis=0) # 0, 1, ..., n_tokens * n_heads - 1
    token, head = pid // n_heads, pid % n_heads

    if token >= n_tokens:
        return

    # pair indices: 0, 1, ..., head_dim/2-1
    half = head_dim // 2
    offs = tl.arange(0, BLOCK_D)
    mask = offs < half

    # the addr is linear
    x_base = x_ptr + token * stride_xt + head * stride_xh
    o_base = out_ptr + token * stride_ot + head * stride_oh

    x1 = tl.load(x_base + (2 * offs) * stride_xd, mask=mask, other=0.0)
    x2 = tl.load(x_base + (2 * offs + 1) * stride_xd, mask=mask, other=0.0)

    # cos/sin: (seq_len, head_dim/2) or broadcast over heads
    cos = tl.load(
        cos_ptr + token * stride_cos_t + offs * stride_cos_d, mask=mask, other = 0.0
    )

    sin = tl.load(
        sin_ptr + token * stride_sin_t + offs * stride_sin_d, mask=mask, other=0.0
    )

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    tl.store(o_base + (2 * offs) * stride_od, y1, mask=mask)
    tl.store(o_base + (2 * offs + 1) * stride_od, y2, mask=mask)



def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:

    """
    Apply RoPE to x

    x: [bs. nheads, seq_len, head_dim]  -- [B, H, S, D]

    cos, sin: module (S, D/2)
    """
    if x.device.type != "cuda":
        raise ValueError("rope requires CUDA tensors")
    if x.ndim != 4:
        raise ValueError("x must be (batch, n_heads, seq, head_dim)")
    if x.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even")

    
    b, h, s, d = x.shape
    half = d // 2

    if cos.shape != (s, half) or sin.shape != (s, half):
        raise ValueError(
            f"cos/sin must be ({s}, {half}), but got {tuple(cos.shape)}"
        )
    
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    out = torch.empty_like(x)

    # treat B*H as nheads for grid: one program per (token, head) over B*H heads
    x_ = x.reshape(b * h, s, d)
    out_ = out.reshape(b * h, s, d)

    n_tokens = s
    n_heads = b * h

    # reorder to (b*h, s, d)
    x_th = x_.transpose(0, 1).contiguous() # (s, b * h, d)
    out_th = out_.transpose(0, 1).contiguous()

    BLOCK_D = triton.next_power_of_2(half)
    grid = (n_tokens * n_heads,)

    _rope_kernel[grid](
        x_th,
        cos,
        sin,
        out_th,
        n_tokens, 
        n_heads,
        d,
        x_th.stride(0),
        x_th.stride(1),
        x_th.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out_th.stride(0),
        out_th.stride(1),
        out_th.stride(2),
        BLOCK_D=BLOCK_D,
    )

    # (s, B*H, D) -> (B, h, s, d)
    return out_th.transpose(0, 1).reshape(b, h, s, d).contiguous()