"""Core modules for a small decoder-only Transformer language model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Bias-free linear layer: ``x @ weight.T``."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in and d_out must be positive")

        self.weight = nn.Parameter(
            torch.empty(d_out, d_in, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"expected last dimension {self.weight.shape[1]}, got {x.shape[-1]}"
            )
        return x @ self.weight.T


class RMSNorm(nn.Module):
    """RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"expected last dimension {self.weight.shape[0]}, got {x.shape[-1]}"
            )

        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_float * inverse_rms * self.weight.float()).to(dtype=x.dtype)


class Embedding(nn.Module):
    """Lookup table mapping token IDs to vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have integer dtype")
        return self.weight[token_ids.long()]


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax, computed in float32."""
    x_float = x.float()
    shifted = x_float - x_float.amax(dim=dim, keepdim=True)
    probabilities = shifted.exp()
    probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True)
    return probabilities.to(dtype=x.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional embeddings (RoPE) to the last dimension."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0 or d_k <= 0 or d_k % 2 != 0 or max_seq_len <= 0:
            raise ValueError("theta, even d_k, and max_seq_len must be positive")

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        frequencies = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        )
        angles = positions[:, None] * frequencies[None, :]

        self.d_k = d_k
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected d_k={self.d_k}, got {x.shape[-1]}")
        if token_positions.numel() == 0:
            return x
        if token_positions.min() < 0 or token_positions.max() >= self.cos.shape[0]:
            raise ValueError("token_positions are outside the configured range")

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)


class SwiGLU(nn.Module):
    """SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2((gate * torch.sigmoid(gate)) * self.w3(x))


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal MHA with optional RoPE applied to Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        if rope is not None and rope.d_k != self.head_dim:
            raise ValueError("RoPE d_k must equal the attention head dimension")

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch_size, seq_len, d_model)")

        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        if self.rope is not None:
            positions = torch.arange(seq_len, device=x.device).view(1, 1, seq_len)
            positions = positions.expand(batch_size, self.num_heads, seq_len)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = softmax(scores, dim=-1)
        attention_output = attention_weights @ v
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.out_proj(attention_output)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: x + MHA(RMSNorm(x)), then x + FFN(RMSNorm(x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer returning next-token logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10_000.0,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or num_layers <= 0:
            raise ValueError("vocab_size, context_length, and num_layers must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.context_length = context_length
        self.token_embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    rope=self.rope,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")
        if token_ids.shape[1] > self.context_length:
            raise ValueError(
                f"sequence length exceeds context_length={self.context_length}"
            )

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)

    model = TransformerLM(
        vocab_size=10_000,
        context_length=16,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
    )
    token_ids = torch.randint(0, 10_000, (2, 16))
    logits = model(token_ids)
    logits.mean().backward()

    print("Logits:", logits.shape)
    print("Forward and backward check passed.")
