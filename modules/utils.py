from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None, # rand generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-token prediction pairs from a 1D token array."""
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D array")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")

    num_start_positions = len(tokens) - context_length
    if num_start_positions <= 0:
        raise ValueError("not enough tokens for one input-target pair")

    starts = torch.randint(
        num_start_positions,
        (batch_size,),
        generator=generator,
        device="cpu",
    ).tolist()
    inputs = np.stack([tokens[start : start + context_length] for start in starts])
    targets = np.stack(
        [tokens[start + 1 : start + context_length + 1] for start in starts]
    )

    x = torch.as_tensor(inputs, dtype=torch.long, device=device)
    y = torch.as_tensor(targets, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip the global L2 norm of available gradients in-place."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)

    total_norm = torch.sqrt(
        sum(gradient.float().square().sum() for gradient in gradients)
    )
    scale = max_norm / (total_norm + eps)
    if scale < 1:
        for gradient in gradients:
            gradient.mul_(scale)
    return total_norm
