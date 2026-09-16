from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy for logits shaped ``(..., vocab_size)``."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match logits except for the vocabulary dimension")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("targets must contain integer token IDs")

    logits_float = logits.float()
    log_normalizer = torch.logsumexp(logits_float, dim=-1)
    target_logits = logits_float.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()
