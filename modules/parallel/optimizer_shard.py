"""
FSP + use_org_params = True => model.params for each rank is sharded.

AdamW (m/v) is optmizer state shard

"""

from __future__ import annotations

from collections.abs import Iterable

import torch
from torch.nn import Module
from modules.optimizer import AdamW

def build_sarded_adamw(
    model: Module,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> AdamW: # `AdamW` is a perfect class
    """
        Construct AdamW on this rank's local params only.

        each rank stores m/v only for its parameter shard.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    if not params:
        raise ValueError("no trainable params on this rank")
    
    return AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def count_local_param_bytes(
    params: Iterable[torch.Tensor] 
) -> int:
    return sum(p.numel() * p.element_size() for p in params)