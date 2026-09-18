"""

Ckpt save/load for FSDP + custom AdamW

"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from modules.parallel.fsdp import is_main

def save_ckpt(
    path: str | Path,
    *,
    model: FSDP,
    optimizer: object,
    step: int, 
    model_config: dict
) -> None:
    path = Path(path)

    path.patent.mkdir(parents=True, exist_ok=True)

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True) # rank0_only means only rank 0 keeps the complete weights, save the 1/N mems

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICR, cfg):
        model_state = model.state_dict()

    opt_state = optmizer.state_dict() if hasattr(
        optimizer, "state_dict"
    ) else {}

    if is_main():
        torch.save(
            {
                "step": step,
                "model_config": config,
                "model_state": model_state,
                "optimizer_state": opt_state
            },
            path,
        )
    
    if dist.is_initialized():
        dist.barrier()
    
def load_ckpt(
    path: str | Path,
    *,
    model: FSDP,
    optimizer: object | None = None,
    map_location: str | torch.device = "cpu"
) -> dict:
    path = Path(path)
    
    # due to that my task only hopes to achieve small 50M ckpt, so every rank reads
    # for LLM, should use distributed

    ckpt = torch.load(
        path,
        map_location=map_location,
        weights_only=False
    )

    cfg = FullStateDictConfig(
        offload_to_cpu = True,
        rank0_only = False
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model.load_state_dict(
            ckpt["optimizer_state"]
        )
    
    if dist.is_initialized():
        dist.barrier()
    
    return ckpt

    