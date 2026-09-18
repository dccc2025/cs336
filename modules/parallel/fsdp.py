"""
On this codebase, I only use 2-GPU FSDP helpers for the 50M LLM training
"""

from __future__ import annotations

import os
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.nn import Module
import torch

"""
DDP the gradient sync using `all_reduce`
FSDP splits the params, the forward use all-gather, the backward us reduce-scatter

All the implementations are based on the NCCL(NVIDIA Collective Communications Library)

first combines 2 programs into a nccl (communication group)
"""
def init_distributed() -> tuple[int, int, torch.device]:
    """
        Read the torchrun env (RANK, LOCAL_RANK, WORLD_SIZE), init NCCL
        Returns (rank, world_size, device)
    """

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo" # "gloo" is for cpu communications
        dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_devicelocal_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    return rank, world_size, device



def clearup_distributed() -> None:
    if dist.is_initialized():
        dist.destory_process_group()


def warp_fsdp(
    model: Module,
    *,
    use_fp16: bool = True
) -> FSDP:
    """
        Only save partial params/gradients.optimizer states, instead of save a whole model each card
    """

    mp = None

    if use_fp16 and torch.cuda.is_available():
        mp = MixedPrecision(
            param_type = torch.float16,
            reduce_dtype = torch.float16,
            nuffer_dtype = torch.float16
        )
    
    return FSDP(
        model, 
        sharding_strategy = ShardingStrategy.FULL_SHARD, # split optimizer state, params, grads, (ZeRO-3)
        mixed_precision = mp,
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params = True, # due to self-defined AdamW, keep the org `model params`  
    )


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier() # let all the GPU threads waiting until all the threads reached thisline code, and then continue


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0