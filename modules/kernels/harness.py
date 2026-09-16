"""
The comparison table with triton and pytorch code
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import time

def max_abs_diff(got: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Return max |got - ref| after checking shape and device
    """
    if got.shape != ref.shape:
        raise ValueError(f"shape mismatch: got {tuple(got.shape)}, but ref {tuple(ref.shape)}")
    
    if got.device != ref.device:
        raise ValueError(f"device mismatch: got {got.device}, but ref {ref.device}")

    return (got.float() - ref.float()).abs().max().item()


def assert_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    name: str = "kernel"
) -> float:
    """
    Compare triton 'got' and torch 'ref', if |got - ref| < atol + rtol * |ref|, then return max error, else raise error
    """

    diff = max_abs_diff(got, ref)
    if not torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol):
        flat = (got.float() - ref.float()).abs().argmax()
        idx = tuple(int(i) for i in torch.unravel_index(flat, got.shape)) # index on each-dim 
        
        raise AssertionError(
            f"{name} mismatch: max_abs_diff={diff:.3e}"
            f"at {idx} got={got.reshape(-1)[flat].item():.6g}"
            f"ref={ref.reshape(-1)[flat].item():.6g}"
            f"(atol={atol}, rtol={rtol})"
        )
    return diff


def bench(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    rep: int = 50,
    sync: bool = True
) -> float:
    """
    Time fn() in ms,

    warmup: discard the runs (need triton/gpu compiling),
    rep: run times
    sync: run `torch.cuda.synchronize()` around timing when cuda is used
    """

    if warmup < 0 or rep <= 0:
        raise ValueError("warmup and rep must be >= 0 ")

    for _ in range(warmup):
        fn()
    
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()

    for _ in range(rep):
        fn()
    
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start

    return (elapsed / rep) * 1e3


def ref_vs_triton(
    name: str,
    ref_fn: Callable[..., torch.Tensor],
    tri_fn: Callable[..., torch.Tensor],
    *args: Any,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    warmup: int = 10,
    rep: int = 50,
    check_grads: bool = False
) -> dict[str, float]:
    """
    Run Pytorch ref vs Triton, assrt_close, then bench both.

    *args are forwarded to both callables (same inputs)

    Returns {"max_abs_diff", "ref_ms", "tri_ms", "spped_up"}
    """

    ref_out = ref_fn(*args)
    tri_out = tri_fn(*args)

    diff = assert_close(tri_out, ref_out, atol=atol, rtol=rtol, name=name)

    if check_grads:
        """Testing when training, validate the gradient is whether valid"""
        if not any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
            raise ValueError("check_grads=True but no arg requires grad")

        ref_loss = ref_out.float().sum()
        tri_loss = tri_out.float().sum()

        ref_loss.backward()

    ref_ms = bench(lambda: ref_fn(*args), warmup=warmup, rep=rep)
    tri_ms = bench(lambda: tri_fn(*args), warmup=warmup, rep=rep)

    speedup = ref_ms / tri_ms if tri_ms > 0 else float("inf")

    print(
        f"{name}: max_abs_diff={diff:.3e} "
        f"ref={ref_ms:.3f}ms, tri={tri_ms:.3f}ms, speedup={speedup:.2f}x",
        flush=True # print in the terminal at once, not wait or the program ends
    )        

    return {
        "max_abs_diff": diff,
        "ref_ms": ref_ms,
        "tri_ms": tri_ms,
        "speed_up": speedup
    }

