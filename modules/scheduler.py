"""
Based on the requirements in cs336 [2026 spring],
Cheng Dai only implements the warmup+cosine annealing scheduler
"""

from __future__ import annotations
import math

def get_cosine_scheduler(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_steps: int
) -> float:
    """
    Args:
        step: current training steps, starting from 0

    Return:
        the learning rate current step used
    """

    assert step >= 0, f"step must be non-negative, step: {step}"
    assert max_lr > min_lr and min_lr >= 0, f"require max_lr > min_lr >= 0"
    assert warmup_steps >= 0, f"warmup steps must be non-negative"
    assert cosine_steps > warmup_steps, f"cosine steps must be greater than warmup steps"

    # warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # cosine scheduler
    if step <= cosine_steps:
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (cosine_steps - warmup_steps)))

        return min_lr + cosine_decay * (max_lr - min_lr)

    return min_lr


if __name__ == "__main__":
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = 1000
    cosine_steps = 20000

    checks = {
        0: 0.0,
        warmup_steps: max_lr,
        cosine_steps: min_lr,
        cosine_steps + 100: min_lr,
    }

    for step, expected_lr in checks.items():
        lr = get_cosine_scheduler(
            step, max_lr, min_lr, warmup_steps, cosine_steps
        )
        print(f"step={step:>6}, lr={lr:.8f}")
        assert math.isclose(lr, expected_lr, rel_tol=0.0, abs_tol=1e-12)

    print("Scheduler check passed.")
