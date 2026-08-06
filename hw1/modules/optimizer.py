from __future__ import annotations

from collections.abc import Iterable

import torch


class SGD:
    """Minimal SGD optimizer for educational use."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.params = list(params)
        self.lr = lr

    @torch.no_grad()
    def step(self) -> None:
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-self.lr)

    def zero_grad(self) -> None:
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.grad.zero_()


class AdamW:
    """AdamW with decoupled weight decay and serializable optimizer state."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        beta1, beta2 = betas
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0 or weight_decay < 0:
            raise ValueError("eps must be positive and weight_decay non-negative")

        # ``model.parameters()`` is a one-shot generator: materialize it once.
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1

        for parameter in self.params:
            if parameter.grad is None:
                continue

            grad = parameter.grad
            state = self.state.setdefault(
                parameter,
                {"m": torch.zeros_like(parameter), "v": torch.zeros_like(parameter)},
            )
            m, v = state["m"], state["v"]

            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1**self.step_count)
            v_hat = v / (1 - self.beta2**self.step_count)

            if self.weight_decay:
                parameter.mul_(1 - self.lr * self.weight_decay)
            parameter.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self) -> None:
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def state_dict(self) -> dict:
        """Return state in parameter-list order, so a checkpoint can resume."""
        states: list[dict[str, torch.Tensor] | None] = []
        for parameter in self.params:
            state = self.state.get(parameter)
            states.append(
                None
                if state is None
                else {"m": state["m"].clone(), "v": state["v"].clone()}
            )

        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
            "states": states,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        states = state_dict["states"]
        if len(states) != len(self.params):
            raise ValueError("checkpoint parameter count does not match optimizer")

        self.lr = float(state_dict["lr"])
        self.beta1, self.beta2 = state_dict["betas"]
        self.eps = float(state_dict["eps"])
        self.weight_decay = float(state_dict["weight_decay"])
        self.step_count = int(state_dict["step_count"])
        self.state.clear()

        for parameter, saved_state in zip(self.params, states, strict=True):
            if saved_state is None:
                continue
            self.state[parameter] = {
                "m": saved_state["m"].to(parameter).clone(),
                "v": saved_state["v"].to(parameter).clone(),
            }
