"""Parameters added to a trained policy: which they are, how long they have trained, and their step size.

New input paths are ``zero_start`` layers, so a checkpoint upgrade leaves the policy's outputs unchanged. Adam
moves each element by at most about one learning rate per step, so at the PPO learning rate such a layer takes
hundreds of updates to matter. The trainer can therefore give these parameters ``scale`` times the learning
rate, fading linearly back to 1 over ``decay_updates`` of each parameter's own training age. Ages travel with the
checkpoints, so a parameter added later gets its own fresh boost while older ones have already settled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .components import zero_start_parameters


def _short(name: str) -> str:
    return name.removesuffix(".weight")


class NewParameterScheduleV4:
    """Training age and learning-rate scale of every zero-start parameter."""

    def __init__(
        self, model: nn.Module, *, scale: float = 1.0, decay_updates: int = 0, ages: Mapping[str, int] | None = None
    ) -> None:
        if scale <= 0:
            raise ValueError("new-parameter learning-rate scale must be positive")
        if decay_updates < 0:
            raise ValueError("new-parameter decay updates cannot be negative")
        self.scale = float(scale)
        self.decay_updates = int(decay_updates)
        self.parameters = zero_start_parameters(model)
        known = ages or {}
        self.ages = {name: max(0, int(known.get(name, 0))) for name, _parameter in self.parameters}

    def scale_for(self, name: str) -> float:
        """``scale`` while untrained, then linearly to 1 at ``decay_updates`` (constant when that is 0)."""

        if self.scale == 1.0 or self.decay_updates == 0:
            return self.scale
        remaining = max(0.0, 1.0 - self.ages[name] / self.decay_updates)
        return 1.0 + (self.scale - 1.0) * remaining

    def attach(self, optimizer: torch.optim.Optimizer) -> Sequence[RemovableHandle]:
        """Stretch each optimizer step of these parameters by their current scale; others are untouched.

        With no weight decay this equals a per-parameter learning rate, without a second parameter group, so
        optimizer state from older checkpoints still loads. Every step through this optimizer is stretched,
        the human-data imitation step included.
        """

        if self.scale == 1.0 or not self.parameters:
            return ()
        before: list[tuple[torch.Tensor, torch.Tensor, float]] = []

        def remember(_optimizer, _args, _kwargs) -> None:
            before[:] = [
                (parameter, parameter.detach().clone(), factor)
                for name, parameter in self.parameters
                if (factor := self.scale_for(name)) != 1.0
            ]

        def stretch(_optimizer, _args, _kwargs) -> None:
            with torch.no_grad():
                for parameter, old, factor in before:
                    parameter.add_(parameter - old, alpha=factor - 1.0)
            before.clear()

        return (optimizer.register_step_pre_hook(remember), optimizer.register_step_post_hook(stretch))

    def advance(self) -> None:
        """One PPO update has trained every parameter."""

        for name in self.ages:
            self.ages[name] += 1

    def state(self) -> dict[str, int]:
        return dict(self.ages)

    def contract(self) -> dict[str, float | int] | None:
        """Resume-contract entry; None at the default so existing strict resumes keep matching."""

        return None if self.scale == 1.0 else {"scale": self.scale, "decay_updates": self.decay_updates}

    def metrics(self) -> dict[str, float]:
        """Root mean square, age and current scale of each new parameter."""

        rows: dict[str, float] = {}
        with torch.no_grad():
            for name, parameter in self.parameters:
                short = _short(name)
                rows[f"new_rms/{short}"] = float(parameter.detach().float().square().mean().sqrt())
                rows[f"new_age/{short}"] = float(self.ages[name])
                rows[f"new_lr_scale/{short}"] = self.scale_for(name)
        return rows
