"""Training helpers for the v4.2 structural input weights.

These weights start at zero (bit-identical upgrade), and Adam moves each element by at most about one learning
rate per step. At the PPO learning rate they take hundreds of updates to matter, so the trainer can scale their
steps alone, and it logs their size so a run shows whether the new inputs are being used.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .checkpoint import is_structural_parameter

# Short names for metrics, keyed by the end of the parameter name.
_LABELS = (
    ("tower_input.weight", "tower_planes"),
    ("tower_features.weight", "candidate_lethality"),
    ("child_encoder.timing.weight", "projectile_timing"),
    ("spell_targets.weight", "spell_reach"),
)


def _label(name: str) -> str:
    for suffix, label in _LABELS:
        if name.endswith(suffix):
            return label
    raise ValueError(f"unlabelled structural parameter {name}")


def structural_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Each structural weight once; the shared card encoder's spell weight appears under many names."""

    return tuple((name, parameter) for name, parameter in model.named_parameters() if is_structural_parameter(name))


def structural_weight_rms(model: nn.Module) -> dict[str, float]:
    """Root mean square of each structural weight, as ``structural_rms_<label>`` metrics."""

    with torch.no_grad():
        return {
            f"structural_rms_{_label(name)}": float(parameter.detach().float().square().mean().sqrt())
            for name, parameter in structural_parameters(model)
        }


def scale_structural_steps(
    optimizer: torch.optim.Optimizer, model: nn.Module, scale: float
) -> Sequence[RemovableHandle]:
    """Multiply every optimizer step of the structural weights by ``scale``; other weights are untouched.

    With no weight decay this is the same as giving those weights ``scale`` times the learning rate, without a
    second parameter group, so optimizer state from older checkpoints still loads unchanged.
    """

    if scale <= 0:
        raise ValueError("structural learning-rate scale must be positive")
    if scale == 1.0:
        return ()
    parameters = tuple(parameter for _name, parameter in structural_parameters(model))
    if not parameters:
        raise ValueError("model has no structural parameters to scale")
    before: list[torch.Tensor] = []

    def remember(_optimizer, _args, _kwargs) -> None:
        before[:] = [parameter.detach().clone() for parameter in parameters]

    def stretch(_optimizer, _args, _kwargs) -> None:
        with torch.no_grad():
            for parameter, old in zip(parameters, before, strict=True):
                parameter.add_(parameter - old, alpha=scale - 1.0)
        before.clear()

    return (optimizer.register_step_pre_hook(remember), optimizer.register_step_post_hook(stretch))
