"""Actor-critic checkpoint contract shared by V4 IL and PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import uuid

import torch

from .components import zero_start_parameters
from .config import (
    LEGACY_OBSERVATION_VERSION,
    LEGACY_POLICY_VERSION,
    RECENT_ACTION_POLICY_VERSION,
    UNIVERSAL_ACTION_VERSION,
    UNIVERSAL_OBSERVATION_VERSION,
    UNIVERSAL_POLICY_VERSION,
)
from .model import UniversalCardPolicyV4


CHECKPOINT_SCHEMA_V4 = "universal-card-actor-critic-checkpoint.v4"


def _catalog_ids(model: UniversalCardPolicyV4) -> dict[str, str]:
    return {
        "card": model.catalog.catalog_id,
        "ability": model.ability_catalog.catalog_id,
        "entity_archetype": model.entity_archetype_catalog.catalog_id,
        "effect": model.effect_catalog.catalog_id,
        "mechanic_profile": model.mechanic_profile_catalog.catalog_id,
    }


def checkpoint_contract(model: UniversalCardPolicyV4) -> dict[str, Any]:
    """Return architecture/data identities without changing the model."""

    return {
        "policy_version": UNIVERSAL_POLICY_VERSION,
        "observation_version": UNIVERSAL_OBSERVATION_VERSION,
        "action_version": UNIVERSAL_ACTION_VERSION,
        "model_config": asdict(model.config),
        "catalog_ids": _catalog_ids(model),
    }


@dataclass(frozen=True, slots=True)
class PolicyVersionV4:
    """One step in the policy's history: what a checkpoint saved at this version lacks of the next ones.

    To add a version: build its new paths with ``components.zero_start`` (they are recognized automatically),
    add its config fields to ``ModelConfigV4``, raise ``UNIVERSAL_POLICY_VERSION`` and append a row listing
    those fields. Do not widen existing tensors; ``widened`` and ``parameter_prefixes`` exist only for v4.1,
    which predates ``zero_start``.
    """

    policy_version: str
    observation_version: str
    config_fields: frozenset[str] = frozenset()
    widened: Mapping[str, int] = field(default_factory=dict)
    parameter_prefixes: tuple[str, ...] = ()


# Model config fields and parameters introduced with recent own actions (policy v4.1).
RECENT_ACTION_CONFIG_FIELDS = frozenset(
    {
        "max_recent_actions",
        "recent_action_horizon_ticks",
        "recent_action_feature_dim",
        "recent_action_dim",
        "recent_action_spatial_channels",
    }
)
RECENT_ACTION_PARAMETER_PREFIXES = ("recent_action_encoder.", "recent_action_policy_skip.")
# Existing tensors that gained trailing input columns/channels for recent actions.
RECENT_ACTION_WIDENED = {"input_projection.0.weight": 1, "spatial_encoder.input.weight": 1}

# Model config fields introduced with the derived structural inputs (policy v4.2); their parameters are
# zero-start layers.
STRUCTURAL_CONFIG_FIELDS = frozenset(
    {"tower_spatial_channels", "candidate_tower_feature_dim", "card_spell_target_dim", "child_timing_feature_dim"}
)

POLICY_VERSION_HISTORY = (
    PolicyVersionV4(LEGACY_POLICY_VERSION, LEGACY_OBSERVATION_VERSION),
    PolicyVersionV4(
        RECENT_ACTION_POLICY_VERSION,
        UNIVERSAL_OBSERVATION_VERSION,
        RECENT_ACTION_CONFIG_FIELDS,
        widened=RECENT_ACTION_WIDENED,
        parameter_prefixes=RECENT_ACTION_PARAMETER_PREFIXES,
    ),
    PolicyVersionV4(UNIVERSAL_POLICY_VERSION, UNIVERSAL_OBSERVATION_VERSION, STRUCTURAL_CONFIG_FIELDS),
)
if POLICY_VERSION_HISTORY[-1].policy_version != UNIVERSAL_POLICY_VERSION:
    raise RuntimeError("the newest policy version history row must be the current policy version")


def is_recent_action_parameter(name: str) -> bool:
    return name.startswith(RECENT_ACTION_PARAMETER_PREFIXES)


def zero_start_parameter_names(model: UniversalCardPolicyV4) -> frozenset[str]:
    """Every state-dict name of a zero-start parameter, including the shared card encoder's aliases."""

    tagged = {id(parameter) for _name, parameter in zero_start_parameters(model)}
    return frozenset(
        name for name, parameter in model.named_parameters(remove_duplicate=False) if id(parameter) in tagged
    )


def contract_at_version(model: UniversalCardPolicyV4, index: int) -> dict[str, Any]:
    """The contract a checkpoint saved at ``POLICY_VERSION_HISTORY[index]`` carries."""

    if not 0 <= index < len(POLICY_VERSION_HISTORY):
        raise ValueError("policy version index out of range")
    contract = checkpoint_contract(model)
    later = frozenset().union(*(row.config_fields for row in POLICY_VERSION_HISTORY[index + 1 :]))
    contract["policy_version"] = POLICY_VERSION_HISTORY[index].policy_version
    contract["observation_version"] = POLICY_VERSION_HISTORY[index].observation_version
    contract["model_config"] = {key: value for key, value in contract["model_config"].items() if key not in later}
    return contract


def recent_action_checkpoint_contract(model: UniversalCardPolicyV4) -> dict[str, Any]:
    """The contract a checkpoint saved with recent own actions but before the structural inputs carries."""

    return contract_at_version(model, 1)


def legacy_checkpoint_contract(model: UniversalCardPolicyV4) -> dict[str, Any]:
    """The contract a checkpoint saved before recent own actions carries."""

    return contract_at_version(model, 0)


def _added_after(index: int) -> tuple[dict[str, int], tuple[str, ...]]:
    widened: dict[str, int] = {}
    prefixes: list[str] = []
    for row in POLICY_VERSION_HISTORY[index + 1 :]:
        widened.update(row.widened)
        prefixes.extend(row.parameter_prefixes)
    return widened, tuple(prefixes)


def _parameter_aliases(model: UniversalCardPolicyV4) -> dict[str, list[str]]:
    """Every state-dict name that refers to the same tensor (shared card encoders)."""

    by_tensor: dict[int, list[str]] = {}
    for name, tensor in (*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)):
        by_tensor.setdefault(id(tensor), []).append(name)
    return {name: names for names in by_tensor.values() for name in names}


def upgrade_legacy_model_state(
    model: UniversalCardPolicyV4, legacy_state: Mapping[str, torch.Tensor], *, version_index: int
) -> dict[str, torch.Tensor]:
    """Embed a state dict saved at ``POLICY_VERSION_HISTORY[version_index]`` so the model computes the same outputs.

    Parameters added since then keep their initialization: zero-start layers add exact zeros, and v4.1's
    widened input columns, spatial channels and policy skip start at zero, so outputs are unchanged and
    gradients still reach the new weights.
    """

    widened_axes, prefixes = _added_after(version_index)
    zero_start_names = zero_start_parameter_names(model)

    def is_new(name: str) -> bool:
        return name in zero_start_names or name.startswith(prefixes)

    current = model.state_dict()
    aliases = _parameter_aliases(model)
    upgraded: dict[str, torch.Tensor] = {}
    for name, fresh in current.items():
        if name in legacy_state:
            old = legacy_state[name]
            if tuple(old.shape) == tuple(fresh.shape):
                upgraded[name] = old
                continue
            axis = widened_axes.get(name)
            if axis is None or old.ndim != fresh.ndim or old.shape[axis] >= fresh.shape[axis]:
                raise ValueError(f"legacy checkpoint tensor {name} has an incompatible shape")
            widened = torch.zeros_like(fresh, device=old.device, dtype=old.dtype)
            widened.narrow(axis, 0, old.shape[axis]).copy_(old)
            upgraded[name] = widened
            continue
        shared = next((alias for alias in aliases.get(name, ()) if alias in legacy_state), None)
        if shared is not None:
            upgraded[name] = legacy_state[shared]
        elif is_new(name):
            upgraded[name] = fresh
        else:
            raise ValueError(f"legacy checkpoint is missing tensor {name}")
    unexpected = sorted(set(legacy_state).difference(current))
    if unexpected:
        raise ValueError(f"legacy checkpoint has unknown tensors: {unexpected[:3]}")
    return upgraded


def upgrade_legacy_optimizer_state(
    model: UniversalCardPolicyV4,
    optimizer: torch.optim.Optimizer,
    legacy_state: Mapping[str, torch.Tensor],
    legacy_optimizer_state: Mapping[str, Any],
    *,
    version_index: int,
) -> dict[str, Any]:
    """Map Adam moments by parameter name; new parameters start without state."""

    widened_axes, _prefixes = _added_after(version_index)

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    fresh = optimizer.state_dict()
    if len(fresh["param_groups"]) != len(legacy_optimizer_state["param_groups"]):
        raise ValueError("legacy optimizer has a different parameter-group layout")
    state: dict[int, Any] = {}
    groups = []
    for live_group, fresh_group, old_group in zip(
        optimizer.param_groups, fresh["param_groups"], legacy_optimizer_state["param_groups"], strict=True
    ):
        retained = [
            (parameter, new_index)
            for parameter, new_index in zip(live_group["params"], fresh_group["params"], strict=True)
            if names[id(parameter)] in legacy_state
        ]
        if len(retained) != len(old_group["params"]):
            raise ValueError("legacy optimizer parameter count does not match the legacy model")
        for (parameter, new_index), old_index in zip(retained, old_group["params"], strict=True):
            old = legacy_optimizer_state["state"].get(old_index)
            if old is None:
                continue
            moved = {}
            for key, value in old.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and tuple(value.shape) != tuple(parameter.shape):
                    axis = widened_axes.get(names[id(parameter)])
                    if axis is None:
                        raise ValueError(f"legacy optimizer state for {names[id(parameter)]} has the wrong shape")
                    widened = torch.zeros(parameter.shape, dtype=value.dtype, device=value.device)
                    widened.narrow(axis, 0, value.shape[axis]).copy_(value)
                    value = widened
                moved[key] = value
            state[new_index] = moved
        groups.append({**{k: v for k, v in old_group.items() if k != "params"}, "params": fresh_group["params"]})
    return {"state": state, "param_groups": groups}


def save_actor_critic_checkpoint(
    path: str | Path,
    model: UniversalCardPolicyV4,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    grad_scaler: object | None = None,
    update_step: int = 0,
    training_stage: str,
    gamma_per_decision: float,
    gae_lambda: float | None = None,
    extra: Mapping[str, Any] | None = None,
    parameter_ages: Mapping[str, int] | None = None,
) -> str:
    """Save the complete actor, critic, optimizer, and behavior temperatures.

    ``parameter_ages`` (PPO updates each zero-start parameter has trained, see ``new_parameters.py``) is stored
    when given; a checkpoint without it counts those parameters as untrained.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA_V4,
        "checkpoint_id": checkpoint_id,
        "contract": checkpoint_contract(model),
        "model_state_dict": model.state_dict(),
        **{
            f"{name}_state_dict": None if value is None else value.state_dict()
            for name, value in (("optimizer", optimizer), ("scheduler", scheduler), ("grad_scaler", grad_scaler))
        },
        "update_step": int(update_step),
        "training_stage": str(training_stage),
        "gamma_per_decision": float(gamma_per_decision),
        "gae_lambda": None if gae_lambda is None else float(gae_lambda),
        "ppo_gate_temperature": model.ppo_gate_temperature,
        "ppo_action_temperature": model.ppo_action_temperature,
        "ppo_continue_temperature": model.ppo_continue_temperature,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "extra": dict(extra or {}),
    }
    if parameter_ages is not None:
        payload["parameter_ages"] = {str(name): int(age) for name, age in parameter_ages.items()}
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_id


def load_actor_critic_checkpoint(
    path: str | Path,
    model: UniversalCardPolicyV4,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    grad_scaler: object | None = None,
    map_location: torch.device | str | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a strict checkpoint after verifying V4 and catalog identities."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_V4:
        raise ValueError("unsupported V4 actor-critic checkpoint")
    version_index = next(
        (
            index
            for index in range(len(POLICY_VERSION_HISTORY) - 2, -1, -1)
            if payload.get("contract") == contract_at_version(model, index)
        ),
        None,
    )
    if version_index is not None:
        legacy_state = payload["model_state_dict"]
        payload["model_state_dict"] = upgrade_legacy_model_state(model, legacy_state, version_index=version_index)
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            payload["optimizer_state_dict"] = upgrade_legacy_optimizer_state(
                model, optimizer, legacy_state, payload["optimizer_state_dict"], version_index=version_index
            )
        payload["upgraded_from_contract"] = payload["contract"]
        payload["contract"] = checkpoint_contract(model)
    elif payload.get("contract") != checkpoint_contract(model):
        raise ValueError("checkpoint model/catalog contract does not match")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_ppo_gate_temperature(float(payload["ppo_gate_temperature"]))
    model.set_ppo_action_temperature(float(payload["ppo_action_temperature"]))
    model.set_ppo_continue_temperature(float(payload.get("ppo_continue_temperature", 1.0)))

    for name, value in (("optimizer", optimizer), ("scheduler", scheduler), ("grad_scaler", grad_scaler)):
        state = payload.get(f"{name}_state_dict")
        if value is not None and state is not None:
            value.load_state_dict(state)
    if restore_rng:
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_state = payload.get("cuda_rng_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    return payload
