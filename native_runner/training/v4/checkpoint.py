"""Actor-critic checkpoint contract shared by V4 IL and PPO."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import uuid

import torch

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


# Model config fields and parameters introduced with recent own actions.
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

# Model config fields and parameters introduced with the derived structural inputs (policy v4.2). They are
# separate zero-start modules, so no existing tensor is widened. The card encoder is shared, so its new
# projection appears under several state-dict names that all end the same way.
STRUCTURAL_CONFIG_FIELDS = frozenset(
    {"tower_spatial_channels", "candidate_tower_feature_dim", "card_spell_target_dim", "child_timing_feature_dim"}
)
STRUCTURAL_PARAMETER_NAMES = frozenset(
    {"spatial_encoder.tower_input.weight", "candidate_encoder.tower_features.weight", "child_encoder.timing.weight"}
)
STRUCTURAL_PARAMETER_SUFFIX = "card_encoder.spell_targets.weight"


def is_structural_parameter(name: str) -> bool:
    return name in STRUCTURAL_PARAMETER_NAMES or name.endswith(STRUCTURAL_PARAMETER_SUFFIX)


def is_recent_action_parameter(name: str) -> bool:
    return name.startswith(RECENT_ACTION_PARAMETER_PREFIXES)


def recent_action_checkpoint_contract(model: UniversalCardPolicyV4) -> dict[str, Any]:
    """The contract a checkpoint saved with recent own actions but before the structural inputs carries."""

    contract = checkpoint_contract(model)
    contract["policy_version"] = RECENT_ACTION_POLICY_VERSION
    contract["model_config"] = {
        key: value for key, value in contract["model_config"].items() if key not in STRUCTURAL_CONFIG_FIELDS
    }
    return contract


def legacy_checkpoint_contract(model: UniversalCardPolicyV4) -> dict[str, Any]:
    """The contract a checkpoint saved before recent own actions carries."""

    contract = recent_action_checkpoint_contract(model)
    contract["policy_version"] = LEGACY_POLICY_VERSION
    contract["observation_version"] = LEGACY_OBSERVATION_VERSION
    contract["model_config"] = {
        key: value for key, value in contract["model_config"].items() if key not in RECENT_ACTION_CONFIG_FIELDS
    }
    return contract


def _parameter_aliases(model: UniversalCardPolicyV4) -> dict[str, list[str]]:
    """Every state-dict name that refers to the same tensor (shared card encoders)."""

    by_tensor: dict[int, list[str]] = {}
    for name, tensor in (*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)):
        by_tensor.setdefault(id(tensor), []).append(name)
    return {name: names for names in by_tensor.values() for name in names}


def upgrade_legacy_model_state(
    model: UniversalCardPolicyV4, legacy_state: Mapping[str, torch.Tensor], *, recent_actions_present: bool = False
) -> dict[str, torch.Tensor]:
    """Embed an older state dict so the upgraded model computes the same outputs.

    A checkpoint from before recent own actions (``recent_actions_present=False``) lacks both the recent-action
    and the structural modules; one from policy v4.1 lacks only the structural modules. New modules keep their
    initialization; every path from them into existing activations starts at zero (the widened input columns,
    the spatial input channels, the policy skip and the structural projections), so outputs are unchanged and
    gradients still reach the new weights.
    """

    def is_new(name: str) -> bool:
        return is_structural_parameter(name) or (not recent_actions_present and is_recent_action_parameter(name))

    current = model.state_dict()
    aliases = _parameter_aliases(model)
    upgraded: dict[str, torch.Tensor] = {}
    for name, fresh in current.items():
        if name in legacy_state:
            old = legacy_state[name]
            if tuple(old.shape) == tuple(fresh.shape):
                upgraded[name] = old
                continue
            axis = RECENT_ACTION_WIDENED.get(name)
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
) -> dict[str, Any]:
    """Map Adam moments by parameter name; new parameters start without state."""

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
                    axis = RECENT_ACTION_WIDENED.get(names[id(parameter)])
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
) -> str:
    """Save the complete actor, critic, optimizer, and behavior temperatures."""

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
    older = next(
        (
            (contract, recent_actions_present)
            for contract, recent_actions_present in (
                (legacy_checkpoint_contract(model), False),
                (recent_action_checkpoint_contract(model), True),
            )
            if payload.get("contract") == contract
        ),
        None,
    )
    if older is not None:
        older_contract, recent_actions_present = older
        legacy_state = payload["model_state_dict"]
        payload["model_state_dict"] = upgrade_legacy_model_state(
            model, legacy_state, recent_actions_present=recent_actions_present
        )
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            payload["optimizer_state_dict"] = upgrade_legacy_optimizer_state(
                model, optimizer, legacy_state, payload["optimizer_state_dict"]
            )
        payload["contract"] = checkpoint_contract(model)
        payload["upgraded_from_contract"] = older_contract
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
