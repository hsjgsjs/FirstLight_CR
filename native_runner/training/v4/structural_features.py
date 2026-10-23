"""Inputs the V4.2 policy derives from tensors it already receives.

Nothing here reads the observation: tower planes come from the tower set, spell lethality from the candidate
card rows plus the tower set, and projectile time to impact from child features. Training, evaluation, the
human cache and live play therefore get the same values with no tensorizer, cache or transport change.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import CHILD_FEATURE_NAMES, ModelConfigV4
from .tensors import ActionCandidatesV4, BattleGroupSetV4, TowerSetV4

OWNER_SELF = 0
OWNER_ENEMY = 1
TOWER_KING = 0
TOWER_LEFT_PRINCESS = 1
TOWER_RIGHT_PRINCESS = 2
CANDIDATE_CARD = 0

_TOWER_HITPOINTS_RATIO = 0
_TOWER_HITPOINTS_OVER_5000 = 5
_TOWER_ACTIVE = 2
_CHILD = {name: index for index, name in enumerate(CHILD_FEATURE_NAMES)}
# Native positions advance 20 ticks per second; child velocities are tiles per tick divided by 10.
_TICKS_PER_SECOND = 20.0
_VELOCITY_SCALE = 10.0
_TIME_TO_IMPACT_CAP_SECONDS = 5.0
_RATIO_CAP = 2.0


def _alive(towers: TowerSetV4) -> Tensor:
    return towers.mask & (towers.features[..., _TOWER_ACTIVE] > 0.5) & (towers.features[..., _TOWER_HITPOINTS_RATIO] > 0)


def tower_planes(towers: TowerSetV4, config: ModelConfigV4) -> Tensor:
    """[B,3,H,W]: each standing tower's footprint carrying its hitpoints, own and enemy separately."""

    features = towers.features
    alive = _alive(towers)
    cell_x = torch.arange(config.board_width, device=features.device, dtype=features.dtype) + 0.5
    cell_y = torch.arange(config.board_height, device=features.device, dtype=features.dtype) + 0.5
    extent = towers.extent.to(features.dtype)
    # Same cell rule as the learned rectangle scatter: a cell belongs when its centre lies inside.
    inside_x = (cell_x[None, None, :] >= extent[..., 0:1]) & (cell_x[None, None, :] < extent[..., 2:3])
    inside_y = (cell_y[None, None, :] >= extent[..., 1:2]) & (cell_y[None, None, :] < extent[..., 3:4])
    footprint = (inside_y[..., :, None] & inside_x[..., None, :]) & alive[..., None, None]  # [B,T,H,W]
    footprint = footprint.to(features.dtype)
    own = (towers.owner_type == OWNER_SELF).to(features.dtype)
    enemy = (towers.owner_type == OWNER_ENEMY).to(features.dtype)
    ratio = features[..., _TOWER_HITPOINTS_RATIO]
    hitpoints = features[..., _TOWER_HITPOINTS_OVER_5000]
    weights = torch.stack((own * ratio, enemy * ratio, enemy * hitpoints), dim=1)  # [B,3,T]
    return torch.einsum("bct,bthw->bchw", weights, footprint)


def candidate_tower_features(
    candidates: ActionCandidatesV4,
    towers: TowerSetV4,
    *,
    card_crown_tower_damage: Tensor,
    card_damage_known: Tensor,
    tower_hitpoints_level_11: Tensor,
) -> Tensor:
    """[B,C,6]: what one cast of each candidate card would take off the enemy towers.

    The spell's level-11 crown-tower damage over the tower's level-11 full hitpoints is the fraction of full
    hitpoints it removes; with cards and towers at one level that fraction holds at every level, so it is
    compared with the tower's remaining hitpoints ratio. Abilities and non-spell cards get zeros.
    """

    dtype = towers.features.dtype
    card = candidates.effective_card_vocab_id
    known = card_damage_known[card] & (candidates.variant == CANDIDATE_CARD) & candidates.mask
    damage = card_crown_tower_damage[card].to(dtype) * known.to(dtype)  # [B,C]
    full = tower_hitpoints_level_11[towers.tower_type, towers.tower_troop_type].to(dtype)  # [B,T]
    target = _alive(towers) & (towers.owner_type == OWNER_ENEMY) & (full > 0)
    fraction = damage[:, :, None] / full.clamp_min(1.0)[:, None, :]  # [B,C,T]
    remaining = towers.features[..., _TOWER_HITPOINTS_RATIO].clamp_min(1e-4)[:, None, :]
    usable = target[:, None, :] & (damage[:, :, None] > 0)
    ratio = torch.where(usable, (fraction / remaining).clamp(max=_RATIO_CAP), torch.zeros_like(fraction))
    kills = usable & (fraction >= remaining)

    def kills_type(tower_type: int) -> Tensor:
        return (kills & (towers.tower_type == tower_type)[:, None, :]).any(dim=-1)

    return torch.stack(
        (
            ratio.amax(dim=-1) / _RATIO_CAP,
            kills.any(dim=-1).to(dtype),
            kills_type(TOWER_LEFT_PRINCESS).to(dtype),
            kills_type(TOWER_RIGHT_PRINCESS).to(dtype),
            kills_type(TOWER_KING).to(dtype),
            known.to(dtype),
        ),
        dim=-1,
    )


def child_impact_timing(groups: BattleGroupSetV4, config: ModelConfigV4) -> Tensor:
    """[B,N,2]: seconds until an in-flight projectile reaches its target point (capped at 5 s), and whether known."""

    features = groups.child_features
    in_flight = features[..., _CHILD["projectile_in_flight"]] > 0.5
    target_known = features[..., _CHILD["projectile_target_position_known"]] > 0.5
    target_x = features[..., _CHILD["projectile_target_x_over_board_width"]] * config.board_width
    target_y = features[..., _CHILD["projectile_target_y_over_board_height"]] * config.board_height
    velocity_x = features[..., _CHILD["velocity_x_over_10"]] * _VELOCITY_SCALE
    velocity_y = features[..., _CHILD["velocity_y_over_10"]] * _VELOCITY_SCALE
    speed = torch.sqrt(velocity_x.square() + velocity_y.square())  # tiles per tick
    distance = torch.sqrt(
        (target_x - groups.child_position[..., 0]).square() + (target_y - groups.child_position[..., 1]).square()
    )
    known = groups.child_mask & in_flight & target_known & (speed > 1e-4)
    seconds = distance / speed.clamp_min(1e-4) / _TICKS_PER_SECOND
    timing = (seconds.clamp(max=_TIME_TO_IMPACT_CAP_SECONDS) / _TIME_TO_IMPACT_CAP_SECONDS) * known.to(features.dtype)
    return torch.stack((timing, known.to(features.dtype)), dim=-1)
