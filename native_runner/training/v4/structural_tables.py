"""Per-card and per-tower facts the policy reads directly instead of inferring them.

* How much one cast of a spell takes off a crown tower, as a fraction of that tower's full hitpoints.
* Whether a spell's area reaches air and ground units.
* Each crown tower's full hitpoints.

A spell limited to its caster's side (Log, Barbarian Barrel) only damages a tower its roll can reach: the
Barbarian Barrel rolls 4.5 tiles, so from the river it stops short of the enemy princess towers and its tower
damage is 0. The pocket that opens after a princess tower falls is not modelled.

Every number is derived from the compiled game data (``card_logic``), the same tables the native engine
executes. A balance change therefore reaches the policy through regenerated data, not through relearning a
number memorized from experience.

Card damage and tower hitpoints grow with level along the same curve (``rarity_curves``; every rarity is
aligned so that card level 11 is the tournament standard). In a battle where cards and towers share one level,
the fraction of a tower's full hitpoints that a spell removes is therefore the same at every level, so the
policy never needs to know the level: a spell kills a tower when that fraction reaches the tower's remaining
hitpoints ratio, which is an existing tower input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from ...arena import CELL_UNITS, PlacementRule, deployment_mask, spell_deploys_on_own_side, standard_active_tower_footprints
from ...competitive_data import load_competitive_data
from .catalog import FIRST_REAL_CARD_VOCAB_ID

# Tower rows in the model: tower_type 0 king, 1 left princess, 2 right princess; tower_troop_type 0 none (a
# plain King Tower), 1 Tower Princess, 2 Cannoneer, 3 Dagger Duchess, 4 Royal Chef (tensorizer.py).
TOWER_TYPE_COUNT = 3
TOWER_TROOP_TYPE_COUNT = 5
_KING_BUILDING_BY_TROOP_TYPE = {0: "BUILDING.KingTower", 4: "EXT.ChefTowerKing"}
_SIDE_BUILDING_BY_TROOP_TYPE = {1: "BUILDING.PrincessTower", 2: "BUILDING.Cannoneer", 3: "BUILDING.DaggerDuchess"}

SPELL_TARGET_FEATURE_NAMES = ("hits_air", "hits_ground", "reach_known")


@dataclass(frozen=True, slots=True)
class SpellFactsV1:
    """One cast at card level 11 against a single crown tower."""

    crown_tower_damage: float
    damage_known: bool
    hits_air: bool
    hits_ground: bool
    reach_known: bool


def _record(nodes: Mapping[str, Any], node_id: str) -> Mapping[str, Any] | None:
    node = nodes.get(node_id)
    return None if node is None else node["effective_record"]


def _number(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None or isinstance(value, bool):
        return None
    return float(value)


def _crown_factor(record: Mapping[str, Any]) -> float:
    percent = _number(record, "CrownTowerDamagePercent")
    return max(0.0, 1.0 + (percent or 0.0) / 100.0)


@lru_cache(maxsize=1)
def _level_11_multiplier() -> float:
    curve = load_competitive_data("card_logic")["rarity_curves"]["Common"]
    return float(curve["active_power_level_multipliers_percent"][int(curve["tournament_level_index"])]) / 100.0


class _SpellResolver:
    """Follow a spell's projectiles, area effects and buffs to its damage on one crown tower."""

    def __init__(self, nodes: Mapping[str, Any]) -> None:
        self.nodes = nodes
        self.damage = 0.0
        self.damage_known = True
        self.air = False
        self.ground = False
        # Farthest forward edge, in world units from the cast point, of a projectile that rolls on (Log).
        self.rolling_reach: float | None = None

    def _reach(self, record: Mapping[str, Any], air_key: str, ground_key: str) -> None:
        self.air |= record.get(air_key) is True
        self.ground |= record.get(ground_key) is True

    def projectile(self, name: str, *, depth: int = 0) -> None:
        record = _record(self.nodes, f"PROJECTILE.{name}")
        if record is None or depth > 4:
            self.damage_known = False
            return
        damage = _number(record, "Damage")
        if damage:
            self.damage += damage * _crown_factor(record)
            self._reach(record, "AoeToAir", "AoeToGround")
            roll = _number(record, "ProjectileRange")
            if roll:
                reach = roll + (_number(record, "ProjectileRadiusY") or _number(record, "ProjectileRadius") or 0.0)
                self.rolling_reach = max(self.rolling_reach or 0.0, reach)
        spawned = record.get("SpawnProjectile")
        if isinstance(spawned, str) and spawned:
            self.projectile(spawned, depth=depth + 1)
        elif not damage:
            # A carrier that only spawns units (Goblin Barrel): its area still says what it lands on.
            self._reach(record, "AoeToAir", "AoeToGround")

    def area(self, name: str) -> None:
        record = _record(self.nodes, f"AEO.{name}")
        if record is None:
            self.damage_known = False
            return
        self._reach(record, "HitsAir", "HitsGround")
        if record.get("HitsAir") is not True and record.get("HitsGround") is not True:
            # Targets chosen by native actions (Vines, Void): neither damage nor reach is in the record.
            self.damage_known = False
            return
        if record.get("IgnoreBuildings") is True:
            # Towers are buildings: nothing this area does reaches them.
            return
        hit_speed = _number(record, "HitSpeed")
        if hit_speed is not None and hit_speed < 0:
            # Scripted per-target behaviour (Void, Goblin Curse) is outside this simple model.
            self.damage_known = False
            return
        damage = _number(record, "Damage")
        if damage:
            self.damage += damage * _crown_factor(record)
        projectile = record.get("Projectile")
        if isinstance(projectile, str) and projectile:
            # Lightning: each target is struck once, so a tower takes one hit.
            self.projectile(projectile)
        buff_name = record.get("Buff")
        buff = _record(self.nodes, f"BUFF.{buff_name}") if isinstance(buff_name, str) else None
        if buff is not None:
            per_second = _number(buff, "DamagePerSecond")
            lifetime = _number(record, "LifeDuration")
            if per_second:
                if lifetime is None:
                    self.damage_known = False
                    return
                self.damage += per_second * lifetime / 1000.0 * _crown_factor(buff)


@lru_cache(maxsize=1)
def _own_side_gap_to_enemy_towers() -> float:
    """World units from the front of the caster's territory to the nearest enemy princess tower's front edge."""

    rows = deployment_mask(0, PlacementRule.OWN_TERRITORY_AREA)
    front = (max(y for y, row in enumerate(rows) if any(row)) + 1) * CELL_UNITS
    enemy_front = min(
        tower.center_y_units - tower.height_tiles * CELL_UNITS / 2
        for tower in standard_active_tower_footprints(0)
        if tower.source.startswith("tower:1:princess")
    )
    return enemy_front - front


def resolve_spell_facts(nodes: Mapping[str, Any], card: Mapping[str, Any]) -> SpellFactsV1:
    root = _record(nodes, str(card["root_node_id"])) or {}
    resolver = _SpellResolver(nodes)
    projectile = root.get("Projectile")
    area = root.get("AreaEffectObject")
    if isinstance(projectile, str) and projectile:
        waves = _number(root, "ProjectileWaves") or 1.0
        resolver.projectile(projectile)
        resolver.damage *= waves
    elif isinstance(area, str) and area:
        resolver.area(area)
    else:
        # Spells that only summon (Rage, Heal Spirit, Mirror): no direct damage, reach unknown.
        return SpellFactsV1(0.0, True, False, False, False)
    if (
        resolver.rolling_reach is not None
        and spell_deploys_on_own_side(root)
        and resolver.rolling_reach < _own_side_gap_to_enemy_towers()
    ):
        # Cast from its own side, the roll ends before any enemy tower (Barbarian Barrel).
        resolver.damage = 0.0
    return SpellFactsV1(
        crown_tower_damage=resolver.damage * _level_11_multiplier() if resolver.damage_known else 0.0,
        damage_known=resolver.damage_known,
        hits_air=resolver.air,
        hits_ground=resolver.ground,
        reach_known=resolver.air or resolver.ground,
    )


@lru_cache(maxsize=1)
def compiled_spell_facts() -> Mapping[int, SpellFactsV1]:
    data = load_competitive_data("card_logic")
    nodes = data["nodes"]
    return {
        int(card["card_id"]): resolve_spell_facts(nodes, card) for card in data["cards"] if card["kind"] == "spell"
    }


@lru_cache(maxsize=1)
def compiled_tower_hitpoints_level_11() -> tuple[tuple[float, ...], ...]:
    """Full hitpoints at level 11 by [tower_type][tower_troop_type]; 0 for impossible pairs."""

    nodes = load_competitive_data("card_logic")["nodes"]
    multiplier = _level_11_multiplier()

    def hitpoints(node_id: str) -> float:
        record = _record(nodes, node_id)
        value = None if record is None else _number(record, "Hitpoints")
        if not value:
            raise ValueError(f"compiled data has no hitpoints for {node_id}")
        return value * multiplier

    table = [[0.0] * TOWER_TROOP_TYPE_COUNT for _ in range(TOWER_TYPE_COUNT)]
    for troop, node_id in _KING_BUILDING_BY_TROOP_TYPE.items():
        table[0][troop] = hitpoints(node_id)
    for troop, node_id in _SIDE_BUILDING_BY_TROOP_TYPE.items():
        table[1][troop] = table[2][troop] = hitpoints(node_id)
    return tuple(tuple(row) for row in table)


@dataclass(frozen=True, slots=True)
class StructuralTablesV1:
    """Tensors indexed by card vocabulary row and by tower type/troop."""

    card_crown_tower_damage: Tensor  # [card_vocab] crown-tower damage of one cast at level 11, in hitpoints
    card_damage_known: Tensor  # [card_vocab] bool
    card_spell_targets: Tensor  # [card_vocab, 3] hits_air, hits_ground, reach_known
    tower_hitpoints_level_11: Tensor  # [tower_type, tower_troop_type]

    @classmethod
    def for_card_ids(cls, raw_card_ids: Sequence[int]) -> StructuralTablesV1:
        facts = compiled_spell_facts()
        rows = FIRST_REAL_CARD_VOCAB_ID + len(raw_card_ids)
        damage = torch.zeros(rows, dtype=torch.float32)
        known = torch.zeros(rows, dtype=torch.bool)
        targets = torch.zeros(rows, len(SPELL_TARGET_FEATURE_NAMES), dtype=torch.float32)
        for offset, card_id in enumerate(raw_card_ids):
            fact = facts.get(int(card_id))
            if fact is None:
                continue
            row = FIRST_REAL_CARD_VOCAB_ID + offset
            damage[row] = fact.crown_tower_damage
            known[row] = fact.damage_known
            targets[row] = torch.tensor((float(fact.hits_air), float(fact.hits_ground), float(fact.reach_known)))
        return cls(
            card_crown_tower_damage=damage,
            card_damage_known=known,
            card_spell_targets=targets,
            tower_hitpoints_level_11=torch.tensor(compiled_tower_hitpoints_level_11(), dtype=torch.float32),
        )
