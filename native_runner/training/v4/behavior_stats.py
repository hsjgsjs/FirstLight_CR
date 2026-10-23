"""Count the specific habits seen in live play, from combat traces, per model.

Record traces with the round robin (``round_robin.sh --trace``, written to ``pair-*/traces/``) or with
``defense_trace record``, then:

    python -m native_runner.training.v4.behavior_stats evaluations/rr-XXXX [more runs or trace directories]

Measured, for each model over all of its games:

1. Lethal spells left unused: a spell in hand, affordable, whose crown-tower damage at least equals an enemy
   tower's remaining hitpoints. Each such stretch ends with the spell cast, the tower falling some other way,
   or the chance lapsing (spell cycled out, elixir spent, game over). Overtime is reported separately.
2. Spells aimed at towers: spells cast within 3 tiles of a standing enemy tower.
3. Log and Barbarian Barrel against air: casts whose rolling path held only air units.
4. Goblin Barrel defense: towers' hitpoints lost in the 6 s after an enemy Goblin Barrel, and whether a spell
   was cast near its landing point between 1 s before and 4 s after.
5. Tower level check: each tower's full hitpoints over the level-11 value the structural inputs assume
   (1.00 at level 11; 4.09 / 2.56 = 1.60 at level 16 when towers follow the card curve).

Traces carry exact native state; everything here is a statistic, never a model input.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .defense_trace import read_trace

TICKS_PER_SECOND = 20
TILE = 1000.0
OVERTIME_PHASES = frozenset({"overtime", "sudden_death"})
GOBLIN_BARREL = 28000004
ROLLING_SPELLS = {28000011: "滚木", 28000015: "野蛮人滚桶"}
AIMED_RADIUS = 3 * TILE
BARREL_RADIUS = 3 * TILE
BARREL_WINDOW_TICKS = 6 * TICKS_PER_SECOND
TOWER_TROOP_TYPE = {159_000_000: 1, 159_000_001: 2, 159_000_002: 3, 159_000_004: 4}


@lru_cache(maxsize=1)
def _game_data() -> dict[str, Any]:
    from ...card_specs import build_card_catalog
    from ...competitive_data import load_competitive_data
    from .structural_tables import compiled_spell_facts, compiled_tower_hitpoints_level_11

    logic = load_competitive_data("card_logic")
    nodes = logic["nodes"]

    def record(node_id: str) -> Mapping[str, Any]:
        node = nodes.get(node_id)
        return {} if node is None else node["effective_record"]

    air_cards = set()
    for card in logic["cards"]:
        summoned = record(card["root_node_id"]).get("SummonCharacter")
        if isinstance(summoned, str) and (record(f"CHARACTER.{summoned}").get("FlyingHeight") or 0) > 0:
            air_cards.add(int(card["card_id"]))
    rolling = {}
    for card_id, node_id in ((28000011, "PROJECTILE.LogProjectileRolling"), (28000015, "PROJECTILE.BarbLogProjectileRolling")):
        values = record(node_id)
        rolling[card_id] = (float(values["ProjectileRadius"]), float(values["ProjectileRange"]))
    catalog = build_card_catalog().by_id
    return {
        "spells": compiled_spell_facts(),
        "cost": {card_id: float(spec.elixir_cost or 0.0) for card_id, spec in catalog.items()},
        "towers": compiled_tower_hitpoints_level_11(),
        "air_cards": air_cards,
        "rolling": rolling,
    }


def tower_full_hitpoints_level_11(tower: Mapping[str, Any]) -> float:
    table = _game_data()["towers"]
    troop = TOWER_TROOP_TYPE.get(tower.get("troop"), 0)
    return float(table[0 if tower["kind"] == "king" else 1][troop])


def _standing(tower: Mapping[str, Any]) -> bool:
    return bool(tower.get("active")) and (tower.get("hp") or 0) > 0 and (tower.get("max_hp") or 0) > 0


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    (ax, ay), (bx, by) = a, b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def lethal_spells(view: Mapping[str, Any]) -> dict[int, set[int]]:
    """Spell in hand -> enemy tower ids one affordable cast would destroy."""

    data = _game_data()
    owner, elixir = view["owner"], float(view.get("elixir", 0.0))
    result: dict[int, set[int]] = {}
    for card in view.get("hand", ()):
        fact = data["spells"].get(card)
        if fact is None or not fact.damage_known or fact.crown_tower_damage <= 0 or data["cost"].get(card, 99) > elixir:
            continue
        for tower in view["towers"]:
            if tower["owner"] == owner or not _standing(tower):
                continue
            fraction = fact.crown_tower_damage / tower_full_hitpoints_level_11(tower)
            if fraction >= tower["hp"] / tower["max_hp"]:
                result.setdefault(card, set()).add(tower["id"])
    return result


def own_plays(view: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        event
        for event in view["events"]
        if event["type"] == "action_executed" and event.get("owner") == view["owner"] and event.get("card")
    ]


@dataclass
class ModelStats:
    games: int = 0
    lethal: Counter = field(default_factory=Counter)  # (overtime, outcome) -> stretches
    lethal_delays: list = field(default_factory=list)  # seconds from chance to cast
    spells: int = 0
    spells_at_towers: Counter = field(default_factory=Counter)  # card -> casts near an enemy tower
    rolling: Counter = field(default_factory=Counter)  # outcome -> casts
    barrels: int = 0
    barrels_answered: int = 0
    barrel_tower_losses: list = field(default_factory=list)
    level_ratios: Counter = field(default_factory=Counter)  # (kind, ratio) -> views


def _lethal_stretches(views: list[Mapping[str, Any]], stats: ModelStats) -> None:
    open_stretch: dict[str, Any] | None = None
    for view in views:
        standing = {tower["id"] for tower in view["towers"] if _standing(tower)}
        plays = own_plays(view)
        if open_stretch is not None:
            cast = next((event for event in plays if event["card"] in open_stretch["spells"]), None)
            if cast is not None:
                positions = [tower["pos"] for tower in view["towers"] if tower["id"] in open_stretch["towers"]]
                aimed = cast.get("pos") is not None and any(
                    _distance(cast["pos"], position) <= AIMED_RADIUS for position in positions
                )
                outcome = "cast_at_tower" if aimed else "cast_elsewhere"
                stats.lethal[(open_stretch["overtime"], outcome)] += 1
                if aimed:
                    stats.lethal_delays.append((cast["t"] - open_stretch["tick"]) / TICKS_PER_SECOND)
                open_stretch = None
            elif not open_stretch["towers"] & standing:
                stats.lethal[(open_stretch["overtime"], "tower_fell_otherwise")] += 1
                open_stretch = None
        chances = lethal_spells(view)
        if open_stretch is not None and not chances:
            stats.lethal[(open_stretch["overtime"], "lapsed")] += 1
            open_stretch = None
        elif open_stretch is None and chances:
            open_stretch = {
                "tick": view["tick"],
                "spells": set(chances),
                "towers": set().union(*chances.values()),
                "overtime": view.get("phase") in OVERTIME_PHASES,
            }
        elif open_stretch is not None:
            open_stretch["spells"] |= set(chances)
            open_stretch["towers"] |= set().union(*chances.values())
    if open_stretch is not None:
        stats.lethal[(open_stretch["overtime"], "lapsed")] += 1


def _spell_and_rolling_casts(views: list[Mapping[str, Any]], stats: ModelStats) -> None:
    data = _game_data()
    for view in views:
        owner = view["owner"]
        kings = {tower["owner"]: tower["pos"] for tower in view["towers"] if tower["kind"] == "king"}
        for event in own_plays(view):
            card = event["card"]
            if card not in data["spells"] or event.get("pos") is None:
                continue
            stats.spells += 1
            if any(
                tower["owner"] != owner and _standing(tower) and _distance(event["pos"], tower["pos"]) <= AIMED_RADIUS
                for tower in view["towers"]
            ):
                stats.spells_at_towers[card] += 1
            if card in data["rolling"] and owner in kings and 1 - owner in kings:
                half_width, length = data["rolling"][card]
                direction = 1.0 if kings[1 - owner][1] > kings[owner][1] else -1.0
                x, y = event["pos"]
                inside = [
                    item
                    for item in view["entities"]
                    if item["owner"] not in (owner, None)
                    and item.get("kind") in ("troop", "hero")
                    and abs(item["pos"][0] - x) <= half_width
                    and 0.0 <= (item["pos"][1] - y) * direction <= length
                ]
                air = sum(item["card"] in data["air_cards"] for item in inside)
                if not inside:
                    stats.rolling["empty"] += 1
                elif air == len(inside):
                    stats.rolling["air_only"] += 1
                elif air:
                    stats.rolling["mixed"] += 1
                else:
                    stats.rolling["ground_only"] += 1


def _goblin_barrels(views: list[Mapping[str, Any]], stats: ModelStats) -> None:
    owner = views[0]["owner"] if views else None
    barrels = [
        event
        for view in views
        for event in view["events"]
        if event["type"] == "action_executed"
        and event.get("card") == GOBLIN_BARREL
        and event.get("owner") not in (owner, None)
        and event.get("pos") is not None
    ]
    seen = set()
    for barrel in barrels:
        key = (barrel["t"], tuple(barrel["pos"]))
        if key in seen:
            continue
        seen.add(key)
        stats.barrels += 1
        before = next((view for view in reversed(views) if view["tick"] <= barrel["t"]), views[0])
        window = [view for view in views if barrel["t"] <= view["tick"] <= barrel["t"] + BARREL_WINDOW_TICKS]
        start = {tower["id"]: tower["hp"] or 0 for tower in before["towers"] if tower["owner"] == owner}
        lowest = dict(start)
        for view in window:
            for tower in view["towers"]:
                if tower["id"] in lowest:
                    lowest[tower["id"]] = min(lowest[tower["id"]], tower["hp"] or 0)
        stats.barrel_tower_losses.append(sum(start[key] - lowest[key] for key in start))
        answered = any(
            event["card"] in _game_data()["spells"]
            and event.get("pos") is not None
            and barrel["t"] - TICKS_PER_SECOND <= event["t"] <= barrel["t"] + 4 * TICKS_PER_SECOND
            and _distance(event["pos"], barrel["pos"]) <= BARREL_RADIUS
            for view in views
            for event in own_plays(view)
        )
        stats.barrels_answered += answered


def _level_ratios(views: list[Mapping[str, Any]], stats: ModelStats) -> None:
    if not views:
        return
    for tower in views[0]["towers"]:
        if tower.get("max_hp"):
            ratio = round(tower["max_hp"] / tower_full_hitpoints_level_11(tower), 2)
            stats.level_ratios[(tower["kind"].split("_")[0], ratio)] += 1


def _model_names(trace: Path, header: Mapping[str, Any]) -> dict[int, str]:
    a_owner = int(header.get("a_owner", 0))
    pair = trace.parent.parent / "pair.json"
    if pair.is_file():
        names = json.loads(pair.read_text(encoding="utf-8"))["models"]
    else:
        names = [f"A {header.get('checkpoint_ids', ['?', '?'])[0][:8]}", f"B {header.get('checkpoint_ids', ['?', '?'])[1][:8]}"]
    return {a_owner: names[0], 1 - a_owner: names[1]}


def collect(paths: Iterable[Path]) -> dict[str, ModelStats]:
    results: dict[str, ModelStats] = defaultdict(ModelStats)
    traces = sorted({trace for path in paths for trace in (path.rglob("trace-*.jsonl.gz") if path.is_dir() else [path])})
    for trace in traces:
        header, views, _combat = read_trace(trace)
        names = _model_names(trace, header)
        for owner, name in names.items():
            own = sorted((view for view in views if view["owner"] == owner), key=lambda view: view["tick"])
            if not own:
                continue
            if "hand" not in own[0]:
                raise ValueError(f"{trace} predates hand/elixir in traces; record it again")
            stats = results[name]
            stats.games += 1
            _lethal_stretches(own, stats)
            _spell_and_rolling_casts(own, stats)
            _goblin_barrels(own, stats)
            _level_ratios(own, stats)
    return dict(results)


def _card_name(card_id: int) -> str:
    from ...card_names import CARD_NAMES

    return CARD_NAMES.get(int(card_id), (str(card_id),))[0]


def _share(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def format_report(results: Mapping[str, ModelStats]) -> str:
    lines = []
    for name, stats in sorted(results.items()):
        lines.append(f"== {name}：{stats.games} 局")
        for overtime, label in ((False, "常规时间"), (True, "加时")):
            counts = {outcome: stats.lethal[(overtime, outcome)] for outcome in
                      ("cast_at_tower", "cast_elsewhere", "tower_fell_otherwise", "lapsed")}
            total = sum(counts.values())
            lines.append(
                f"  可斩杀（{label}）：{total} 次；打塔 {_share(counts['cast_at_tower'], total)}，"
                f"打别处 {_share(counts['cast_elsewhere'], total)}，塔被别的打掉 "
                f"{_share(counts['tower_fell_otherwise'], total)}，错过 {_share(counts['lapsed'], total)}"
            )
        if stats.lethal_delays:
            lines.append(f"  可斩杀到出手：中位数 {statistics.median(stats.lethal_delays):.1f} 秒")
        at_towers = sum(stats.spells_at_towers.values())
        top = "、".join(f"{_card_name(card)}×{count}" for card, count in stats.spells_at_towers.most_common(4))
        lines.append(
            f"  法术打塔：每局 {at_towers / max(1, stats.games):.2f} 次，占法术 {_share(at_towers, stats.spells)}"
            + (f"（{top}）" if top else "")
        )
        rolling = sum(stats.rolling.values())
        lines.append(
            f"  滚木/滚桶 {rolling} 次：只有空中单位 {_share(stats.rolling['air_only'], rolling)}，"
            f"空放 {_share(stats.rolling['empty'], rolling)}，混合 {_share(stats.rolling['mixed'], rolling)}，"
            f"只有地面 {_share(stats.rolling['ground_only'], rolling)}"
        )
        if stats.barrels:
            lines.append(
                f"  对方飞桶 {stats.barrels} 次：法术应对 {_share(stats.barrels_answered, stats.barrels)}，"
                f"6 秒内塔掉血平均 {statistics.mean(stats.barrel_tower_losses):.0f}"
            )
        else:
            lines.append("  对方飞桶 0 次")
        ratios = "、".join(f"{kind} {ratio:.2f}×{count}" for (kind, ratio), count in sorted(stats.level_ratios.items()))
        lines.append(f"  塔满血 / 11 级表值：{ratios}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="rr-* run directories, trace directories or trace files")
    parser.add_argument("--json", type=Path, help="also write the raw counts here")
    args = parser.parse_args(argv)
    results = collect(args.paths)
    if not results:
        parser.error("no traces found; run round_robin.sh with --trace")
    if args.json is not None:
        raw = {
            name: {
                "games": stats.games,
                "lethal": {f"{'overtime' if key[0] else 'regular'}:{key[1]}": value for key, value in stats.lethal.items()},
                "lethal_delays": stats.lethal_delays,
                "spells": stats.spells,
                "spells_at_towers": dict(stats.spells_at_towers),
                "rolling": dict(stats.rolling),
                "barrels": stats.barrels,
                "barrels_answered": stats.barrels_answered,
                "barrel_tower_losses": stats.barrel_tower_losses,
                "level_ratios": {f"{kind}:{ratio}": count for (kind, ratio), count in stats.level_ratios.items()},
            }
            for name, stats in results.items()
        }
        args.json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
