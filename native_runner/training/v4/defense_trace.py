"""Record compact combat traces from evaluation games and audit who kills enemy units.

Observations report hitpoint drops and disappearances without an attacker (FAIR combat is derived
from public snapshots). The native combat ring, which the environment already consumes, does carry
exact damage and death records with their sources. A redundant-defense metric or an auxiliary
"killed by an existing source" label may use those records (they are labels, never model inputs),
so this tool measures:

1. how many enemy-unit deaths the exact native records attribute to a known attacker, and to what;
2. how well a heuristic from public state alone (unit/tower attack targets, projectile
   source/target/destination, spawner links) recovers the same attacker, for data without the ring.

Record a few round-robin blocks on a resident engine (round_robin.local.json conditions), then audit:
    python -m native_runner.training.v4.defense_trace record --model-a "BC 200" --model-b "General 460" \
        --existing-engine 127.0.0.1:28000
Audit traces again (also those from evaluate --trace-dir):
    python -m native_runner.training.v4.defense_trace analyze evaluations/defense-audit/traces
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TRACE_VERSION = "defense-trace.v1"
TRACE_EVENTS = frozenset(
    {
        "action_executed",
        "action_rejected",
        "spawn",
        "damage",
        "death_or_despawn",
        "tower_damage",
        "tower_destroyed",
        "runtime_ability_activation",
    }
)
# Public-state heuristic: look this many ticks back from a death for the attacker.
HEURISTIC_WINDOW_TICKS = 10
# Area sources (spells, projectile destinations) explain a hit within 2.5 tiles of the victim (native units).
AREA_RADIUS = 2500.0


def _round_pair(value: Any) -> list[float] | None:
    return None if value is None else [round(float(value[0]), 1), round(float(value[1]), 1)]


def compact_view(observation: Any) -> dict[str, Any]:
    """One owner's view of one step, reduced to what kill attribution needs."""

    events = []
    for event in observation.events:
        if event.event_type not in TRACE_EVENTS:
            continue
        item: dict[str, Any] = {"t": event.tick, "type": event.event_type}
        for key, value in (("owner", event.owner), ("entity", event.entity_id), ("card", event.card_id)):
            if value is not None:
                item[key] = value
        if event.position is not None:
            item["pos"] = _round_pair(event.position)
        for key, name in (("amount", "amount"), ("effective_card_id", "effective_card")):
            value = event.data.get(key)
            if value is not None:
                item[name] = value
        events.append(item)
    entities = []
    for entity in observation.entities:
        item = {
            "id": entity.entity_id,
            "owner": entity.owner,
            "card": entity.card_id,
            "kind": entity.entity_kind,
            "pos": _round_pair(entity.position),
        }
        if entity.velocity is not None:
            item["vel"] = _round_pair(entity.velocity)
        if entity.hitpoints is not None:
            item["hp"] = entity.hitpoints
        if entity.shield:
            item["shield"] = entity.shield
        target = entity.attack_state.target_entity if entity.attack_state is not None else None
        if target is None:
            target = entity.visible_target
        if target is not None:
            item["target"] = target
        if entity.source_entity is not None:
            item["src"] = entity.source_entity
        if entity.causal_group is not None:
            item["group"] = entity.causal_group.handle
        projectile = entity.projectile_state
        if projectile is not None:
            item["proj"] = {
                "src": projectile.source_entity,
                "tgt": projectile.target_entity,
                "dest": _round_pair(projectile.target_position),
                "flight": projectile.phase.value,
            }
        entities.append(item)
    towers = []
    for tower in observation.towers:
        target = tower.attack_state.target_entity if tower.attack_state is not None else None
        if target is None:
            target = tower.visible_target
        item = {
            "id": tower.entity_id,
            "owner": tower.owner,
            "kind": tower.tower_kind,
            "pos": _round_pair(tower.position),
            "hp": tower.hitpoints,
            "max_hp": getattr(tower, "max_hitpoints", None),
            "active": tower.active,
        }
        if getattr(tower, "tower_troop_id", None) is not None:
            item["troop"] = tower.tower_troop_id
        if target is not None:
            item["target"] = target
        towers.append(item)
    view = {
        "tick": observation.tick,
        "owner": observation.owner,
        "phase": getattr(observation, "phase", None),
        "events": events,
        "entities": entities,
        "towers": towers,
    }
    # The acting owner's own hand and elixir, for behavior statistics (labels only, never model inputs).
    player = next((item for item in getattr(observation, "players", ()) if item.owner == observation.owner), None)
    if player is not None:
        view["hand"] = list(getattr(player, "hand", ()))
        elixir = getattr(player, "elixir_exact", None)
        if elixir is not None:
            view["elixir"] = round(float(elixir), 2)
    return view


class TraceWriter:
    """Write one gzip JSON-lines trace per game: a header, each owner's view of every step, and the
    exact native damage/death records gathered since the previous step. Games may be open at once."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles: dict[int, Any] = {}

    def start(self, game_index: int, header: Mapping[str, Any]) -> None:
        self.close(game_index)
        path = self.directory / f"trace-{game_index + 1:04d}.jsonl.gz"
        handle = gzip.open(path, "wt", encoding="utf-8")  # noqa: SIM115 - one file spans many calls
        handle.write(json.dumps({"version": TRACE_VERSION, **header}) + "\n")
        self._handles[game_index] = handle

    def write(self, game_index: int, observations: Mapping[int, Any], combat: list[Mapping[str, Any]] = ()) -> None:
        handle = self._handles.get(game_index)
        if handle is None:
            raise RuntimeError(f"trace writer has no open game {game_index}")
        if combat:
            handle.write(json.dumps({"combat": list(combat)}, separators=(",", ":")) + "\n")
        for owner in sorted(observations):
            handle.write(json.dumps(compact_view(observations[owner]), separators=(",", ":")) + "\n")

    def close(self, game_index: int | None = None) -> None:
        """Close one game's trace, or every open trace."""

        for index in tuple(self._handles) if game_index is None else (game_index,):
            handle = self._handles.pop(index, None)
            if handle is not None:
                handle.close()


def read_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the header, the per-owner views and the exact native combat records of one game."""

    views: list[dict[str, Any]] = []
    combat: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("version") != TRACE_VERSION:
            raise ValueError(f"{path} is not a {TRACE_VERSION} trace")
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if "combat" in item:
                combat.extend(item["combat"])
            else:
                views.append(item)
    return header, views, combat


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    (ax, ay), (bx, by) = a, b
    return math.hypot(ax - bx, ay - by)


def exact_kills(combat: list[dict[str, Any]], tower_ids: set[int]) -> dict[int, dict[str, Any]]:
    """Victim -> attacker for every non-tower death in the native records.

    The death record's own source wins; otherwise the lethal damage record on the same victim.
    ``source`` is the owning attacker (e.g. the Musketeer), ``immediate`` what landed the hit
    (e.g. its bullet); each is ``None`` when the engine left it absent.
    """

    lethal: dict[int, dict[str, Any]] = {}
    deaths: dict[int, dict[str, Any]] = {}
    for record in combat:
        target = record.get("target")
        if not target or target.get("id") is None or target["id"] in tower_ids:
            continue
        if record["kind"] == "damage" and record.get("lethal"):
            lethal[target["id"]] = record
        elif record["kind"] == "death":
            deaths.setdefault(target["id"], record)
    kills = {}
    for victim, death in deaths.items():
        damage = lethal.get(victim, {})
        source = death.get("source") or damage.get("source")
        immediate = death.get("immediate") or damage.get("immediate")
        kills[victim] = {
            "t": death["t"],
            "victim_owner": death["target"].get("owner"),
            "victim_card": death["target"].get("card"),
            "source": source,
            "immediate": immediate,
            "lethal_damage_seen": bool(damage),
        }
    return kills


class _OwnerTimeline:
    """One owner's views in tick order, with spawner links, for windowed lookups."""

    def __init__(self, views: list[dict[str, Any]], owner: int) -> None:
        self.owner = owner
        self.views = sorted((view for view in views if view["owner"] == owner), key=lambda view: view["tick"])
        self.ticks = [view["tick"] for view in self.views]
        self.parents = {
            item["id"]: item["src"] for view in self.views for item in view["entities"] if item.get("src") is not None
        }

    def root(self, entity_id: int) -> int:
        seen = set()
        while entity_id in self.parents and entity_id not in seen:
            seen.add(entity_id)
            entity_id = self.parents[entity_id]
        return entity_id

    def window(self, start: int, end: int) -> list[dict[str, Any]]:
        return self.views[bisect_left(self.ticks, start) : bisect_right(self.ticks, end)]


def heuristic_origins(timeline: _OwnerTimeline, victim: int, tick: int) -> dict[int, str]:
    """Public-state guess of which of the timeline owner's sources killed ``victim``: root id -> kind."""

    owner, root = timeline.owner, timeline.root
    window = timeline.window(tick - HEURISTIC_WINDOW_TICKS, tick)
    victim_position = None
    for view in window:
        for item in view["entities"]:
            if item["id"] == victim:
                victim_position = item["pos"]
    origins: dict[int, str] = {}
    for view in window:
        for item in view["entities"]:
            if item["owner"] != owner:
                continue
            projectile = item.get("proj")
            if item.get("target") == victim:
                origins[root(item["id"])] = "building" if item["kind"] == "building" else "unit"
            elif projectile is not None and (
                projectile.get("tgt") == victim
                or (
                    victim_position is not None
                    and projectile.get("dest") is not None
                    and _distance(projectile["dest"], victim_position) <= AREA_RADIUS
                )
            ):
                origins[root(projectile.get("src") or item["id"])] = "projectile"
            elif (
                item["kind"] == "spell"
                and victim_position is not None
                and _distance(item["pos"], victim_position) <= AREA_RADIUS
            ):
                origins[root(item["id"])] = "spell_area"
        for tower in view["towers"]:
            if tower["owner"] == owner and tower.get("target") == victim:
                origins[tower["id"]] = "tower"
    return origins


def audit_game(views: list[dict[str, Any]], combat: list[dict[str, Any]]) -> dict[str, Any]:
    tower_ids = {tower["id"] for view in views for tower in view["towers"]}
    entity_kind = {item["id"]: item["kind"] for view in views for item in view["entities"]}
    timelines = {owner: _OwnerTimeline(views, owner) for owner in (0, 1)}

    result: dict[str, Any] = {"games": 1, "combat_records": len(combat)}
    kills = exact_kills(combat, tower_ids)
    outcomes: Counter = Counter()
    attackers: Counter = Counter()
    heuristic: Counter = Counter()
    for victim, kill in kills.items():
        source = kill["source"] or kill["immediate"]
        if source is None:
            outcomes["no_source"] += 1
            continue
        if source.get("id") is None:
            outcomes["source_without_identity"] += 1
            continue
        if source.get("owner") == kill["victim_owner"]:
            outcomes["own_side_source"] += 1
            continue
        outcomes["attributed"] += 1
        attacker = source["id"]
        kind = "tower" if attacker in tower_ids else entity_kind.get(attacker, "unseen")
        attackers[(kind, source.get("card"))] += 1
        if kill["immediate"] and kill["source"] and kill["immediate"].get("id") != kill["source"].get("id"):
            outcomes["immediate_differs_from_source"] += 1
        owner = source.get("owner")
        if owner in (0, 1):
            timeline = timelines[owner]
            guesses = heuristic_origins(timeline, victim, kill["t"])
            if not guesses:
                heuristic["none"] += 1
            elif timeline.root(attacker) in guesses or attacker in guesses:
                heuristic["includes_exact" if len(guesses) > 1 else "exact_only"] += 1
            else:
                heuristic["misses_exact"] += 1
    result["kills"] = len(kills)
    result["outcomes"] = dict(outcomes)
    result["heuristic"] = dict(heuristic)
    result["attackers"] = {f"{kind}:{card}": count for (kind, card), count in attackers.items()}
    return result


def _merge(total: dict[str, Any], part: Mapping[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, Mapping):
            _merge(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value


def audit_directory(directory: Path) -> dict[str, Any]:
    total: dict[str, Any] = {"games": 0}
    for path in sorted(Path(directory).glob("trace-*.jsonl.gz")):
        _header, views, combat = read_trace(path)
        _merge(total, audit_game(views, combat))
    return total


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "0/0"


def format_report(audit: Mapping[str, Any]) -> str:
    from ...card_names import CARD_NAMES

    kills = audit.get("kills", 0)
    outcomes = audit.get("outcomes", {})
    heuristic = audit.get("heuristic", {})
    attributed = outcomes.get("attributed", 0)
    lines = [
        f"games: {audit['games']}, native damage/death records: {audit.get('combat_records', 0)}",
        f"unit deaths in native records: {kills}",
        f"  attacker known:                {_share(attributed, kills)}",
        f"  source present but unmapped:   {_share(outcomes.get('source_without_identity', 0), kills)}",
        f"  no source:                     {_share(outcomes.get('no_source', 0), kills)}",
        f"  source on the victim's side:   {_share(outcomes.get('own_side_source', 0), kills)}",
        f"  shooter differs from the hit:  {_share(outcomes.get('immediate_differs_from_source', 0), attributed)}",
        "public-state heuristic against the known attacker:",
        f"  only the right one:            {_share(heuristic.get('exact_only', 0), attributed)}",
        f"  right one among several:       {_share(heuristic.get('includes_exact', 0), attributed)}",
        f"  wrong:                         {_share(heuristic.get('misses_exact', 0), attributed)}",
        f"  nothing found:                 {_share(heuristic.get('none', 0), attributed)}",
        "most frequent killers:",
    ]
    for key, count in sorted(audit.get("attackers", {}).items(), key=lambda item: -item[1])[:15]:
        kind, card = key.split(":", 1)
        name = CARD_NAMES.get(int(card), (card, card))[0] if card not in ("None", "") else "-"
        lines.append(f"  {kind:10s} {name:12s} {count}")
    return "\n".join(lines)


def record(argv: list[str]) -> int:
    """Play a few round-robin blocks on a resident engine with traces on, then audit them."""

    import torch

    from ...round_robin import base_match, existing_endpoint, parse_blocks, read_options
    from .deck_pool import DeckPoolManifestV4
    from .evaluate import evaluate
    from .evaluation_protocol import natural_schedule

    parser = argparse.ArgumentParser(
        prog="defense_trace record",
        description="Play round-robin blocks (same config and conditions as round_robin.sh) with combat traces.",
    )
    parser.add_argument("--config", type=Path, help="round-robin config (default: round_robin.local.json)")
    parser.add_argument("--model-a", required=True, help="model name from the config, e.g. 'BC 200'")
    parser.add_argument("--model-b", required=True, help="model name from the config, e.g. 'General 460'")
    parser.add_argument("--blocks", type=parse_blocks, default=[0, 1, 2], help="block IDs (4 games each), default 0-2")
    parser.add_argument("--existing-engine", type=existing_endpoint, required=True, help="HOST:PORT")
    parser.add_argument("--output", type=Path, default=Path("evaluations/defense-audit"))
    args = parser.parse_args(argv)
    for key in ("cpu_threads", "lanes", "deck_pool", "delay_ticks", "min_play_gap_ticks", "battle_level", "device"):
        setattr(args, key, None)
    options = read_options(args)
    for name in (args.model_a, args.model_b):
        if name not in options["models"]:
            parser.error(f"{name!r} is not a model in the config: {sorted(options['models'])}")
    torch.set_num_threads(options["cpu_threads"])
    match = base_match(options)
    schedule = natural_schedule(match, args.blocks, deck_pool=DeckPoolManifestV4.load(options["deck_pool"]))
    host, port = args.existing_engine
    traces = args.output / "traces"
    print(f"{args.model_a} vs {args.model_b}: {len(schedule)} games, blocks {args.blocks}", flush=True)
    evaluate(
        options["models"][args.model_a],
        options["models"][args.model_b],
        games=len(schedule),
        match=match,
        output=args.output,
        host=host,
        port=port,
        device=options["device"],
        sample=options["sample"],
        delay_a_ticks=tuple(options["delay_ticks"]),
        delay_b_ticks=tuple(options["delay_ticks"]),
        min_play_gap_ticks=tuple(options["min_play_gap_ticks"]),
        gate_temperature_a=options["gate_temperature"],
        gate_temperature_b=options["gate_temperature"],
        card_tile_temperature_a=options.get("card_tile_temperature"),
        card_tile_temperature_b=options.get("card_tile_temperature"),
        schedule=schedule,
        resident_lanes=options["lanes"],
        trace_dir=traces,
    )
    audit = audit_directory(traces)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(format_report(audit))
    return 0


def analyze(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="defense_trace", description="Audit enemy-unit kill attribution.")
    parser.add_argument("traces", type=Path, help="directory of trace-*.jsonl.gz written with --trace-dir")
    parser.add_argument("--json", type=Path, help="also write the raw audit counts here")
    args = parser.parse_args(argv)
    audit = audit_directory(args.traces)
    if not audit["games"]:
        parser.error(f"no traces in {args.traces}")
    if args.json is not None:
        args.json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(format_report(audit))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "record":
        return record(arguments[1:])
    if arguments and arguments[0] == "analyze":
        arguments = arguments[1:]
    return analyze(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
