"""Behavior statistics from combat traces."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from native_runner.training.v4.behavior_stats import collect, format_report, lethal_spells, main
from native_runner.training.v4.defense_trace import TRACE_VERSION

FIREBALL, LOG, GOBLIN_BARREL, BALLOON, KNIGHT, HOG = 28000000, 28000011, 28000004, 26000006, 26000000, 26000021
PRINCESS = 159_000_000
PRINCESS_FULL_11 = 1400 * 2.56


def _towers(enemy_left_hp: float, *, own_left_hp: float = PRINCESS_FULL_11, max_hp: float = PRINCESS_FULL_11):
    return [
        {"id": 1, "owner": 0, "kind": "king", "pos": [9000, 3000], "hp": 6144, "max_hp": 6144, "active": True},
        {"id": 2, "owner": 0, "kind": "princess_left", "pos": [3500, 6500], "hp": own_left_hp, "max_hp": max_hp,
         "active": True, "troop": PRINCESS},
        {"id": 4, "owner": 1, "kind": "king", "pos": [9000, 29000], "hp": 6144, "max_hp": 6144, "active": True},
        {"id": 5, "owner": 1, "kind": "princess_left", "pos": [3500, 25500], "hp": enemy_left_hp, "max_hp": max_hp,
         "active": enemy_left_hp > 0, "troop": PRINCESS},
    ]


def _view(tick, owner, *, hand=(KNIGHT,), elixir=5.0, phase="regular", towers=None, entities=(), events=()):
    return {
        "tick": tick,
        "owner": owner,
        "phase": phase,
        "hand": list(hand),
        "elixir": elixir,
        "events": list(events),
        "entities": list(entities),
        "towers": towers if towers is not None else _towers(PRINCESS_FULL_11),
    }


def _play(tick, owner, card, position):
    return {"t": tick, "type": "action_executed", "owner": owner, "card": card, "pos": list(position)}


def _write(directory: Path, index: int, views, *, a_owner: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"trace-{index:04d}.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"version": TRACE_VERSION, "a_owner": a_owner, "checkpoint_ids": ["aaaa", "bbbb"]}) + "\n")
        for view in views:
            handle.write(json.dumps(view) + "\n")


def _pair_dir(tmp_path: Path) -> Path:
    pair = tmp_path / "rr-test" / "shard-00" / "pair-00"
    pair.mkdir(parents=True)
    (pair / "pair.json").write_text(json.dumps({"models": ["New", "General 460"]}))
    return pair / "traces"


def test_lethal_needs_the_spell_in_hand_affordable_and_enough_damage() -> None:
    low = _towers(150.0)  # 150 / 3584 is below one Fireball's 172 crown-tower damage
    assert lethal_spells(_view(0, 0, hand=(FIREBALL, KNIGHT), towers=low)) == {FIREBALL: {5}}
    assert lethal_spells(_view(0, 0, hand=(FIREBALL,), elixir=3.0, towers=low)) == {}
    assert lethal_spells(_view(0, 0, hand=(FIREBALL,), towers=_towers(400.0))) == {}
    # Own towers never count; the same low tower is a target for the other side.
    own_low = _towers(PRINCESS_FULL_11, own_left_hp=100.0)
    assert lethal_spells(_view(0, 0, hand=(FIREBALL,), towers=own_low)) == {}
    assert lethal_spells(_view(0, 1, hand=(FIREBALL,), towers=own_low)) == {FIREBALL: {2}}


def test_overtime_lethal_cast_and_regular_lapse_are_counted(tmp_path) -> None:
    traces = _pair_dir(tmp_path)
    low = _towers(150.0)
    game = [
        # Overtime: Fireball in hand kills the left princess; cast on it one second later.
        _view(3600, 0, hand=(FIREBALL, KNIGHT), phase="overtime", towers=low),
        _view(3620, 0, hand=(KNIGHT,), elixir=1.0, phase="overtime", towers=low,
              events=[_play(3618, 0, FIREBALL, (3600, 25400))]),
        # The other owner sees the same towers from its side: its own towers are full, nothing to count.
        _view(3600, 1, hand=(FIREBALL,), towers=low),
    ]
    _write(traces, 1, game)
    regular = [
        _view(100, 0, hand=(FIREBALL,), towers=low),
        _view(105, 0, hand=(KNIGHT,), towers=low),  # cycled out without casting
        _view(100, 1, hand=(KNIGHT,), towers=low),
    ]
    _write(traces, 2, regular)
    results = collect([tmp_path / "rr-test"])
    new = results["New"]
    assert new.games == 2
    assert new.lethal[(True, "cast_at_tower")] == 1
    assert new.lethal[(False, "lapsed")] == 1
    assert new.lethal_delays == [0.9]
    assert new.spells_at_towers[FIREBALL] == 1 and new.spells == 1
    assert results["General 460"].games == 2 and not results["General 460"].lethal


def test_log_on_air_only_and_goblin_barrel_damage(tmp_path) -> None:
    traces = _pair_dir(tmp_path)
    balloon = {"id": 50, "owner": 1, "card": BALLOON, "kind": "troop", "pos": [3500, 12000]}
    hog = {"id": 51, "owner": 1, "card": HOG, "kind": "troop", "pos": [14000, 12000]}
    game = [
        _view(200, 0, entities=[balloon, hog], events=[_play(200, 0, LOG, (3500, 8000))]),
        _view(205, 0, entities=[hog], events=[_play(205, 0, LOG, (14000, 8000))]),
        _view(300, 0, events=[_play(300, 1, GOBLIN_BARREL, (3500, 7000))]),
        _view(340, 0, towers=_towers(PRINCESS_FULL_11, own_left_hp=PRINCESS_FULL_11 - 250)),
        _view(360, 0, towers=_towers(PRINCESS_FULL_11, own_left_hp=PRINCESS_FULL_11 - 300),
              events=[_play(350, 0, LOG, (3500, 6000))]),
        _view(200, 1),
    ]
    _write(traces, 1, game)
    new = collect([traces])["New"]
    assert new.rolling["air_only"] == 1 and new.rolling["ground_only"] == 1
    assert new.barrels == 1 and new.barrels_answered == 1
    assert new.barrel_tower_losses == [300.0]
    # Both sides' towers at level 11: two princess and two king towers in the first view.
    assert new.level_ratios[("princess", 1.0)] == 2 and new.level_ratios[("king", 1.0)] == 2


def test_report_and_cli(tmp_path, capsys) -> None:
    traces = _pair_dir(tmp_path)
    _write(traces, 1, [_view(0, 0), _view(0, 1)])
    report = format_report(collect([traces]))
    assert "== New：1 局" in report and "可斩杀（加时）：0 次" in report
    assert main([str(tmp_path), "--json", str(tmp_path / "raw.json")]) == 0
    assert "General 460" in capsys.readouterr().out
    assert json.loads((tmp_path / "raw.json").read_text())["New"]["games"] == 1
