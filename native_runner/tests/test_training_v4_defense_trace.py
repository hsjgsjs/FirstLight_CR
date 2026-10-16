"""Combat traces: compact views, trace files, and the kill-attribution audit."""

import json

from native_runner.tests.test_training_v4_tensorizer import _observation
from native_runner.training.v4.defense_trace import (
    TraceWriter,
    audit_directory,
    audit_game,
    compact_view,
    format_report,
    read_trace,
)

GOBLIN = 26000002
KNIGHT = 26000000
LOG = 28000011


def _view(tick, owner, entities, towers=()):
    return {"tick": tick, "owner": owner, "events": [], "entities": list(entities), "towers": list(towers)}


def _unit(entity_id, owner, card, position, **extra):
    return {"id": entity_id, "owner": owner, "card": card, "kind": "troop", "pos": list(position), **extra}


def _fact(entity_id, owner, card):
    return {"id": entity_id, "owner": owner, "card": card, "object_kind": 1}


def _death(tick, victim, source):
    return {
        "t": tick,
        "kind": "death",
        "lethal": True,
        "target": victim,
        "source": source,
        "immediate": source,
        "projectile": None,
    }


def test_compact_view_is_json_and_keeps_entities() -> None:
    observation = _observation(skeleton_count=2)
    view = compact_view(observation)
    json.dumps(view)
    assert view["tick"] == observation.tick and view["owner"] == observation.owner
    assert {item["id"] for item in view["entities"]} == {entity.entity_id for entity in observation.entities}
    assert all(item["src"] == 999 for item in view["entities"] if item["id"] >= 1000)


def test_trace_round_trip(tmp_path) -> None:
    writer = TraceWriter(tmp_path)
    writer.start(0, {"seed": 7})
    observation = _observation(skeleton_count=1)
    writer.write(0, {0: observation, 1: observation}, [{"t": 1, "kind": "damage"}])
    writer.start(1, {"seed": 8})
    writer.write(1, {1: observation})
    writer.write(0, {0: observation})
    writer.close()
    header, views, combat = read_trace(tmp_path / "trace-0001.jsonl.gz")
    assert header["seed"] == 7 and len(views) == 3 and combat == [{"t": 1, "kind": "damage"}]
    header, views, combat = read_trace(tmp_path / "trace-0002.jsonl.gz")
    assert header["seed"] == 8 and len(views) == 1 and combat == []


def test_audit_attributes_kills_and_scores_the_public_heuristic() -> None:
    tower = {"id": 1, "owner": 0, "kind": "princess", "pos": [3500, 6500], "hp": 3000, "active": True}
    views = []
    for owner in (0, 1):
        views.append(
            _view(
                95,
                owner,
                [
                    _unit(10, 1, GOBLIN, (3500, 9000)),
                    _unit(11, 1, GOBLIN, (9000, 9000)),
                    _unit(12, 1, GOBLIN, (14000, 9000)),
                    _unit(20, 0, KNIGHT, (9000, 8500), target=11),
                    {
                        "id": 30,
                        "owner": 0,
                        "card": LOG,
                        "kind": "projectile",
                        "pos": [14000, 8000],
                        "proj": {"src": None, "tgt": None, "dest": [14000, 9000], "flight": "in_flight"},
                    },
                ],
                [{**tower, "target": 10}],
            )
        )
    combat = [
        _death(100, _fact(10, 1, GOBLIN), _fact(1, 0, None)),
        _death(100, _fact(11, 1, GOBLIN), _fact(20, 0, KNIGHT)),
        _death(100, _fact(12, 1, GOBLIN), _fact(30, 0, LOG)),
        _death(100, _fact(20, 0, KNIGHT), None),
        {"t": 90, "kind": "damage", "lethal": False, "target": _fact(11, 1, GOBLIN), "source": _fact(20, 0, KNIGHT)},
    ]
    audit = audit_game(views, combat)
    assert audit["kills"] == 4
    assert audit["outcomes"] == {"attributed": 3, "no_source": 1}
    assert audit["heuristic"] == {"exact_only": 3}
    assert audit["attackers"] == {"tower:None": 1, "troop:26000000": 1, "projectile:28000011": 1}


def test_audit_directory_report(tmp_path) -> None:
    writer = TraceWriter(tmp_path)
    writer.start(0, {})
    writer.write(0, {0: _observation(skeleton_count=1)})
    writer.close(0)
    audit = audit_directory(tmp_path)
    assert audit["games"] == 1 and audit["kills"] == 0
    assert "unit deaths in native records: 0" in format_report(audit)


def test_record_plays_round_robin_blocks_with_traces(monkeypatch, tmp_path) -> None:
    from native_runner import round_robin
    from native_runner.match_factory import MatchConfig
    from native_runner.training.v4 import defense_trace, evaluate
    from native_runner.training.v4.deck_pool import DeckPoolManifestV4

    options = {
        "models": {"BC 200": "bc.pt", "General 460": "general.pt"},
        "deck_pool": "pool.json",
        "cpu_threads": 1,
        "device": "cpu",
        "sample": True,
        "gate_temperature": 0.2,
        "delay_ticks": [18, 24],
        "min_play_gap_ticks": [6, 10],
        "lanes": 16,
    }
    calls = []
    monkeypatch.setattr(round_robin, "read_options", lambda args: options)
    monkeypatch.setattr(round_robin, "base_match", lambda options: MatchConfig(seed=3))
    monkeypatch.setattr(DeckPoolManifestV4, "load", staticmethod(lambda path: None))

    def fake_evaluate(a, b, **kwargs):
        calls.append((a, b, kwargs))
        writer = TraceWriter(kwargs["trace_dir"])
        writer.start(0, {})
        writer.write(0, {0: _observation(skeleton_count=1)})
        writer.close()

    monkeypatch.setattr(evaluate, "evaluate", fake_evaluate)
    output = tmp_path / "audit"
    output.mkdir()
    assert (
        defense_trace.main(
            [
                "record",
                "--model-a",
                "BC 200",
                "--model-b",
                "General 460",
                "--blocks",
                "0-1",
                "--existing-engine",
                "127.0.0.1:28000",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    a, b, kwargs = calls[0]
    assert (a, b) == ("bc.pt", "general.pt") and kwargs["games"] == 8 and kwargs["port"] == 28000
    assert kwargs["delay_a_ticks"] == (18, 24) and kwargs["gate_temperature_b"] == 0.2
    assert kwargs["resident_lanes"] == 16 and kwargs["trace_dir"] == output / "traces"
    assert json.loads((output / "audit.json").read_text())["games"] == 1
