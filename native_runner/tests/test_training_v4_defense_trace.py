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
    writer.write({0: observation, 1: observation}, [{"t": 1, "kind": "damage"}])
    writer.write({0: observation})
    writer.close()
    header, views, combat = read_trace(tmp_path / "trace-0001.jsonl.gz")
    assert header["seed"] == 7 and len(views) == 3 and combat == [{"t": 1, "kind": "damage"}]


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
    writer.write({0: _observation(skeleton_count=1)})
    writer.close()
    audit = audit_directory(tmp_path)
    assert audit["games"] == 1 and audit["kills"] == 0
    assert "unit deaths in native records: 0" in format_report(audit)
