from __future__ import annotations

import json
from pathlib import Path

from native_runner.training.v4.cache import write_il_cache_index
from native_runner.training.v4.compare_replay_fidelity import compare, load_replay_fidelity, main


def _record(tag: str, *, completed: bool = True, towers: bool | None = True, winner: bool | None = True) -> dict:
    return {
        "replay_tag": tag,
        "completed": completed,
        "failure_reason": None if completed else "scripted",
        "winner_matches": winner,
        "crowns_match": True,
        "towers_match": towers,
        "executed_expert_actions": 10,
        "expert_actions": 10,
    }


def _cache(root: Path, records: list[dict], failed: list[dict] = ()) -> Path:
    root.mkdir(parents=True)
    shard = {"replay_results": len(records), "sequences": 0, "owner_frames": 0, "match_decision_frames": 0,
             "compressed_bytes": 0, "container_bytes": 0, "replay_fidelity": records}
    write_il_cache_index(root / "index.json", [shard], contract={})
    if failed:
        (root / "part-00-shard-000001.failures.json").write_text(
            json.dumps({"results": list(failed)}), encoding="utf-8"
        )
    return root


def test_compare_counts_exact_rebuilds_per_run_and_paired(tmp_path: Path, capsys: object) -> None:
    first = _cache(tmp_path / "plus1", [_record("A", towers=False), _record("B"), _record("C", winner=False)])
    second = _cache(
        tmp_path / "plus21", [_record("A"), _record("B")], failed=[_record("C", completed=False, towers=None)]
    )
    runs = {"plus1": load_replay_fidelity(first), "plus21": load_replay_fidelity(second)}
    report = compare(runs)

    assert report["runs"]["plus1"]["exact"] == 1
    assert report["runs"]["plus21"]["exact"] == 2
    assert report["runs"]["plus1"]["towers_match"] == {"comparable": 3, "matches": 2}
    assert report["runs"]["plus21"]["completed"] == 2
    assert report["common_replays"] == 3
    assert report["paired"] == {"exact_only_plus1": 0, "exact_only_plus21": 1, "exact_both": 1}

    assert main(["--run", f"plus1={first}", "--run", f"plus21={second}"]) == 0
    assert "exact (all three) 2/3" in capsys.readouterr().out
