import json

import pyarrow as pa
import pyarrow.parquet as pq

from native_runner.training.v4 import select_expert_replays as sel

KEYS = ("hog-rider", "musketeer", "cannon", "fireball", "the-log", "skeletons", "ice-golem", "ice-spirit")


def payload(winner_leak=1.0, loser_leak=2.0, data_i=1, keys=KEYS):
    battle = {}
    for side, leak in (("team", winner_leak), ("opponent", loser_leak)):
        battle[side] = {
            "players": [
                {"deck": [{"card_key": key, "name": key} for key in keys], "elixir_leaked": leak, "tower_card": {}}
            ]
        }
    return {"battle": battle, "events": [{"source_fields": {"data_i": data_i}}]}


def row(tag, team_crowns=1, opponent_crowns=0, mode="Ranked", result="victory", **kwargs):
    return {
        "replay_tag": tag,
        "game_mode": mode,
        "result": result,
        "team_crowns": team_crowns,
        "opponent_crowns": opponent_crowns,
        "payload_json": json.dumps(payload(**kwargs)),
    }


def test_candidate_filters_and_maps_owners():
    support = sel.DeckSupport(sel.supported_card_ids())
    item = sel.candidate(row("A"), sel.DEFAULT_MODES, support)
    assert item["scoreline"] == "1-0"
    assert item["sides"]["winner"] == {"side": "team", "owner": 0, "elixir_leaked": 1.0}
    assert item["sides"]["loser"]["owner"] == 1
    assert sel.candidate(row("B", mode="1v1 Battle"), sel.DEFAULT_MODES, support) is None
    assert sel.candidate(row("C", result="draw"), sel.DEFAULT_MODES, support) is None
    bad = KEYS[:7] + ("not-a-real-card",)
    assert sel.candidate(row("D", keys=bad), sel.DEFAULT_MODES, support) is None
    defeat = sel.candidate(row("E", 0, 2, result="defeat"), sel.DEFAULT_MODES, support)
    assert defeat["scoreline"] == "2-0" and defeat["sides"]["winner"]["side"] == "opponent"


def fake(tag, scoreline, leak=1.0, loser_leak=1.0, deck=("d",)):
    return {
        "replay_tag": tag,
        "game_mode": "Ranked",
        "scoreline": scoreline,
        "deck": deck,
        "sides": {
            "winner": {"side": "team", "owner": 0, "elixir_leaked": leak},
            "loser": {"side": "opponent", "owner": 1, "elixir_leaked": loser_leak},
        },
    }


def test_select_applies_quotas_cutoff_and_deck_cap():
    candidates = [fake(f"a{i}", "1-0", deck=(str(i),)) for i in range(100)]
    candidates += [fake(f"b{i}", "3-0", deck=(str(i),)) for i in range(100)]
    candidates += [fake(f"c{i}", "1-0", leak=9.0, deck=(str(i),)) for i in range(50)]
    chosen = sel.select(candidates, target=20, deck_cap=1000, cutoff=5.0)
    counts = {s: sum(1 for c in chosen if c["scoreline"] == s) for s in ("1-0", "3-0")}
    assert len(chosen) == 20 and counts["1-0"] >= 10 and counts["3-0"] >= 2
    assert not any(c["replay_tag"].startswith("c") for c in chosen)
    capped = sel.select([fake(f"x{i}", "1-0") for i in range(10)], target=10, deck_cap=3, cutoff=5.0)
    assert len(capped) == 3


def test_close_loss_side_only_for_one_crown_games():
    one = sel.learn_sides(fake("a", "1-0", loser_leak=1.0), cutoff=2.0)
    assert [side["outcome"] for side in one["learn"]] == ["win", "close_loss"]
    leaky = sel.learn_sides(fake("b", "1-0", loser_leak=3.0), cutoff=2.0)
    two = sel.learn_sides(fake("c", "2-0"), cutoff=2.0)
    assert len(leaky["learn"]) == 1 and len(two["learn"]) == 1


def test_main_writes_selected_rows(tmp_path):
    rows = [row(f"r{i}", 1 if i % 2 else 3, 0) for i in range(12)]
    (tmp_path / "data" / "replays").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data" / "replays" / "part-000000.parquet")
    out = tmp_path / "out"
    assert sel.main([str(tmp_path / "data"), str(out), "--target", "4", "--leak-quantile", "1.0"]) == 0
    written = pq.read_table(out / "replays" / "part-000000.parquet")
    lines = [json.loads(line) for line in (out / "selection.jsonl").read_text().splitlines()]
    assert written.num_rows == 4 and [line["replay_tag"] for line in lines] == written["replay_tag"].to_pylist()
    assert json.loads((out / "summary.json").read_text())["selected_replays"] == 4
