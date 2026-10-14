"""Read V4 PPO logs without importing Torch or connecting to training engines.

Copy this file to the training machine and run:
    python tools/analyze_ppo_elixir.py runs/spar16 --last 256 --csv /tmp/spar16-elixir.csv
Use --last 0 for every recorded match in each stage. CSV rows are learner seats,
not independent games; self-play contributes two seats and has no reported win rate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys


SEAT_FIELDS = {
    "result": "result",
    "deck_elixir": "deck_average_elixir",
    "actions": "active_actions",
    "waste": "cumulative_elixir_overflow_wasted",
    "opening_30s_waste": "opening_elixir_overflow_wasted",
    "after_first_waste": "game_elixir_overflow_wasted",
    "max_stretch_waste": "max_continuous_elixir_overflow_wasted",
    "personal_cost": "personal_elixir_overflow_penalty_total",
    "game_cost": "game_elixir_overflow_penalty_total",
    "unilateral_cost": "unilateral_elixir_overflow_penalty_total",
    "grace_crossings": "personal_elixir_overflow_grace_crossings",
    "full_wait_decisions_before_first": "full_elixir_deferred_decision_ticks_before_first_card",
}
TIME_FIELDS = {
    "first_submit_s": "first_nonwait_action_tick",
    "first_card_s": "first_successful_card_tick",
    "first_overflow_s": "first_elixir_overflow_tick",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1 and not line.endswith("\n"):
                print(f"WARNING: skipped unfinished final line in {path}", file=sys.stderr)
                break
            raise ValueError(f"{path}:{index + 1}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index + 1}: expected an object")
        rows.append(row)
    return rows


def read_object(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def seat_value(match: dict, field: str, owner: int):
    values = match.get(field)
    return values[owner] if isinstance(values, (list, tuple)) and len(values) > owner else None


def learner_rows(
    stage: str, matches: list[dict], names: dict[str, str], game_tracking_enabled: bool | None = None
) -> list[dict]:
    rows = []
    for match_index, match in enumerate(matches):
        owners = match["current_owners"]
        self_play = len(owners) == 2
        opponent_id = match.get("opponent_checkpoint_id")
        opponent = "self-play" if self_play else names.get(opponent_id, str(opponent_id or match["policy_matchup"]))
        for owner in owners:
            row = {
                "stage": stage,
                "match_index": match_index,
                "sequence": match.get("sequence"),
                "owner": owner,
                "self_play": self_play,
                "opponent": opponent,
                "opponent_checkpoint_id": opponent_id,
                "policy_matchup": match["policy_matchup"],
                "category": match.get("category"),
                "deck_id": match.get(f"deck{owner}_id"),
                "opponent_deck_id": match.get(f"deck{1 - owner}_id"),
                "truncated": bool(match.get("truncated", False)),
                "game_tracking_enabled": game_tracking_enabled,
                "duration_s": match["terminal_tick"] / 20.0 if match.get("terminal_tick") is not None else None,
            }
            row.update({label: seat_value(match, field, owner) for label, field in SEAT_FIELDS.items()})
            if game_tracking_enabled is False:
                row["after_first_waste"] = None
            for label, field in TIME_FIELDS.items():
                ticks = seat_value(match, field, owner)
                row[label] = None if ticks is None else ticks / 20.0
            rival_tick = seat_value(match, "first_successful_card_tick", 1 - owner)
            row["opponent_first_card_s"] = None if rival_tick is None else rival_tick / 20.0
            costs = [row[key] for key in ("personal_cost", "game_cost", "unilateral_cost")]
            row["overflow_cost"] = sum(costs) if all(value is not None for value in costs) else None
            duration = row["duration_s"]
            row["actions_per_minute"] = (
                row["actions"] * 60.0 / duration if duration and row["actions"] is not None else None
            )
            rows.append(row)
    return rows


def numbers(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(rows: list[dict], key: str) -> str:
    values = numbers(rows, key)
    if not values:
        return f"{key}: unavailable"
    positive = sum(value > 0 for value in values)
    return (
        f"{key}: n={len(values)}/{len(rows)} mean={statistics.mean(values):.6f} "
        f"p50/p90/p99={quantile(values, .5):.6f}/{quantile(values, .9):.6f}/{quantile(values, .99):.6f} "
        f"max={max(values):.6f} positive={positive}/{len(values)} ({positive / len(values):.1%})"
    )


def win_summary(rows: list[dict]) -> str:
    eligible = [row for row in rows if not row["self_play"] and not row["truncated"]]
    values = numbers(eligible, "result")
    if not values:
        return "win rate: unavailable (self-play/truncated/missing results excluded)"
    n = len(values)
    wins = sum(value > 0 for value in values)
    draws = sum(value == 0 for value in values)
    p, z = wins / n, 1.96
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (
        f"n={n} win={p:.1%} draw={draws / n:.1%} "
        f"Wilson95%=[{center - radius:.1%}, {center + radius:.1%}]"
    )


def timing_bucket(value) -> str:
    if value is None:
        return "no recorded first card"
    return "<10s" if value < 10 else "10-20s" if value < 20 else ">=20s"


def waste_bucket(value) -> str:
    if value is None:
        return "missing"
    return "zero" if value == 0 else "(0,1]" if value <= 1 else "(1,3]" if value <= 3 else ">3"


def print_group(rows: list[dict], details: bool) -> None:
    print(f"  seats={len(rows)} truncated_seats={sum(row['truncated'] for row in rows)}; {win_summary(rows)}")
    for key in (
        "first_card_s", "actions_per_minute", "waste", "after_first_waste",
        "max_stretch_waste", "personal_cost", "game_cost", "overflow_cost",
    ):
        print("  " + distribution(rows, key))
    eligible = [row for row in rows if not row["self_play"] and not row["truncated"]]
    paired = [row for row in eligible if row["first_card_s"] is not None and row["opponent_first_card_s"] is not None]
    if paired:
        proactive = sum(row["first_card_s"] < row["opponent_first_card_s"] for row in paired)
        print(f"  learner played before opponent: {proactive}/{len(paired)} (both first cards recorded)")
    for label, predicate in (("wins", lambda value: value > 0), ("losses", lambda value: value < 0)):
        selected = [row for row in eligible if row["result"] is not None and predicate(row["result"])]
        if selected:
            print(f"  {label}: {distribution(selected, 'waste')}")
    if not details:
        return
    strata = defaultdict(list)
    for row in eligible:
        cost = row["deck_elixir"]
        deck_band = "missing" if cost is None else "<3" if cost < 3 else "3-4" if cost < 4 else ">=4"
        strata[(row["owner"], deck_band)].append(row)
    for (owner, deck_band), seats in sorted(strata.items()):
        for key, bucket in (("first_card_s", timing_bucket), ("waste", waste_bucket)):
            buckets = defaultdict(list)
            for row in seats:
                buckets[bucket(row[key])].append(row)
            for label, selected in sorted(buckets.items()):
                print(f"  owner={owner} deck_cost={deck_band} {key}={label}: {win_summary(selected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory or one train-* stage")
    parser.add_argument("--last", type=int, default=256, help="last N matches per stage; 0 selects all")
    parser.add_argument("--details", action="store_true", help="add descriptive splits by side, deck cost and behavior")
    parser.add_argument("--csv", type=Path, help="export selected learner seats; refuses to overwrite an existing file")
    args = parser.parse_args()
    if args.last < 0:
        parser.error("--last must be nonnegative")
    stages = [args.run] if (args.run / "matches.jsonl").exists() else sorted(args.run.glob("train-*"))
    stages = [stage for stage in stages if stage.is_dir()]
    if not stages:
        parser.error(f"no training stages found in {args.run}")
    print("Observed telemetry, not tick-exact engine accounting. Missing values are never replaced with zero.")
    print("Stages/opponents are separate. Self-play seats are descriptive, not independent win-rate samples.")
    print("Intervals and behavior splits are descriptive: changing policies/decks prevent causal conclusions.")
    print("after_first_waste is only meaningful when game_elixir_overflow_penalty is enabled.")
    exports = []
    for stage in stages:
        config = read_object(stage / "config.json")
        arguments = config.get("arguments", {})
        print(f"\n=== {stage.name} ===")
        print(f"initial_update={config.get('initial_update_step', '?')} source={config.get('source_checkpoint', '?')}")
        for key in (
            "resume_checkpoint", "personal_elixir_overflow_penalty", "personal_elixir_overflow_grace",
            "game_elixir_overflow_penalty", "game_elixir_overflow_grace", "gate_temperature_start",
            "gate_temperature_end", "gate_temperature_anneal_updates", "execution_delay_ticks", "min_play_gap_ticks",
        ):
            print(f"  {key}={arguments.get(key, '?')}")
        metrics = read_jsonl(stage / "metrics.jsonl")
        if metrics:
            for label, metric in (("first", metrics[0]), ("last", metrics[-1])):
                print(
                    f"  {label} metric: update={metric.get('update')} "
                    f"gate_temperature={metric.get('rollout_gate_temperature', '?')} "
                    f"batch={metric.get('batch_index', '?')} batch_complete={metric.get('batch_complete', '?')}"
                )
        league = read_object(stage / "league.json")
        names = {
            entry["checkpoint_id"]: f"{Path(entry['checkpoint_path']).stem} [{entry['checkpoint_id'][:12]}]"
            for entry in league.get("entries", [])
        }
        matches = read_jsonl(stage / "matches.jsonl")
        selected = matches[-args.last:] if args.last else matches
        print(f"  recorded matches={len(matches)} selected={len(selected)} (file completion order)")
        coefficient = arguments.get("game_elixir_overflow_penalty")
        tracking = None if coefficient is None else coefficient > 0
        rows = learner_rows(stage.name, selected, names, tracking)
        groups = defaultdict(list)
        for row in rows:
            # Group by full identity, never by a possibly ambiguous checkpoint filename.
            groups[(row["policy_matchup"], row["opponent_checkpoint_id"] or "")].append(row)
        for _, group in sorted(groups.items()):
            print(f"\n  opponent: {group[0]['opponent']}")
            print_group(group, args.details)
        exports.extend(rows)
    if args.csv and exports:
        with args.csv.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exports[0]))
            writer.writeheader()
            writer.writerows(exports)
        print(f"\nWrote {len(exports)} learner-seat rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
