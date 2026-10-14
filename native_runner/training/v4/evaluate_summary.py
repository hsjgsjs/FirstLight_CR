"""Combine evaluate.py output directories: model A's win rate with a 95% interval and first-play times.

    python -m native_runner.training.v4.evaluate_summary runs/eval/gen-*
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    centre = (p + z * z / (2 * games)) / (1 + z * z / games)
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return max(0.0, centre - half), min(1.0, centre + half)


def quantiles(values: Sequence[float]) -> str:
    ordered = sorted(values)
    if not ordered:
        return "-"
    return "/".join(f"{ordered[min(len(ordered) - 1, int(q * len(ordered)))]:.0f}" for q in (0.1, 0.5, 0.9))


def summarize(directories: Sequence[Path]) -> dict[str, object]:
    rows = [json.loads(line) for d in directories for line in (d / "matches.jsonl").read_text().splitlines() if line]
    games = len(rows)
    a_wins = sum(r["winner"] == "a" for r in rows)
    b_wins = sum(r["winner"] == "b" for r in rows)
    draws = games - a_wins - b_wins
    # A draw counts as half a win, so 50% means even.
    low, high = wilson_interval(a_wins + 0.5 * draws, games)
    first = {
        side: [r["first_play_seconds"][side] for r in rows if (r.get("first_play_seconds") or {}).get(side) is not None]
        for side in ("a", "b")
    }
    return {
        "games": games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score": (a_wins + 0.5 * draws) / games if games else None,
        "a_score_95": (low, high),
        "first_play_10_50_90": {side: quantiles(values) for side, values in first.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", type=Path, nargs="+")
    result = summarize(parser.parse_args(argv).directories)
    if not result["games"]:
        raise SystemExit("no finished games")
    low, high = result["a_score_95"]
    print(
        f"{result['games']} games: A {result['a_wins']} / B {result['b_wins']} / draws {result['draws']}; "
        f"A score {result['a_score']:.0%} (95% {low:.0%}-{high:.0%})"
    )
    print(
        f"first play 10/50/90% (s): A {result['first_play_10_50_90']['a']}, B {result['first_play_10_50_90']['b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
