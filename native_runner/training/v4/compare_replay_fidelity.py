"""Compare how faithfully two cache builds rebuild the same human replays.

Build the same replay sample twice, changing one setting (for example ``--royaleapi-landing-ticks 1`` against
``21``), then:

    python -m native_runner.training.v4.compare_replay_fidelity --run plus1=CACHE_A --run plus21=CACHE_B

For each build it prints how many replays finished and how many match the recorded winner, crowns and all six
towers' end HP; then, on the replays both builds attempted, how many match only in one of them. A cache root is a
directory with ``index.json``, or a parent of partition directories that each hold one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cache import load_il_cache_index

CHECKS = ("winner_matches", "crowns_match", "towers_match")


def _index_paths(root: Path) -> tuple[Path, ...]:
    if (root / "index.json").is_file():
        return (root / "index.json",)
    paths = tuple(sorted(root.glob("*/index.json")))
    if not paths:
        raise FileNotFoundError(f"no index.json under {root}; is the cache build finished?")
    return paths


def load_replay_fidelity(root: Path) -> dict[str, dict[str, Any]]:
    """replay_tag -> its fidelity record, from shard summaries and from fully rejected shards."""

    records: dict[str, dict[str, Any]] = {}
    for index_path in _index_paths(root):
        index = load_il_cache_index(index_path)
        for shard in index["shards"]:
            if "replay_fidelity" not in shard:
                raise ValueError(f"{index_path} predates per-replay fidelity records; rebuild it with this code")
            for record in shard["replay_fidelity"]:
                records[str(record["replay_tag"])] = dict(record)
        for failure_path in sorted(index_path.parent.glob("*.failures.json")):
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            for record in payload.get("results", ()):
                records.setdefault(str(record["replay_tag"]), {**record, "completed": False})
    return records


def _exact(record: Mapping[str, Any]) -> bool:
    """Finished, towers checked and equal, and winner and crowns not contradicted."""

    return (
        bool(record.get("completed"))
        and record.get("towers_match") is True
        and record.get("winner_matches") is not False
        and record.get("crowns_match") is not False
    )


def summarize(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    result: dict[str, Any] = {
        "replays": total,
        "completed": sum(bool(item.get("completed")) for item in records.values()),
        "exact": sum(_exact(item) for item in records.values()),
    }
    for check in CHECKS:
        comparable = [item[check] for item in records.values() if item.get(check) is not None]
        result[check] = {"comparable": len(comparable), "matches": sum(bool(value) for value in comparable)}
    return result


def compare(runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    names = tuple(runs)
    common = sorted(set.intersection(*(set(records) for records in runs.values())))
    report: dict[str, Any] = {
        "runs": {name: summarize(records) for name, records in runs.items()},
        "common_replays": len(common),
    }
    if len(names) == 2:
        first, second = names
        report["paired"] = {
            f"exact_only_{first}": sum(_exact(runs[first][tag]) and not _exact(runs[second][tag]) for tag in common),
            f"exact_only_{second}": sum(_exact(runs[second][tag]) and not _exact(runs[first][tag]) for tag in common),
            "exact_both": sum(_exact(runs[first][tag]) and _exact(runs[second][tag]) for tag in common),
        }
    return report


def _share(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)" if denominator else "0/0"


def format_report(report: Mapping[str, Any]) -> Iterable[str]:
    for name, summary in report["runs"].items():
        yield f"{name}: {summary['replays']} replays"
        yield f"  finished          {_share(summary['completed'], summary['replays'])}"
        for check, label in zip(CHECKS, ("winner", "crowns", "towers HP")):
            item = summary[check]
            yield f"  {label:<17} {_share(item['matches'], item['comparable'])}"
        yield f"  exact (all three) {_share(summary['exact'], summary['replays'])}"
    yield f"replays in every run: {report['common_replays']}"
    for key, value in report.get("paired", {}).items():
        yield f"  {key}: {value}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=CACHE_ROOT")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args(None if argv is None else list(argv))
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or not name or name in runs:
            raise SystemExit(f"--run must be NAME=CACHE_ROOT with a unique name: {item!r}")
        runs[name] = load_replay_fidelity(Path(path))
    report = compare(runs)
    for line in format_report(report):
        print(line)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
