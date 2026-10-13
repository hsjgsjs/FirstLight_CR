"""Run the checkpoint evaluator on several idle engines at once and pool the results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_ports(value):
    ports = []
    for part in value.split(","):
        start, _, end = part.strip().partition("-")
        ports.extend(range(int(start), int(end or start) + 1))
    if not ports or len(set(ports)) != len(ports):
        raise argparse.ArgumentTypeError("ports must be unique, e.g. 28000-28007")
    return ports


def split_games(games, shards):
    """Even game counts per shard where possible, so each shard plays both sides equally."""
    pairs, odd = divmod(games, 2)
    counts = [2 * (pairs // shards) + 2 * (index < pairs % shards) for index in range(shards)]
    if odd:
        counts[counts.index(min(counts))] += 1
    return [count for count in counts if count]


def pooled(output):
    rows = []
    for path in sorted(output.glob("shard-*/matches.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    finished = [row for row in rows if row["terminated"]]

    def actions(model):
        sides = [row["actions_by_owner"][row["a_owner"] if model == "a" else 1 - row["a_owner"]] for row in rows]
        return sum(sides) / len(sides) if sides else 0.0

    return {
        "completed": len(rows),
        "a_wins": sum(row["winner"] == "a" for row in finished),
        "b_wins": sum(row["winner"] == "b" for row in finished),
        "draws": sum(row["winner"] is None for row in finished),
        "truncated": sum(row["truncated"] for row in rows),
        "rejected_actions": sum(row["rejected_actions"] for row in rows),
        "a_actions_per_game": round(actions("a"), 1),
        "b_actions_per_game": round(actions("b"), 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=parse_ports, required=True, help="engine control ports, e.g. 28000-28007")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args, passthrough = parser.parse_known_args(argv)
    if args.games < 1:
        parser.error("--games must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    processes = []
    for index, (port, games) in enumerate(zip(args.ports, split_games(args.games, len(args.ports)))):
        shard = args.output / f"shard-{index:02d}"
        command = [
            sys.executable, "-m", "native_runner.training.v4.evaluate", *passthrough,
            "--port", str(port), "--games", str(games), "--seed", str(seed), "--output", str(shard),
        ]
        shard.mkdir(parents=True, exist_ok=True)
        log = (shard / "evaluate.log").open("w", encoding="utf-8")
        processes.append((port, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        seed += games
    failed = []
    for port, process, log in processes:
        if process.wait():
            failed.append(port)
        log.close()
    summary = {**pooled(args.output), "failed_ports": failed}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
