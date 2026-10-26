"""Training curve: evaluate a run's newest strict checkpoints against one fixed opponent, same games every time.

Every point is a two-model round robin (natural openings, the same global blocks, decks and seeds), so points
differ only in the checkpoint. The run directory can be local (runs/NAME) or remote (HOST:PATH, read with ssh
and copied with scp), so an idle machine can follow a training server.
"""

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .round_robin import ROOT, collect_runs, digest, parse_blocks, read_options, study_identity, write_json
from .training.v4.evaluation_protocol import round_robin_summary

CHECKPOINT = re.compile(r"(?:^|/)(train-[^/]+)/checkpoints/update-(\d+)\.pt$")


def parse_checkpoints(paths):
    """(stage, update, path) for strict checkpoint paths, oldest first; stage names sort by start time."""
    found = []
    for path in paths:
        match = CHECKPOINT.search(path.strip())
        if match:
            found.append((match[1], int(match[2]), path.strip()))
    return sorted(found)


def list_checkpoints(source):
    if Path(source).expanduser().is_dir():
        root = Path(source).expanduser()
        return parse_checkpoints(str(path) for path in root.glob("train-*/checkpoints/update-*.pt"))
    host, _, directory = source.partition(":")
    if not host or not directory:
        raise ValueError(f"运行目录不存在，也不是 HOST:PATH：{source}")
    # The remote shell expands ~ and the glob; a missing match prints nothing.
    command = f"ls -1d {directory}/train-*/checkpoints/update-*.pt 2>/dev/null || true"
    output = subprocess.run(["ssh", host, command], check=True, stdout=subprocess.PIPE, text=True, timeout=120)
    return parse_checkpoints(output.stdout.splitlines())


def choose(checkpoints, done_labels, last, every_updates):
    """The newest checkpoint not yet evaluated, unless it is fewer than every_updates past the last point."""
    if not checkpoints:
        return None
    stage, update, path = checkpoints[-1]
    if f"{stage}/update-{update}" in done_labels:
        return None
    if last is not None and every_updates and last[0] == stage and update - last[1] < every_updates:
        return None
    return stage, update, path


def fetch(source, path, destination):
    """A private copy: the trainer prunes old checkpoints while an evaluation still needs the file."""
    if destination.is_file():
        return destination
    for stale in destination.parent.parent.glob("*/*.pt"):
        stale.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    if Path(source).expanduser().is_dir():
        shutil.copyfile(path, partial)
    else:
        host = source.partition(":")[0]
        # Newer scp sends the path over SFTP unquoted; checkpoint paths contain no spaces or shell characters.
        subprocess.run(["scp", "-q", f"{host}:{path}", str(partial)], check=True, timeout=3600)
    partial.replace(destination)
    return destination


def point_config(args, label, checkpoint):
    config = {
        "models": {label: str(checkpoint), args.opponent_name: str(args.opponent)},
        "anchor": args.opponent_name,
        "sample": args.gate_temperature is not None,
        "gate_temperature": args.gate_temperature,
        # Kept at the model's own action temperature so points stay comparable with curves already run.
        "card_tile_temperature": None,
    }
    # In the file rather than on the command line, so reading the result back sees the same study.
    for key in ("cpu_threads", "lanes"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    return config


def point_result(config_path, label, blocks):
    """Merged result of every rr-* run of this exact study, or None if no block is complete."""
    options = read_options(SimpleNamespace(config=config_path))
    identity = study_identity(options)
    runs = [
        path.parent
        for path in sorted((ROOT / "evaluations").glob("rr-*/study.json"))
        if json.loads(path.read_text(encoding="utf-8")) == identity
    ]
    if not runs:
        return None
    _, pairs, _ = collect_runs(runs)
    pairs = {pair: {block: rows for block, rows in groups.items() if block in blocks} for pair, groups in pairs.items()}
    summary = round_robin_summary(
        pairs,
        identity["model_order"],
        anchor=identity["anchor"],
        bootstrap_samples=1000,
        seed=identity["gameplay"]["seed"],
    )
    pair = next(item for item in summary["pairs"] if item["a"] == label)
    return pair if pair["blocks"] else None


def render(rows, args):
    lines = [
        f"# 训练曲线：{args.name}",
        "",
        f"对手 {args.opponent_name}；每点 {len(args.blocks)} 组 × 4 局正常开局（组号 {args.blocks[0]}–{args.blocks[-1]}，"
        f"每点相同的卡组与种子）；"
        + (f"采样，双方出牌温度 {args.gate_temperature}。" if args.gate_temperature is not None else "确定性出牌。"),
        "",
        "| 检查点 | 局数 | 胜 / 负 / 平 | 得分率 | 95%区间 | 完成时间 |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        interval = row["score_95"]
        lines.append(
            f"| {row['label']} | {row['games']} | {row['wins']} / {row['losses']} / {row['draws']} | "
            f"{row['score']:.1%} | "
            + (f"[{interval[0]:.1%}, {interval[1]:.1%}]" if interval else "—")
            + f" | {row['evaluated_at']} |"
        )
    return "\n".join(lines) + "\n"


def evaluate(args, label, checkpoint, curve, rows):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    config = curve / "configs" / f"{safe}.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    write_json(config, point_config(args, label, checkpoint))
    blocks = (
        f"{args.blocks[0]}-{args.blocks[-1]}"
        if args.blocks == list(range(args.blocks[0], args.blocks[-1] + 1))
        else ",".join(map(str, args.blocks))
    )
    command = [sys.executable, "-m", "native_runner.round_robin", "--config", str(config), "--blocks", blocks]
    for flag in ("existing_engine", "engines"):
        if getattr(args, flag) is not None:
            command += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    print(f"评测 {label}", flush=True)
    code = subprocess.run(command, cwd=ROOT, check=False).returncode
    result = point_result(config, label, set(args.blocks))
    if code or result is None or result["blocks"] < len(args.blocks):
        print(f"{label} 未完成（退出码 {code}），下次继续", flush=True)
        return False
    rows.append(
        {
            "label": label,
            "sha256": digest(Path(checkpoint)),
            "games": result["games"],
            "wins": result["a_wins"],
            "losses": result["b_wins"],
            "draws": result["draws"],
            "score": result["a_score"],
            "score_95": result["a_score_95"],
            "evaluated_at": datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
        }
    )
    with (curve / "curve.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    (curve / "curve.md").write_text(render(rows, args), encoding="utf-8")
    print(render(rows, args), flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="runs/NAME, or HOST:PATH of a run directory on another machine")
    parser.add_argument("--name", help="curve name (default: the run directory's name)")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        type=Path,
        help="checkpoint to evaluate first, e.g. the run's starting point (repeatable)",
    )
    parser.add_argument("--opponent", type=Path, default=Path("checkpoints/General/checkpoint-step-00000460.pt"))
    parser.add_argument("--opponent-name", default="General 460")
    parser.add_argument("--blocks", type=parse_blocks, default=parse_blocks("0-24"))
    parser.add_argument(
        "--gate-temperature", type=float, default=0.2, help="common sampling temperature; 0 means deterministic play"
    )
    parser.add_argument(
        "--every-updates",
        type=int,
        default=0,
        help="skip a checkpoint fewer than this many updates after the last point of the same stage",
    )
    parser.add_argument("--poll-minutes", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="evaluate what is pending now, then exit")
    parser.add_argument("--existing-engine")
    parser.add_argument("--engines", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--lanes", type=int)
    args = parser.parse_args(argv)
    args.gate_temperature = args.gate_temperature or None
    args.name = args.name or Path(args.source.rpartition(":")[2].rstrip("/")).name
    args.opponent = args.opponent if args.opponent.is_absolute() else ROOT / args.opponent
    if not args.opponent.is_file():
        print(f"对手检查点不存在：{args.opponent}", file=sys.stderr)
        return 1
    curve = ROOT / "evaluations" / f"curve-{args.name}"
    curve.mkdir(parents=True, exist_ok=True)
    rows = (
        [json.loads(line) for line in (curve / "curve.jsonl").read_text(encoding="utf-8").splitlines()]
        if (curve / "curve.jsonl").is_file()
        else []
    )
    print(f"曲线目录：{curve}", flush=True)
    for baseline in args.baseline:
        label = f"baseline/{baseline.stem}"
        if label not in {row["label"] for row in rows} and not evaluate(args, label, baseline.resolve(), curve, rows):
            return 1
    last = None
    while True:
        done = {row["label"] for row in rows}
        try:
            picked = choose(list_checkpoints(args.source), done, last, args.every_updates)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print(f"读取检查点失败：{error}", flush=True)
            picked = None
        if picked is not None:
            stage, update, path = picked
            label = f"{stage}/update-{update}"
            try:
                local = fetch(args.source, path, curve / "checkpoints" / stage / f"update-{update:08d}.pt")
            except (OSError, subprocess.SubprocessError) as error:
                print(f"复制 {label} 失败：{error}", flush=True)
                local = None
            if local is not None and evaluate(args, label, local, curve, rows):
                last = (stage, update)
                local.unlink()
                continue
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(60 * args.poll_minutes)


if __name__ == "__main__":
    raise SystemExit(main())
