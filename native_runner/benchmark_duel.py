"""Small CPU throughput benchmark on existing, idle native engines.

Run from the repository root: python -m native_runner.benchmark_duel
This is a self-play speed check, not a round-robin strength evaluation.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/General/checkpoint-step-00000460.pt"


def parse_ports(value):
    try:
        ports = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("ports must be comma-separated integers") from None
    if not ports or len(ports) != len(set(ports)) or any(not 1 <= port <= 65535 for port in ports):
        raise argparse.ArgumentTypeError("ports must be unique and in 1..65535")
    return ports


def worker(args):
    from dataclasses import replace

    import torch

    from .training.v4 import evaluate as evaluator
    from .match_factory import MatchConfig
    from .training.v4.evaluation_protocol import balanced_cases

    torch.set_num_threads(args.threads)
    evaluator.RESIDENT_EVAL_LANES = args.lanes
    match = MatchConfig(seed=1001, level_cap=16, minimum_card_level=16, king_tower_level=16)
    schedule = [
        case
        for block in range(args.block_start, args.block_start + args.blocks)
        for case in balanced_cases(replace(match, seed=1001 + block), block=block, deadline_seconds=0)
        if case.condition == "natural"
    ]
    evaluator.evaluate(
        args.checkpoint, args.checkpoint,
        games=len(schedule), match=match, output=args.output,
        host=args.host, port=args.ports[0], device="cpu", sample=False,
        delay_a_ticks=21, delay_b_ticks=21, min_play_gap_ticks=(4, 8), schedule=schedule,
    )
    return 0


def finished_games(path):
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            count += bool(row["terminated"] and not row["truncated"])
    return count


def stop_workers(workers):
    for process, _destination in workers:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
    for process, _destination in workers:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()


def benchmark(args):
    parent = ROOT / "evaluations"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="mac-dual-speed-", dir=parent))
    expected = len(args.ports) * args.blocks * 4
    print(f"结果目录：{output}", flush=True)
    print(
        f"{len(args.ports)} 个已有引擎，每个 {args.lanes} 槽、{args.threads} 推理线程，共 {expected} 局。",
        flush=True,
    )
    print("General 确定性自对弈，16 级标准卡组，延迟 21 tick，间隔 4–8 tick。", flush=True)
    (output / "settings.json").write_text(
        json.dumps({**vars(args), "checkpoint": str(args.checkpoint), "output": str(output)}, indent=2),
        encoding="utf-8",
    )
    workers = []
    interrupted = False
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            try:
                for index, port in enumerate(args.ports):
                    destination = output / f"engine-{index}"
                    destination.mkdir()
                    log = stack.enter_context((destination / "evaluate.log").open("w", encoding="utf-8"))
                    command = [
                        sys.executable, "-u", "-m", "native_runner.benchmark_duel", "--worker",
                        "--host", args.host, "--ports", str(port),
                        "--threads", str(args.threads), "--lanes", str(args.lanes),
                        "--blocks", str(args.blocks), "--block-start", str(index * args.blocks),
                        "--checkpoint", str(args.checkpoint), "--output", str(destination),
                    ]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    workers.append((process, destination))
                last_notice = started
                while any(process.poll() is None for process, _destination in workers):
                    if any(process.poll() not in (None, 0) for process, _destination in workers):
                        break
                    if time.perf_counter() - last_notice >= 15:
                        completed = sum(finished_games(path / "matches.jsonl") for _process, path in workers)
                        print(f"已正常结束 {completed}/{expected} 局，已用 {time.perf_counter() - started:.0f} 秒。",
                              flush=True)
                        last_notice = time.perf_counter()
                    time.sleep(0.5)
            finally:
                stop_workers(workers)
    except KeyboardInterrupt:
        interrupted = True
    elapsed = time.perf_counter() - started
    finished = sum(finished_games(path / "matches.jsonl") for _process, path in workers)
    codes = [process.returncode for process, _destination in workers]
    success = not interrupted and len(workers) == len(args.ports) and all(code == 0 for code in codes)
    success = success and finished == expected
    result = dict(finished=finished, requested=expected, elapsed_seconds=elapsed,
                  games_per_minute=finished * 60 / elapsed, worker_exit_codes=codes, success=success)
    (output / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"正常结束 {finished}/{expected} 局；总耗时 {elapsed:.1f} 秒；吞吐 {finished * 60 / elapsed:.2f} 局/分钟。")
    if not success:
        print("测速未全部完成。")
        for process, destination in workers:
            if process.returncode:
                print(f"{destination / 'evaluate.log'}：")
                print((destination / "evaluate.log").read_text(encoding="utf-8")[-4000:])
    return 130 if interrupted else (0 if success else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=parse_ports, default=[28000, 28001])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--threads", type=int, default=5, help="PyTorch threads per evaluator process")
    parser.add_argument("--lanes", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--blocks", type=int, default=4, help="four natural games per block, per engine")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--block-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.blocks < 1 or args.block_start < 0:
        parser.error("threads and blocks must be positive; block-start must be nonnegative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.worker and (len(args.ports) != 1 or args.output is None):
        parser.error("worker requires one port and an output directory")
    return worker(args) if args.worker else benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
