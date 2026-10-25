"""Fixed-condition, resumable round robin; raw results can be copied between hosts."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .local_config import load_env
from .match_factory import MatchConfig
from .training.v4.evaluation_protocol import complete_natural_blocks, round_robin_summary, swap_decks

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_blocks(value):
    result = []
    try:
        for part in value.split(","):
            low, separator, high = part.strip().partition("-")
            low, high = int(low), int(high) if separator else int(low)
            if low < 0 or high < low or high - low > 100000:
                raise ValueError()
            result.extend(range(low, high + 1))
        if not result or len(set(result)) != len(result):
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("blocks must be nonoverlapping inclusive IDs, e.g. 0-79 or 80-99") from None
    return sorted(result)


def tick_bounds(value):
    parts = str(value).split("-")
    if len(parts) not in (1, 2):
        raise ValueError("tick range must be N or MIN-MAX")
    low, high = int(parts[0]), int(parts[-1])
    if not 1 <= low <= high <= 65535:
        raise ValueError("tick bounds must satisfy 1 <= MIN <= MAX <= 65535")
    return [low, high]


def existing_endpoint(value):
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("--existing-engine must be HOST:PORT")
    return host.strip("[]"), int(port)


def read_options(args):
    load_env()
    config = args.config or ROOT / "round_robin.local.json"
    defaults = json.loads((ROOT / "round_robin.example.json").read_text(encoding="utf-8"))
    saved = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
    if args.config and not config.is_file():
        raise ValueError(f"configuration does not exist: {config}")
    if not isinstance(saved, dict) or saved.keys() - defaults.keys():
        raise ValueError("unknown round-robin config field")
    options = {**defaults, **saved}
    for key in ("cpu_threads", "lanes", "deck_pool", "delay_ticks", "min_play_gap_ticks", "battle_level", "device"):
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)

    def path(value):
        candidate = Path(value).expanduser()
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        if not candidate.is_file():
            raise ValueError(f"文件不存在：{candidate}")
        return str(candidate.resolve())

    if not isinstance(options["models"], dict) or len(options["models"]) < 2:
        raise ValueError("models must map at least two distinct names to exact checkpoint files")
    options["models"] = {name: path(value) for name, value in options["models"].items()}
    if options["anchor"] not in options["models"]:
        raise ValueError("rating anchor must name one of the models")
    pool = options["deck_pool"] or os.environ.get("CR_PPO_DECK_POOL") or "runs/deck-pool.json"
    options["deck_pool"] = path(pool)
    options["match_config"] = path(options["match_config"]) if options["match_config"] else None
    for key in ("cpu_threads", "bootstrap_samples", "lanes"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if options["lanes"] not in (4, 8, 16) or options["battle_level"] not in (11, 16):
        raise ValueError("lanes must be 4, 8 or 16; battle_level must be 11 or 16")
    if not isinstance(options["sample"], bool):
        raise ValueError("sample must be a boolean")
    if options["sample"] and options["gate_temperature"] is None:
        raise ValueError("sampling requires one explicit common gate_temperature for all models")
    temperature = options["gate_temperature"]
    if temperature is not None and (not math.isfinite(float(temperature)) or float(temperature) <= 0):
        raise ValueError("gate_temperature must be finite and positive")
    card_tile = options["card_tile_temperature"]
    if card_tile is not None and (
        isinstance(card_tile, bool)
        or not isinstance(card_tile, (int, float))
        or not math.isfinite(float(card_tile))
        or float(card_tile) <= 0
    ):
        raise ValueError("card_tile_temperature must be finite and positive, or null")
    if card_tile is not None and not options["sample"]:
        raise ValueError("card_tile_temperature only applies with sample: true")
    for key in ("delay_ticks", "min_play_gap_ticks"):
        options[key] = tick_bounds(options[key])
    return options


def base_match(options):
    values = json.loads(Path(options["match_config"]).read_text(encoding="utf-8")) if options["match_config"] else {}
    for key in ("owner0_account_id", "owner1_account_id"):
        values.pop(key, None)
    values.update(seed=options["seed"], level_cap=options["battle_level"],
                  minimum_card_level=options["battle_level"], king_tower_level=options["battle_level"])
    return MatchConfig(**values)


def study_identity(options):
    gameplay = {key: options[key] for key in ("battle_level", "delay_ticks", "min_play_gap_ticks", "sample",
                                            "gate_temperature", "card_tile_temperature", "seed", "device",
                                            "lanes")}
    # CPU threads and host paths describe resources, not match conditions.
    source = hashlib.sha256()
    for path in sorted((ROOT / "native_runner").rglob("*.py")):
        if "tests" not in path.parts:
            source.update(str(path.relative_to(ROOT)).encode())
            source.update(path.read_bytes())
    identity = dict(schema="round-robin.v1", models={name: digest(Path(path)) for name, path in options["models"].items()},
                    model_order=list(options["models"]), anchor=options["anchor"], gameplay=gameplay,
                    match=asdict(base_match(options)), deck_pool_sha256=digest(Path(options["deck_pool"])),
                    schedule="natural-4-v1", source_sha256=source.hexdigest())
    # In its JSON form (tuples become lists), so it compares equal to a saved study.json and a rerun skips
    # the blocks that earlier runs completed.
    return json.loads(json.dumps(identity))


def read_rows(path):
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                rows.append(json.loads(line))
    return rows


def validate_group(rows):
    first = rows[0]
    original = first["match"]
    # Variant 0 is always the unswapped base match, A in owner 0 with deck 0.
    values = {key: value for key, value in original.items() if key not in ("owner0_account_id", "owner1_account_id")}
    match = MatchConfig(**values)
    for variant, (owner, deck) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        row = rows[variant]
        expected_match = asdict(swap_decks(match) if owner != deck else match)
        if (row["a_owner"] != owner or row["a_deck"] != deck or row["seed"] != first["seed"]
                or canonical(row["match"]) != canonical(expected_match)
                or row.get("opening_deadline_seconds") != 0
                or any(row.get("opening_overrides", {}).values())):
            raise ValueError("inconsistent natural block seed, side, deck assignment or opening")


def collect_runs(runs):
    identity = None
    pairs = {}
    fingerprints = {}
    duplicates = excluded = 0
    engine_identity = None
    for run in sorted(set(Path(path).resolve() for path in runs)):
        saved = json.loads((run / "study.json").read_text(encoding="utf-8"))
        if identity is not None and saved != identity:
            raise ValueError(f"拒绝合并：模型或对局设置身份不一致：{run}")
        identity = saved
        names = saved["model_order"]
        valid_pairs = set(itertools.combinations(names, 2))
        for path in sorted(run.glob("shard-*/engine.json")):
            engine = json.loads(path.read_text(encoding="utf-8"))
            if engine_identity is not None and engine != engine_identity:
                raise ValueError(f"拒绝合并：引擎或游戏资源身份不一致：{path}")
            engine_identity = engine
        for marker in sorted(run.rglob("pair.json")):
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            pair = tuple(metadata["models"])
            if pair not in valid_pairs:
                raise ValueError(f"unknown model pair: {marker}")
            rows = read_rows(marker.parent / "matches.jsonl")
            if not rows:
                continue
            if not (marker.parent.parent / "engine.json").is_file():
                raise ValueError(f"missing engine identity: {marker}")
            actual = json.loads((marker.parent / "identity.json").read_text(encoding="utf-8"))
            if [p["sha256"] for p in actual["checkpoints"]] != [saved["models"][name] for name in pair]:
                raise ValueError(f"loaded model hash mismatch: {marker}")
            gameplay = saved["gameplay"]
            checks = dict(sample=gameplay["sample"], delay_a_ticks=gameplay["delay_ticks"],
                          delay_b_ticks=gameplay["delay_ticks"], min_play_gap_ticks=gameplay["min_play_gap_ticks"],
                          resident_lanes=gameplay["lanes"], transport="resident-compact")
            if any(actual.get(key) != value for key, value in checks.items()):
                raise ValueError(f"actual evaluator settings mismatch: {marker}")
            if gameplay["sample"] and actual["gate_temperatures"] != [gameplay["gate_temperature"]] * 2:
                raise ValueError(f"actual sampling temperatures mismatch: {marker}")
            if actual.get("card_tile_temperatures", [None, None]) != [gameplay.get("card_tile_temperature")] * 2:
                raise ValueError(f"actual card/tile temperatures mismatch: {marker}")
            groups = complete_natural_blocks(rows)
            planned = {(case["block"], case["variant"]): case for case in actual["schedule"]}
            for row in rows:
                case = planned.get((row["block"], row["variant"]))
                if case is None or any(canonical(row[key]) != canonical(case[key])
                                       for key in ("match", "a_owner", "a_deck", "condition")):
                    raise ValueError(f"game differs from recorded schedule: {marker}")
            excluded += len({row["block"] for row in rows}) - len(groups)
            destination = pairs.setdefault(pair, {})
            for block, group in groups.items():
                validate_group(group)
                if group[0]["seed"] != gameplay["seed"] + block:
                    raise ValueError("block seed differs from study identity")
                fingerprint = canonical(group[0]["match"])
                if block in fingerprints and fingerprints[block] != fingerprint:
                    raise ValueError("same global block has different decks across model pairs")
                fingerprints[block] = fingerprint
                if block in destination:
                    clean = lambda items: [{k: v for k, v in row.items() if k != "game"} for row in items]
                    if clean(group) != clean(destination[block]):
                        raise ValueError(f"conflicting duplicate results: {pair}, block {block}")
                    duplicates += 1
                else:
                    destination[block] = group
    if identity is None:
        raise ValueError("no round-robin runs found")
    return identity, pairs, dict(duplicate_blocks=duplicates, excluded_incomplete_blocks=excluded,
                                 engine_identity=engine_identity)


def render_report(summary):
    def pct(value):
        return "—" if value is None else f"{value:.1%}"
    def ci(value, percentage=False):
        if value is None:
            return "—"
        return f"[{pct(value[0])}, {pct(value[1])}]" if percentage else f"[{value[0]:.1f}, {value[1]:.1f}]"
    lines = ["# 正常开局循环赛", "", "| A | B | 完整组 | 局数 | A胜 / B胜 / 平 | A得分率 | 95%区间 |",
             "| --- | --- | ---: | ---: | --- | ---: | --- |"]
    for pair in summary["pairs"]:
        lines.append(f"| {pair['a']} | {pair['b']} | {pair['blocks']} | {pair['games']} | "
                     f"{pair['a_wins']} / {pair['b_wins']} / {pair['draws']} | {pct(pair['a_score'])} | "
                     f"{ci(pair['a_score_95'], True)} |")
    lines += ["", "| 模型 | 等级分（Elo 尺度） | 95%区间 |", "| --- | ---: | --- |"]
    for rating in summary["ratings"] or []:
        lines.append(f"| {rating['model']} | {rating['elo']:.1f} | {ci(rating['elo_95'])} |")
    lines += ["", f"基准：{summary['anchor']} = 0。等级分使用全部模型对均完成的 {summary['shared_rating_blocks']} 组。",
              "得分率 =（胜 + 0.5 × 平）/ 局数；只计完整四局组，截断局不算平局。",
              "区间按全局组号 bootstrap；等级分每次共同重采样所有模型对后重新拟合。",
              "Bradley–Terry 拟合在每对每个方向加入 0.5 个伪胜，避免全胜/全负产生无穷等级分。",
              "这是本次条件下的相对等级分；每对不足 100 组（400 局）时仍为初步结果。",
              "模型、卡组池、规则、时序和代码身份见 study.json；逐局记录见 matches.jsonl。", ""]
    return "\n".join(lines)


def report(runs, output, *, samples=1000):
    identity, pairs, counts = collect_runs(runs)
    summary = round_robin_summary(pairs, identity["model_order"], anchor=identity["anchor"],
                                  bootstrap_samples=samples, seed=identity["gameplay"]["seed"])
    summary.update(counts)
    summary["source_runs"] = [str(Path(path).resolve()) for path in runs]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "study.json", identity)
    write_json(output / "summary.json", summary)
    with (output / "matches.jsonl").open("w", encoding="utf-8") as stream:
        for pair, groups in sorted(pairs.items()):
            for block, rows in sorted(groups.items()):
                for row in rows:
                    stream.write(json.dumps({**row, "models": pair}, ensure_ascii=False) + "\n")
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary), flush=True)
    print(f"报告目录：{output.resolve()}", flush=True)
    return summary


def run(args):
    if args.startup_timeout < 1:
        raise ValueError("startup-timeout must be positive")
    with ExitStack() as stack:
        if os.name == "posix":
            import fcntl
            endpoint = args.existing_engine or "automatic"
            key = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
            lock_name = (f"firstlight-rr-{os.getuid()}-{key}.lock" if args.existing_engine
                         else f"firstlight-duel-{os.getuid()}.lock")
            lock = stack.enter_context((Path(tempfile.gettempdir()) / lock_name).open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("同一引擎已有循环赛在运行") from None
        return _run(args)


def _run(args):
    from .duel import (check_engine_settings, choose_ports, stop_group, tail, wait_ready)

    options = read_options(args)
    config = args.config or ROOT / "round_robin.local.json"
    if config.is_file():
        print(f"配置文件：{config.resolve()}", flush=True)
    else:
        print(f"未找到 {config}，使用 round_robin.example.json 的默认设置。", flush=True)
    print(f"模型：{'、'.join(options['models'])}；采样 {options['sample']}，"
          f"出牌温度 {options['gate_temperature']}，选牌和落点温度 {options['card_tile_temperature']}。", flush=True)
    identity = study_identity(options)
    if args.engines < 1 or args.engines > 64:
        raise ValueError("engines must be in 1..64")
    if args.existing_engine and args.engines != 1:
        raise ValueError("--existing-engine requires --engines 1")
    parent = ROOT / "evaluations"
    parent.mkdir(exist_ok=True)
    history = []
    completed = {}
    historical_engine = None
    for directory in sorted(parent.glob("rr-*")):
        marker = directory / "study.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == identity:
            history.append(directory)
    if history:
        _, completed, metadata = collect_runs(history)
        historical_engine = metadata["engine_identity"]
    pending = {pair: [block for block in args.blocks if block not in completed.get(pair, {})]
               for pair in itertools.combinations(options["models"], 2)}
    if not any(pending.values()):
        print("指定组号已全部完成。", flush=True)
        report(history, parent / "round-robin-report", samples=options["bootstrap_samples"])
        return 0
    count = min(args.engines, len(args.blocks))
    if args.existing_engine:
        host, port = existing_endpoint(args.existing_engine)
        console = None
    else:
        check_engine_settings()
        host = "127.0.0.1"
        port, console = choose_ports(count)
    output = Path(tempfile.mkdtemp(prefix="rr-", dir=parent))
    write_json(output / "study.json", identity)
    # Tracing records extra files only; it is a local setting, not part of the study identity.
    write_json(output / "settings.json", {**options, "trace": bool(getattr(args, "trace", False))})
    write_json(output / "requested-blocks.json", args.blocks)
    if historical_engine is not None:
        write_json(output / "expected-engine.json", historical_engine)
    workers = []
    engine = None
    runtime = None
    interrupted = False
    error = None
    print(f"结果目录：{output}", flush=True)
    print(f"{len(options['models'])} 模型，{len(pending)} 对，{count} 引擎，每引擎 "
          f"{options['lanes']} 槽、{options['cpu_threads']} 推理线程。", flush=True)
    print(f"等级 {options['battle_level']}，延迟 {options['delay_ticks']} tick，间隔 "
          f"{options['min_play_gap_ticks']} tick，采样 {options['sample']}。", flush=True)
    started = time.perf_counter()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt
    for sig in previous:
        signal.signal(sig, interrupt)
    try:
        with ExitStack() as stack:
            # Validate Python dependencies, checkpoints and deck data before booting a VM.
            prepare = subprocess.run([sys.executable, "-m", "native_runner.training.v4.evaluate_round_robin",
                                      "--run", str(output), "--prepare-only"], cwd=ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if prepare.returncode:
                raise RuntimeError(prepare.stdout)
            try:
                if not args.existing_engine:
                    runtime = Path(tempfile.mkdtemp(prefix="firstlight-rr-"))
                    (runtime / "compat").mkdir()
                    node = output / "engine"
                    node.mkdir()
                    environment = {**os.environ, "CR_AI_ROOT": str(ROOT), "CR_AI_PYTHON": sys.executable,
                                   "CR_PPO_GUEST_PREFIX": f"crrr_{os.getuid()}", "CR_PPO_LOOPBACK_ONLY": "1",
                                   "CR_PPO_ENGINES_PER_GUEST": "1", "CR_PPO_CPU_OFFSET": "0",
                                   "CR_PPO_CORES_PER_ENGINE": str(min(8, max(1, (os.cpu_count() or 1) // count))),
                                   "CR_EMULATOR_COMPAT": os.environ.get("CR_EMULATOR_COMPAT") or str(runtime / "compat"),
                                   "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID") or f"rr-{os.getpid()}",
                                   "TMPDIR": str(runtime), "CUDA_VISIBLE_DEVICES": ""}
                    log = stack.enter_context((output / "engine.log").open("w", encoding="utf-8"))
                    engine = subprocess.Popen(["bash", str(ROOT / "native_runner/training/v4/launch_ppo_engine_node.sh"),
                                               str(node), "0", str(count), str(port), str(console)],
                                              cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    wait_ready(engine, node, args.startup_timeout)
                for index in range(count):
                    blocks = args.blocks[len(args.blocks) * index // count:len(args.blocks) * (index + 1) // count]
                    work = [{"models": pair, "blocks": [b for b in blocks if b in ids]}
                            for pair, ids in pending.items() if any(b in ids for b in blocks)]
                    if not work:
                        continue
                    shard = output / f"shard-{index:02d}"
                    shard.mkdir()
                    write_json(shard / "work.json", work)
                    log = stack.enter_context((shard / "evaluate.log").open("w", encoding="utf-8"))
                    worker = subprocess.Popen([sys.executable, "-u", "-m",
                                               "native_runner.training.v4.evaluate_round_robin", "--run", str(output),
                                               "--shard", str(shard), "--host", host, "--port", str(port + index)],
                                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    workers.append((worker, shard))
                last_notice = 0
                while any(worker.poll() is None for worker, _ in workers):
                    if any(worker.poll() not in (None, 0) for worker, _ in workers):
                        error = "评测进程失败"
                        break
                    if engine is not None and engine.poll() is not None:
                        raise RuntimeError("引擎进程退出，详见 engine.log")
                    if time.perf_counter() - last_notice >= 15:
                        statuses = []
                        for _, shard in workers:
                            if (shard / "progress.json").is_file():
                                status = json.loads((shard / "progress.json").read_text(encoding="utf-8"))
                                games = sum(len(read_rows(path)) for path in shard.glob("pair-*/matches.jsonl"))
                                status["stage"] += f"；本进程已记录 {games} 局"
                                statuses.append(status)
                        print(f"已用 {time.perf_counter() - started:.0f} 秒；"
                              + "；".join(f"{s['pair']}：{s['stage']}" for s in statuses), flush=True)
                        last_notice = time.perf_counter()
                    time.sleep(.5)
            finally:
                for worker, _ in workers:
                    stop_group(worker)
                if engine is not None:
                    (output / "engine" / "STOP_ENGINES").touch()
                    stop_group(engine)
    except KeyboardInterrupt:
        interrupted = True
    except (OSError, RuntimeError, ValueError) as exc:
        error = str(exc)
    finally:
        import shutil
        if runtime is not None:
            shutil.rmtree(runtime, ignore_errors=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    for worker, shard in workers:
        if worker.returncode:
            error = error or "评测进程未正常结束"
            tail(shard / "evaluate.log")
    report(history + [output], output / "report", samples=options["bootstrap_samples"])
    _, combined, _ = collect_runs(history + [output])
    complete = all(all(block in combined.get(pair, {}) for block in args.blocks) for pair in pending)
    code = 130 if interrupted else 1 if error or not complete else 0
    write_json(output / "status.json", dict(exit_code=code, error=error, elapsed_seconds=time.perf_counter() - started))
    if error:
        print(error, flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    merge = commands.add_parser("report", help="merge copied rr-* directories without an engine or model files")
    merge.add_argument("runs", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, default=Path("evaluations/round-robin-report"))
    merge.add_argument("--bootstrap-samples", type=int, default=1000)
    # Keep the everyday command short: no run subcommand required.
    parser.add_argument("--config", type=Path)
    parser.add_argument("--blocks", type=parse_blocks, default=parse_blocks("0-99"))
    parser.add_argument("--existing-engine", help="idle HOST:PORT; never start or stop this engine")
    parser.add_argument("--engines", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--deck-pool")
    parser.add_argument("--delay-ticks")
    parser.add_argument("--min-play-gap-ticks")
    parser.add_argument("--battle-level", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="also write a per-step combat trace of every game (pair-*/traces/) for behavior_stats; "
        "about 1-2 MB per game, results and resume identity unchanged",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            report(args.runs, args.output, samples=args.bootstrap_samples)
            return 0
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"循环赛失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
