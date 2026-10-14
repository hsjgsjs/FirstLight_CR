"""One-command lifecycle for CPU opening evaluations on a prepared Linux host."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

from .local_config import load_env


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "duel.local.json"
PATH_KEYS = {"checkpoint_a", "checkpoint_b", "match_config", "deck_pool"}


def say(message):
    print(message, flush=True)


def read_options(path, args):
    defaults = json.loads((ROOT / "duel.example.json").read_text(encoding="utf-8"))
    if not path.exists() and path == DEFAULT_CONFIG:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(defaults, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        say(f"已生成本机配置：{path}。以后可直接修改这个文件。")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.keys() - defaults.keys():
        raise ValueError("配置必须是 JSON 对象，且只能包含 duel.example.json 中的参数名")
    options = {**defaults, **saved}
    for key in ("checkpoint_a", "checkpoint_b", "blocks", "opening_seconds", "engines"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if not isinstance(options["deterministic"], bool):
        raise ValueError("deterministic 必须为 true 或 false")
    for key in ("engines", "blocks", "cpu_threads"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"{key} 必须是正整数")
    if options["engines"] > 64:
        raise ValueError("engines 最多为 64")
    for key in PATH_KEYS:
        value = options[key]
        if value is None and key not in ("checkpoint_a", "checkpoint_b"):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 必须是非空路径")
        candidate = Path(value).expanduser()
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        if key.startswith("checkpoint_") and candidate.is_dir():
            matches = sorted(candidate.glob("checkpoint-step-*.pt")) or sorted(candidate.glob("*.pt"))
            if not matches:
                raise ValueError(f"找不到模型：{candidate}")
            candidate = matches[-1]
        if not candidate.is_file():
            raise ValueError(f"文件不存在：{candidate}")
        options[key] = str(candidate.resolve())
    return options


def evaluator_command(options, *, host, port, output):
    command = [sys.executable, "-m", "native_runner.training.v4.evaluate_opening"]
    for key, value in options.items():
        if key == "engines" or value is None or (key == "deterministic" and not value):
            continue
        command.append("--" + key.replace("_", "-"))
        if key != "deterministic":
            if isinstance(value, (dict, list, bool)):
                raise ValueError(f"配置项 {key} 的类型不正确")
            command.append(str(value))
    return command + ["--host", host, "--port", str(port), "--output", str(output)]


def free_ports(ports):
    with ExitStack() as stack:
        for port in ports:
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", port))
        return True


def choose_ports(engines):
    control = None
    for _ in range(100):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            candidate = sock.getsockname()[1]
        if candidate + engines > 65536:
            continue
        try:
            free_ports(range(candidate, candidate + engines))
            control = candidate
            break
        except OSError:
            continue
    if control is None:
        raise RuntimeError("没有找到连续空闲的引擎控制端口")
    # One guest per engine works with single-engine APKs as well as cluster APKs.
    for console in range(5682 - 2 * (engines - 1), 5553, -2):
        if console < control + engines and control < console + 2 * engines:
            continue
        try:
            free_ports(range(console, console + 2 * engines))
            return control, console
        except OSError:
            continue
    raise RuntimeError("没有足够的连续模拟器端口，请减少 engines 或先关闭不用的模拟器")


def check_engine_settings():
    if sys.platform != "linux":
        raise ValueError("自动启动引擎需要已准备好的 Linux 模拟环境；现有引擎可用 --existing-engine HOST:PORT")
    required = (
        "CR_EMULATOR", "CR_SOURCE_AVD", "CR_SOURCE_AVD_INI",
        "CR_EMULATOR_INITDATA", "CR_TRAINING_APK", "CR_TRAINING_SDK_ROOT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("请先在 .env 填好模拟环境路径：" + ", ".join(missing))
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise ValueError("当前用户无法读写 /dev/kvm；自动启动需要可用的 Linux KVM 模拟环境")
    for name in ("bash", "taskset", os.environ.get("CR_TRAINING_ADB") or "adb"):
        if shutil.which(name) is None:
            raise ValueError(f"找不到引擎所需程序：{name}")


def stop_group(process):
    """Signal only a child process group created by this launcher."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Also reap descendants if the shell has exited before its guest did.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def tail(path):
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 6000))
            say(stream.read().decode("utf-8", errors="replace"))


def wait_ready(process, node, timeout):
    started = last_notice = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError("模拟引擎启动失败，详见 engine.log 和 engine/ 下的日志")
        ready = node / "engine-node-ready.json"
        if ready.is_file():
            try:
                value = json.loads(ready.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                value = {}
            if value.get("ok") is True:
                return
        if time.monotonic() - last_notice >= 15:
            say(f"正在启动模拟引擎……已等待 {int(time.monotonic() - started)} 秒")
            last_notice = time.monotonic()
        time.sleep(1)
    raise RuntimeError(f"引擎在 {timeout} 秒内未就绪，详见 engine.log")


def split_blocks(blocks, engines):
    """Contiguous, complete paired blocks; never split the eight games across engines."""
    quotient, remainder = divmod(blocks, engines)
    start = 0
    shards = []
    for index in range(engines):
        count = quotient + int(index < remainder)
        if count:
            shards.append((start, count))
            start += count
    return shards


def wait_evaluations(workers, engine, total):
    streams = {}
    completed = 0
    with ExitStack() as stack:
        while True:
            codes = [worker["process"].poll() for worker in workers]
            for index, worker in enumerate(workers):
                path = worker["results"] / "matches.jsonl"
                if index not in streams and path.is_file():
                    streams[index] = stack.enter_context(path.open(encoding="utf-8"))
                if index not in streams:
                    continue
                stream = streams[index]
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        stream.seek(position)
                        break
                    row = json.loads(line)
                    completed += 1
                    condition = "正常" if row["condition"] == "natural" else "强制"
                    outcome = "未正常结束" if row["truncated"] or not row["terminated"] else {
                        "a": "A 胜", "b": "B 胜", None: "平局",
                    }[row["winner"]]
                    say(f"[{completed}/{total}] 引擎 {index + 1} · 第 {row['block'] + 1} 组 · {condition} · {outcome}")
            # Fail promptly and let cleanup stop the other workers. Completed
            # blocks remain usable; a dead worker must not leave a partial run looking successful.
            if any(code is not None and code != 0 for code in codes):
                return 1
            if all(code is not None for code in codes):
                return 0
            if engine is not None and engine.poll() is not None:
                raise RuntimeError("模拟引擎在评测中退出，详见 engine.log")
            time.sleep(1)


def combine_results(workers, results, options):
    from .training.v4.evaluate_opening import render_report
    from .training.v4.evaluation_protocol import paired_summary

    rows = []
    identities = []
    for index, worker in enumerate(workers):
        path = worker["results"] / "matches.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break  # A process interrupted mid-write has no complete record here.
                    row = json.loads(line)
                    row["game"] += worker["block_start"] * 8
                    row["engine"] = index + 1
                    rows.append(row)
        identities.append({
            "engine": index + 1, "block_start": worker["block_start"], "blocks": worker["blocks"],
            "results": str(worker["results"]), "exit_code": worker["process"].poll(),
        })
    rows.sort(key=lambda row: row["game"])
    summary = paired_summary(rows, max_score_drop=float(options["max_score_drop"]), seed=int(options["seed"]))
    summary.update(
        requested_blocks=options["blocks"], requested_games=options["blocks"] * 8,
        missing_blocks=options["blocks"] - len({row["block"] for row in rows}), workers=identities,
    )
    with (results / "matches.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (results / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = render_report(summary)
    if summary["complete_blocks"] != options["blocks"]:
        report = f"本次评测未全部完成：完整组 {summary['complete_blocks']}/{options['blocks']}，以下仅为已完成数据。\n\n" + report
    (results / "report.md").write_text(report, encoding="utf-8")
    return summary


def run(args):
    load_env()
    options = read_options(args.config.expanduser().resolve(), args)
    engine_count = min(options["engines"], options["blocks"])
    partitions = split_blocks(options["blocks"], engine_count)
    if args.startup_timeout < 1:
        raise ValueError("启动等待时间必须大于零")
    host, port, console = "127.0.0.1", None, None
    if args.existing_engine:
        if options["engines"] != 1:
            raise ValueError("--existing-engine 只接受一个引擎，请同时设置 --engines 1")
        host, separator, value = args.existing_engine.rpartition(":")
        if not separator or not host or not value.isdigit() or not 1 <= int(value) <= 65535:
            raise ValueError("--existing-engine 格式为 HOST:PORT")
        host, port = host.strip("[]"), int(value)
    else:
        check_engine_settings()
        available_cpus = len(os.sched_getaffinity(0))
        if engine_count > available_cpus:
            raise ValueError(f"本机可用 CPU 为 {available_cpus} 个，请把 engines 调到这个数以内")
    with ExitStack() as stack:
        if not args.existing_engine:
            import fcntl

            lock = stack.enter_context(open(f"/tmp/firstlight-duel-{os.getuid()}.lock", "a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("已有一个自动对弈任务正在运行，请等它结束后再启动") from None
            port, console = choose_ports(engine_count)
        parent = ROOT / "evaluations"
        parent.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("duel-%Y%m%d-%H%M%S-"), dir=parent))
        results = output / "results"
        if engine_count > 1:
            results.mkdir()
        node = output / "engine"
        node.mkdir()
        (parent / "duel-latest.txt").write_text(str(output) + "\n", encoding="utf-8")
        (output / "settings.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
        say(f"结果目录：{output}\n配置：{args.config}\n共 {options['blocks']} 组，每组 8 局，使用 {engine_count} 个引擎。")
        eval_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        engine_env_base = {**eval_env, "CUDA_VISIBLE_DEVICES": ""}
        engine = evaluation = None
        workers = []
        runtime = None
        code, error = 1, None
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous_handlers = {sig: signal.getsignal(sig) for sig in signals}

        def interrupt(_signal, _frame):
            raise KeyboardInterrupt

        for sig in previous_handlers:
            signal.signal(sig, interrupt)
        try:
            # Parse all evaluation options and deck data before spending time booting a VM.
            with (output / "prepare.log").open("w", encoding="utf-8") as log:
                evaluation = subprocess.Popen(
                    evaluator_command(options, host=host, port=port, output=output / "plan-check") + ["--plan-only"],
                    cwd=ROOT, env=eval_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                if evaluation.wait():
                    tail(output / "prepare.log")
                    raise RuntimeError("评测参数或 Python 依赖有问题，请查看 prepare.log")
                evaluation = None
            if not args.existing_engine:
                runtime = Path(tempfile.mkdtemp(prefix="firstlight-duel-", dir="/tmp"))
                compat = runtime / "compat"
                compat.mkdir()
                engine_env = {
                    **engine_env_base, "CR_AI_ROOT": str(ROOT), "CR_AI_PYTHON": sys.executable,
                    "CR_PPO_GUEST_PREFIX": f"crduel_{os.getuid()}", "CR_PPO_LOOPBACK_ONLY": "1",
                    "CR_PPO_ENGINES_PER_GUEST": "1", "CR_PPO_CPU_OFFSET": "0",
                    "CR_EMULATOR_COMPAT": os.environ.get("CR_EMULATOR_COMPAT") or str(compat),
                    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID") or f"duel-{os.getpid()}",
                    "TMPDIR": str(runtime),
                }
                engine_log = stack.enter_context((output / "engine.log").open("w", encoding="utf-8"))
                say(f"启动 {engine_count} 个 CPU 模拟引擎，端口 {port}–{port + engine_count - 1}。启动日志：{output / 'engine.log'}")
                engine = subprocess.Popen(
                    ["bash", str(ROOT / "native_runner/training/v4/launch_ppo_engine_node.sh"),
                     str(node), "0", str(engine_count), str(port), str(console)],
                    cwd=ROOT, env=engine_env, stdout=engine_log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                wait_ready(engine, node, args.startup_timeout)
            say("开始对弈。按 Ctrl+C 可停止并清理本次任务。")
            for index, (block_start, count) in enumerate(partitions):
                destination = results if engine_count == 1 else results / f"shard-{index:02d}"
                log_path = output / ("evaluate.log" if engine_count == 1 else f"evaluate-{index:02d}.log")
                log = stack.enter_context(log_path.open("w", encoding="utf-8"))
                shard_options = {**options, "blocks": count, "block_start": block_start}
                evaluation = subprocess.Popen(
                    evaluator_command(shard_options, host=host, port=port + index, output=destination),
                    cwd=ROOT, env=eval_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                workers.append({
                    "process": evaluation, "results": destination, "log": log_path,
                    "block_start": block_start, "blocks": count,
                })
                evaluation = None
            code = wait_evaluations(workers, engine, options["blocks"] * 8)
            if code:
                for worker in workers:
                    if worker["process"].poll() not in (None, 0):
                        tail(worker["log"])
                error = "评测未全部完成，请查看 evaluate*.log 和已保存的逐局记录"
        except KeyboardInterrupt:
            code, error = 130, "已停止，已完成对局的记录会保留"
        except (OSError, ValueError, RuntimeError) as exc:
            code, error = 1, str(exc)
            tail(output / "engine.log")
        finally:
            # Ignore repeated terminal signals while terminating our own child groups.
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            stop_group(evaluation)
            for worker in workers:
                stop_group(worker["process"])
            if engine is not None:
                say("正在关闭本次启动的模拟引擎……")
                (node / "STOP_ENGINES").touch()
                try:
                    engine.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                stop_group(engine)
            if runtime is not None:
                try:
                    shutil.rmtree(runtime)
                except OSError as exc:
                    say(f"临时目录未能完全删除：{runtime}（{exc}）")
            if engine_count > 1 and workers:
                try:
                    summary = combine_results(workers, results, options)
                    if summary["complete_blocks"] != options["blocks"] and code == 0:
                        code, error = 1, "评测缺少完整配对组，请查看合并报告"
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    code, error = 1, f"合并结果失败，分引擎原始记录已保留：{exc}"
            (output / "status.json").write_text(
                json.dumps({"exit_code": code, "error": error, "results": str(results)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if (results / "report.md").is_file():
            say((results / "report.md").read_text(encoding="utf-8"))
        if error:
            say(error)
        say(f"本次记录：{output}")
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description="一键 CPU 对弈：准备引擎、正常/强制开局对照、报告和清理。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="本机配置，默认首次自动创建 duel.local.json")
    parser.add_argument("--checkpoint-a", help="本次使用的参考模型；不修改保存的配置")
    parser.add_argument("--checkpoint-b", help="本次使用的被测对手；不修改保存的配置")
    parser.add_argument("--blocks", type=int, help="本次组数，每组 8 局")
    parser.add_argument("--engines", type=int, help="并行引擎数，默认读取本机配置（初始为 1）")
    parser.add_argument("--opening-seconds", help="本次 B 的开局超时，例如 8-20 或 12")
    parser.add_argument("--existing-engine", help="使用已有空闲引擎 HOST:PORT，不启动或关闭该引擎")
    parser.add_argument("--startup-timeout", type=int, default=900, help="引擎启动等待秒数，默认 900")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        say(f"无法开始对弈：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
