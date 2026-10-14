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
    for key in ("checkpoint_a", "checkpoint_b", "blocks", "opening_seconds"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if not isinstance(options["deterministic"], bool):
        raise ValueError("deterministic 必须为 true 或 false")
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
        if value is None or (key == "deterministic" and not value):
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


def choose_ports():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        control = sock.getsockname()[1]
    for console in range(5682, 5553, -2):
        if control in (console, console + 1):
            continue
        try:
            free_ports((console, console + 1))
            return control, console
        except OSError:
            continue
    raise RuntimeError("没有空闲的模拟器端口，请先关闭一个不用的模拟器")


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


def wait_evaluation(process, engine, results, total):
    stream = None
    try:
        while True:
            code = process.poll()
            if stream is None and (results / "matches.jsonl").is_file():
                stream = (results / "matches.jsonl").open(encoding="utf-8")
            if stream is not None:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        stream.seek(position)
                        break
                    row = json.loads(line)
                    condition = "正常" if row["condition"] == "natural" else "强制"
                    outcome = "未正常结束" if row["truncated"] or not row["terminated"] else {
                        "a": "A 胜", "b": "B 胜", None: "平局",
                    }[row["winner"]]
                    say(f"[{row['game']}/{total}] 第 {row['block'] + 1} 组 · {condition} · {outcome}")
            if code is not None:
                return code
            if engine is not None and engine.poll() is not None:
                raise RuntimeError("模拟引擎在评测中退出，详见 engine.log")
            time.sleep(1)
    finally:
        if stream is not None:
            stream.close()


def run(args):
    load_env()
    options = read_options(args.config.expanduser().resolve(), args)
    if args.startup_timeout < 1:
        raise ValueError("启动等待时间必须大于零")
    host, port, console = "127.0.0.1", None, None
    if args.existing_engine:
        host, separator, value = args.existing_engine.rpartition(":")
        if not separator or not host or not value.isdigit() or not 1 <= int(value) <= 65535:
            raise ValueError("--existing-engine 格式为 HOST:PORT")
        host, port = host.strip("[]"), int(value)
    else:
        check_engine_settings()
    with ExitStack() as stack:
        if not args.existing_engine:
            import fcntl

            lock = stack.enter_context(open(f"/tmp/firstlight-duel-{os.getuid()}.lock", "a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("已有一个自动对弈任务正在运行，请等它结束后再启动") from None
            port, console = choose_ports()
        parent = ROOT / "evaluations"
        parent.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("duel-%Y%m%d-%H%M%S-"), dir=parent))
        results = output / "results"
        node = output / "engine"
        node.mkdir()
        (parent / "duel-latest.txt").write_text(str(output) + "\n", encoding="utf-8")
        (output / "settings.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
        say(f"结果目录：{output}\n配置：{args.config}\n共 {options['blocks']} 组，每组 8 局。")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
        engine = evaluation = None
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
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
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
                    **env, "CR_AI_ROOT": str(ROOT), "CR_AI_PYTHON": sys.executable,
                    "CR_PPO_GUEST_PREFIX": f"crduel_{os.getuid()}", "CR_PPO_LOOPBACK_ONLY": "1",
                    "CR_PPO_ENGINES_PER_GUEST": "1", "CR_PPO_CPU_OFFSET": "0",
                    "CR_EMULATOR_COMPAT": os.environ.get("CR_EMULATOR_COMPAT") or str(compat),
                    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID") or f"duel-{os.getpid()}",
                    "TMPDIR": str(runtime),
                }
                engine_log = stack.enter_context((output / "engine.log").open("w", encoding="utf-8"))
                say(f"启动 CPU 模拟引擎，自动选择端口 {port}。启动日志：{output / 'engine.log'}")
                engine = subprocess.Popen(
                    ["bash", str(ROOT / "native_runner/training/v4/launch_ppo_engine_node.sh"),
                     str(node), "0", "1", str(port), str(console)],
                    cwd=ROOT, env=engine_env, stdout=engine_log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                wait_ready(engine, node, args.startup_timeout)
            say("开始对弈。按 Ctrl+C 可停止并清理本次任务。")
            log = stack.enter_context((output / "evaluate.log").open("w", encoding="utf-8"))
            evaluation = subprocess.Popen(
                evaluator_command(options, host=host, port=port, output=results),
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            code = wait_evaluation(evaluation, engine, results, int(options["blocks"]) * 8)
            if (results / "report.md").is_file():
                say((results / "report.md").read_text(encoding="utf-8"))
            if code:
                tail(output / "evaluate.log")
                error = "评测未全部完成，请查看 evaluate.log 和已保存的逐局记录"
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
            (output / "status.json").write_text(
                json.dumps({"exit_code": code, "error": error, "results": str(results)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
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
