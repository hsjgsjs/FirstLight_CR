#!/usr/bin/env python3
"""Check whether FirstLight's full observation can be read from a live battle.

Use your own accounts only (a friendly between them, or Training Camp).  The
probe's live mode is read-only: it attaches to the battle the game itself runs
and encodes FirstLight's complete atomic observation on the game thread.  This
script samples it during a battle, runs FirstLight's own validators on every
sample and stores the raw captures so they can be replayed through the whole
observation pipeline offline.

    python -m native_runner.live_diagnose start     # lift the offline block, restart the game, forward the port
    python -m native_runner.live_diagnose run       # then start a friendly / Training Camp battle
    python -m native_runner.live_diagnose offline   # put the network block back when done

Send the folder printed at the end (runs/live-diagnose/<time>/).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from .cr_native_env import NativeClashEnv, RunnerError, _validate_atomic_observation, _validate_rich_observation
from .local_config import setting
from .paths import WORKSPACE_ROOT

PACKAGE = "nullsroyale.rel.free"
ACTIVITY = "com.supercell.clashroyale.GameApp"
OFFLINE_CHAIN = "CR_NATIVE_OFFLINE"
CONTROL_HOST = setting("CR_CONTROL_HOST", "127.0.0.1")
CONTROL_PORT = int(setting("CR_CONTROL_PORT", "26789"))
GUEST_CONTROL_PORT = int(setting("CR_GUEST_CONTROL_PORT", "26789"))
OUTPUT_ROOT = WORKSPACE_ROOT / "runs" / "live-diagnose"


def _adb(*args: str, check: bool = True, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    adb = setting("CR_ADB", "adb") or "adb"
    serial = setting("CR_ADB_SERIAL")
    command = [adb, *(["-s", serial] if serial else []), *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {(result.stdout + result.stderr).strip()}")
    return result


def set_network(online: bool) -> None:
    """Remove (online) or restore (offline) the instance's CR_NATIVE_OFFLINE jump."""

    _adb("root")
    _adb("wait-for-device")
    for family in ("iptables", "ip6tables"):
        exists = _adb("shell", family, "-S", OFFLINE_CHAIN, check=False).returncode == 0
        if online:
            # Delete every jump; the chain itself stays for start_offline.ps1.
            while _adb("shell", family, "-D", "OUTPUT", "-j", OFFLINE_CHAIN, check=False).returncode == 0:
                pass
        elif exists and _adb("shell", family, "-C", "OUTPUT", "-j", OFFLINE_CHAIN, check=False).returncode != 0:
            _adb("shell", family, "-I", "OUTPUT", "1", "-j", OFFLINE_CHAIN)
    print("网络已放开（实战模式）" if online else "已恢复离线隔离")


def start_online() -> None:
    set_network(True)
    _adb("forward", f"tcp:{CONTROL_PORT}", f"tcp:{GUEST_CONTROL_PORT}")
    _adb("shell", "am", "force-stop", PACKAGE)
    _adb("shell", "am", "start", "-n", f"{PACKAGE}/{ACTIVITY}")
    env = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=5.0)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            env.status()
            print("游戏已联网启动，probe 控制端口就绪。进入大厅后运行：python -m native_runner.live_diagnose run")
            return
        except (RunnerError, OSError):
            time.sleep(1)
    raise RuntimeError("60 秒内没有连上 probe 控制端口；确认安装的是本次重新构建的离线 APK")


def summarize_capture(capture: dict[str, Any]) -> dict[str, Any]:
    ordinary = capture.get("ordinary") if isinstance(capture.get("ordinary"), dict) else {}
    players = []
    for player in ordinary.get("players", ()) or ():
        if not isinstance(player, dict):
            continue
        players.append(
            {
                "owner": player.get("owner"),
                "accountId": player.get("accountId"),
                "elixirRaw": player.get("elixirRaw"),
                "hand": len(player.get("hand") or ()),
                "deck": len(player.get("deck") or ()),
                "keys": sorted(player),
            }
        )
    return {
        "tick": capture.get("tick"),
        "generation": capture.get("generation"),
        "objects": ordinary.get("count"),
        "ordinaryKeys": sorted(ordinary),
        "richKeys": sorted(capture["rich"]) if isinstance(capture.get("rich"), dict) else None,
        "players": players,
    }


def check_capture(capture: dict[str, Any]) -> dict[str, Any]:
    """FirstLight's own validators, reported instead of raised."""

    report: dict[str, Any] = {}
    try:
        _validate_atomic_observation(json.loads(json.dumps(capture)))
        report["atomic"] = "ok"
    except Exception as error:  # noqa: BLE001 - every failure is the finding we want
        report["atomic"] = f"{type(error).__name__}: {error}"
    rich = capture.get("rich")
    if isinstance(rich, dict):
        try:
            _validate_rich_observation(json.loads(json.dumps(rich)))
            report["rich"] = "ok"
        except Exception as error:  # noqa: BLE001
            report["rich"] = f"{type(error).__name__}: {error}"
    return report


def run(samples: int, interval: float, wait: float) -> Path:
    env = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=30.0)
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    env._request("live on")
    try:
        print("实战模式已开启。现在开始一局友谊战或训练营……")
        deadline = time.monotonic() + wait
        while True:
            status = env._request("live status")
            if status.get("attached") and int(status.get("tick", -1)) >= 150:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"{wait:.0f} 秒内没有检测到进行中的对局：{status}")
            time.sleep(1)
        print(f"已挂上对局（tick {status.get('tick')}），开始采样 {samples} 次……")
        for index in range(samples):
            started = time.monotonic()
            entry: dict[str, Any] = {"sample": index}
            try:
                capture = env._request("live observe-atomic")
            except RunnerError as error:
                entry["error"] = str(error)
                print(f"  样本 {index}: 采集失败：{error}")
            else:
                path = output / f"sample-{index}.json.gz"
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    json.dump(capture, stream, ensure_ascii=False)
                entry["file"] = path.name
                entry["bytes"] = len(json.dumps(capture))
                entry["captureMs"] = round((time.monotonic() - started) * 1000)
                entry["summary"] = summarize_capture(capture)
                entry["validation"] = check_capture(capture)
                print(f"  样本 {index}: tick {capture.get('tick')}，校验 atomic={entry['validation'].get('atomic')}")
            results.append(entry)
            time.sleep(interval)
    finally:
        try:
            env._request("live off")
        except (RunnerError, OSError):
            pass
    (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"完成。请把整个文件夹发给我：{output}")
    return output


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="lift the offline block, restart the game online, forward the control port")
    run_parser = sub.add_parser("run", help="sample full observations from the battle in progress")
    run_parser.add_argument("--samples", type=int, default=5)
    run_parser.add_argument("--interval", type=float, default=3.0, help="seconds between samples")
    run_parser.add_argument("--wait", type=float, default=600.0, help="seconds to wait for a battle to start")
    sub.add_parser("offline", help="restore the instance's network block")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "start":
        start_online()
    elif args.command == "offline":
        set_network(False)
    else:
        run(args.samples, args.interval, args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
