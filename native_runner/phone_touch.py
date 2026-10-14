#!/usr/bin/env python3
"""Play the offline engine on a real Android phone instead of MuMu.

The phone runs the patched game (rendering and touch); this PC keeps running
the console, the model and the native control client.  They talk through
``adb forward`` (USB or wireless debugging) to the probe's loopback port, so
nothing on the phone needs root.

On the phone the stock spectator HUD is the controller: tap or drag a card
onto the arena, and tap your own ready Champion/Hero on the arena to cast its
ability.  The touch geometry differs per phone, so ``calibrate`` measures it
once on the device and stores it locally.

    python -m native_runner.phone_touch connect      # forward port, start the game
    python -m native_runner.phone_touch calibrate    # with a match on screen
    python -m native_runner.phone_touch apply --owner 1
    python -m native_runner.phone_touch status
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .cr_native_env import TOUCH_LAYOUT_FIELDS, NativeClashEnv, RunnerError
from .local_config import setting
from .paths import WORKSPACE_ROOT

PACKAGE = "nullsroyale.rel.free"
ACTIVITY = "com.supercell.clashroyale.GameApp"
CONTROL_HOST = setting("CR_CONTROL_HOST", "127.0.0.1")
CONTROL_PORT = int(setting("CR_CONTROL_PORT", "26789"))
GUEST_CONTROL_PORT = int(setting("CR_GUEST_CONTROL_PORT", "26789"))
LAYOUT_PATH = WORKSPACE_ROOT / "runs" / "interface" / "phone_touch_layout.json"

# The probe's built-in MuMu layout (native_touch_layout.h, same field order).
DEFAULT_LAYOUT = (1080, 1920, 54, 282, 1026, 1666, 111, 436, 967, 1666, 171, 1749, 67, 86, 211, 354, 497, 640)
_DEFAULT_ARENA = DEFAULT_LAYOUT[2:6]
_DEFAULT_WORLD = DEFAULT_LAYOUT[6:10]
_DEFAULT_CARD_SPACING = (DEFAULT_LAYOUT[17] - DEFAULT_LAYOUT[14]) / 3

# Ground-level and tower calibration targets in native world units.
LEFT_BRIDGE = (3_500, 16_000)
RIGHT_BRIDGE = (14_500, 16_000)
TOP_LEFT_PRINCESS = (3_500, 6_500)
BOTTOM_RIGHT_PRINCESS = (14_500, 25_500)
WORLD_WIDTH = 18_000
WORLD_HEIGHT = 32_000


def phone_mode_enabled() -> bool:
    return setting("CR_DEVICE_MODE", "mumu").strip().casefold() == "phone"


@dataclass(frozen=True, slots=True)
class CalibrationPoints:
    """Raw touch coordinates measured on the phone's game surface."""

    screen: tuple[int, int]
    bottom_first_card: tuple[int, int]
    bottom_last_card: tuple[int, int]
    top_card: tuple[int, int]
    left_bridge: tuple[int, int]
    right_bridge: tuple[int, int]
    top_left_tower: tuple[int, int]
    bottom_right_tower: tuple[int, int]


def layout_from_calibration(points: CalibrationPoints) -> tuple[int, ...]:
    """Build the probe's 18-value layout from seven measured taps.

    Bridges are on the ground, so they fix the horizontal scale and both
    origins.  Princess-tower sprites stand above their ground point by the
    same amount, so only their *difference* is used, for the vertical scale.
    """

    width, height = points.screen
    if width <= 0 or height <= 0:
        raise ValueError("screen size must be positive")
    scale_x = (points.right_bridge[0] - points.left_bridge[0]) / (RIGHT_BRIDGE[0] - LEFT_BRIDGE[0])
    scale_y = (points.bottom_right_tower[1] - points.top_left_tower[1]) / (
        BOTTOM_RIGHT_PRINCESS[1] - TOP_LEFT_PRINCESS[1]
    )
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("calibration taps are out of order (left/right or top/bottom swapped)")
    bridge_x = (points.left_bridge[0] + points.right_bridge[0]) / 2
    bridge_y = (points.left_bridge[1] + points.right_bridge[1]) / 2
    origin_x = bridge_x - (LEFT_BRIDGE[0] + RIGHT_BRIDGE[0]) / 2 * scale_x
    origin_y = bridge_y - LEFT_BRIDGE[1] * scale_y
    world = (origin_x, origin_y, origin_x + WORLD_WIDTH * scale_x, origin_y + WORLD_HEIGHT * scale_y)

    # Keep the stock HUD's decorative catchment proportional to the grid.
    default_world_width = _DEFAULT_WORLD[2] - _DEFAULT_WORLD[0]
    default_world_height = _DEFAULT_WORLD[3] - _DEFAULT_WORLD[1]
    world_width = world[2] - world[0]
    world_height = world[3] - world[1]
    arena = (
        world[0] - (_DEFAULT_WORLD[0] - _DEFAULT_ARENA[0]) / default_world_width * world_width,
        world[1] - (_DEFAULT_WORLD[1] - _DEFAULT_ARENA[1]) / default_world_height * world_height,
        world[2] + (_DEFAULT_ARENA[2] - _DEFAULT_WORLD[2]) / default_world_width * world_width,
        world[3] + (_DEFAULT_ARENA[3] - _DEFAULT_WORLD[3]) / default_world_height * world_height,
    )

    first_x, bottom_y = points.bottom_first_card
    last_x, last_y = points.bottom_last_card
    spacing = (last_x - first_x) / 3
    if spacing <= 0:
        raise ValueError("tap the leftmost bottom card first, then the rightmost one")
    card_scale = spacing / _DEFAULT_CARD_SPACING
    bottom_hand_y = (bottom_y + last_y) / 2
    top_hand_y = points.top_card[1]

    def clamp(value: float, limit: int) -> int:
        return max(0, min(limit - 1, round(value)))

    layout = (
        width,
        height,
        clamp(arena[0], width),
        clamp(arena[1], height),
        clamp(arena[2], width),
        clamp(arena[3], height),
        clamp(world[0], width),
        clamp(world[1], height),
        clamp(world[2], width),
        clamp(world[3], height),
        clamp(top_hand_y, height),
        clamp(bottom_hand_y, height),
        max(1, round(DEFAULT_LAYOUT[12] * card_scale)),
        max(1, round(DEFAULT_LAYOUT[13] * card_scale)),
        *(clamp(first_x + index * spacing, width) for index in range(4)),
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: Sequence[int]) -> None:
    """Host mirror of cr_native_touch::layout_is_valid for early, clear errors."""

    if len(layout) != TOUCH_LAYOUT_FIELDS:
        raise ValueError(f"layout needs {TOUCH_LAYOUT_FIELDS} values")
    width, height, al, at, ar, ab, wl, wt, wr, wb, top_y, bottom_y, half_w, half_h, *centers = layout
    if not (0 < width <= 16384 and 0 < height <= 16384):
        raise ValueError("screen size is out of range")
    xs_inside = all(0 <= value < width for value in (al, ar, wl, wr, *centers))
    ys_inside = all(0 <= value < height for value in (at, ab, wt, wb, top_y, bottom_y))
    if not xs_inside or not ys_inside:
        raise ValueError("layout extends beyond the screen")
    if ar <= al or ab <= at or wr - wl < 18 or wb - wt < 32:
        raise ValueError("arena or world rectangle is degenerate")
    if top_y >= bottom_y or half_w <= 0 or half_h <= 0 or 2 * half_h > bottom_y - top_y:
        raise ValueError("hand rows overlap")
    if any(right - left < 2 * half_w for left, right in zip(centers, centers[1:])):
        raise ValueError("hand cards overlap")


def save_layout(layout: Sequence[int], path: Path = LAYOUT_PATH) -> Path:
    validate_layout(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "firstlight-phone-touch.v1", "layout": list(layout)}) + "\n", "utf-8")
    return path


def load_layout(path: Path = LAYOUT_PATH) -> tuple[int, ...] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    layout = tuple(int(value) for value in payload["layout"])
    validate_layout(layout)
    return layout


def apply_touch_settings(env: NativeClashEnv, *, owner: int | None, path: Path = LAYOUT_PATH) -> dict[str, Any]:
    """Send the saved phone layout, the owner lock and ability taps to the probe."""

    layout = load_layout(path)
    if layout is None:
        try:
            screen = screen_size()
            if screen and screen != (1080, 1920):
                scale = screen[0] / 1080
                candidate = tuple(round(x * scale) for x in DEFAULT_LAYOUT)
                validate_layout(candidate)
                layout = candidate
        except Exception:
            pass
    if layout is not None:
        _require_ok(env.set_touch_layout(layout), "touch layout")
    _require_ok(env.set_touch_calibrating(False), "touch calibrate")
    _require_ok(env.set_touch_ability_tap(True), "touch ability")
    return _require_ok(env.set_touch_owner(owner), "touch owner")


def _require_ok(result: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not result.get("ok"):
        raise RunnerError(f"{label} rejected: {result.get('error', result)}")
    return dict(result)


# --- adb lifecycle ---------------------------------------------------------


def _adb_command() -> list[str]:
    adb = setting("CR_ADB", "adb") or "adb"
    serial = setting("CR_PHONE_SERIAL") or setting("CR_ADB_SERIAL")
    return [adb, *(["-s", serial] if serial else [])]


def _adb(*args: str, timeout: float = 30.0, check: bool = True) -> str:
    result = subprocess.run([*_adb_command(), *args], capture_output=True, text=True, timeout=timeout)
    output = (result.stdout + result.stderr).strip()
    if check and result.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {output}")
    return output


def screen_size() -> tuple[int, int] | None:
    """Return the phone's current display size (override first) from ``wm size``."""

    text = _adb("shell", "wm", "size", check=False)
    sizes = re.findall(r"(Override|Physical) size:\s*(\d+)x(\d+)", text)
    for kind in ("Override", "Physical"):
        for found_kind, width, height in sizes:
            if found_kind == kind:
                return int(width), int(height)
    return None


def _cold_ready(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=1.0) as connection:
                connection.sendall(b"status\n")
                line = connection.makefile("r", encoding="utf-8").readline()
            status = json.loads(line)
            if status.get("ok") and status.get("coldReady"):
                return True
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    return False


def start_phone_runner(progress: Callable[[str], None], *, force_restart: bool = False) -> None:
    """Forward the control port to the phone and (re)start the offline game."""

    progress("正在连接手机并转发 native 控制端口…")
    _adb("wait-for-device", timeout=60.0)
    abi = _adb("shell", "getprop", "ro.product.cpu.abi")
    if abi.strip() != "arm64-v8a":
        raise RuntimeError(f"手机 ABI 为 {abi.strip()!r}；离线引擎只支持 arm64-v8a")
    _adb("forward", f"tcp:{CONTROL_PORT}", f"tcp:{GUEST_CONTROL_PORT}")
    if not force_restart and _cold_ready(1.5):
        progress("手机上的离线 native runner 已就绪，直接复用。")
        return
    progress("正在手机上重启离线游戏…")
    _adb("shell", "am", "force-stop", PACKAGE)
    _adb("shell", "am", "start", "-n", f"{PACKAGE}/{ACTIVITY}")
    if not _cold_ready(60.0):
        raise RuntimeError("手机上的 native 控制端口 60 秒内没有就绪；请确认已安装离线 APK 且游戏在前台")
    progress("手机上的离线 native runner 已就绪。请保持屏幕常亮、游戏在前台。")


# --- calibration -------------------------------------------------------------

CALIBRATION_STEPS = (
    ("bottom_first_card", "点一下【下方】手牌最左边那张卡的中心"),
    ("bottom_last_card", "点一下【下方】手牌最右边那张卡（第 4 张）的中心"),
    ("top_card", "点一下【上方】任意一张手牌的中心"),
    ("left_bridge", "点一下【左边桥】的中心"),
    ("right_bridge", "点一下【右边桥】的中心"),
    ("top_left_tower", "点一下【上方左侧公主塔】的中心"),
    ("bottom_right_tower", "点一下【下方右侧公主塔】的中心"),
)


def _wait_for_tap(env: NativeClashEnv, after_sequence: int, timeout: float = 120.0) -> tuple[int, int, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = env.touch_status()
        last_up = status.get("lastUp") or {}
        sequence = int(last_up.get("sequence", 0))
        if sequence > after_sequence:
            return sequence, int(last_up["x"]), int(last_up["y"])
        time.sleep(0.05)
    raise TimeoutError("没有收到手机上的点击")


def calibrate(env: NativeClashEnv, *, screen: tuple[int, int] | None, output: Callable[[str], None] = print) -> tuple[int, ...]:
    status = env.touch_status()
    if "lastUp" not in status:
        raise RunnerError("手机上的 probe 版本过旧：请用当前代码重新构建并安装离线 APK")
    _require_ok(env.set_touch_calibrating(True), "touch calibrate")
    measured: dict[str, tuple[int, int]] = {}
    try:
        sequence = int((status.get("lastUp") or {}).get("sequence", 0))
        output("校准中：这期间点击不会出牌。请在手机上依次点击：")
        for name, prompt in CALIBRATION_STEPS:
            output(f"  → {prompt}")
            sequence, x, y = _wait_for_tap(env, sequence)
            measured[name] = (x, y)
            output(f"    记录 ({x}, {y})")
    finally:
        env.set_touch_calibrating(False)
    if screen is None:
        widest = max(x for x, _ in measured.values())
        tallest = max(y for _, y in measured.values())
        screen = (max(widest + 1, round(widest * 1.1)), max(tallest + 1, round(tallest * 1.1)))
    layout = layout_from_calibration(CalibrationPoints(screen=screen, **measured))
    _require_ok(env.set_touch_layout(layout), "touch layout")
    return layout


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    connect = sub.add_parser("connect", help="adb forward the control port and start the offline game")
    connect.add_argument("--restart", action="store_true")
    calibrate_parser = sub.add_parser("calibrate", help="measure the phone's touch layout (match must be on screen)")
    calibrate_parser.add_argument("--screen", help="game surface size WxH; default: adb wm size")
    apply = sub.add_parser("apply", help="send the saved layout and owner lock to the running game")
    apply.add_argument("--owner", type=int, choices=(-1, 0, 1), default=-1, help="-1 = control both sides")
    sub.add_parser("reset", help="restore the built-in MuMu layout")
    sub.add_parser("status", help="print the probe's touch status")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "connect":
        start_phone_runner(print, force_restart=args.restart)
        return 0
    env = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=10.0)
    if args.command == "calibrate":
        if args.screen:
            width, height = (int(value) for value in args.screen.lower().split("x"))
            screen: tuple[int, int] | None = (width, height)
        else:
            try:
                screen = screen_size()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                screen = None
        layout = calibrate(env, screen=screen)
        print(f"已保存并应用：{save_layout(layout)}")
        print(json.dumps(list(layout)))
    elif args.command == "apply":
        result = apply_touch_settings(env, owner=None if args.owner < 0 else args.owner)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "reset":
        print(json.dumps(env.set_touch_layout(None), ensure_ascii=False))
    else:
        print(json.dumps(env.touch_status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
