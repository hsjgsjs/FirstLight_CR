#!/usr/bin/env python3
"""Let a FirstLight checkpoint play the MuMu account in live battles (your own accounts only).

The probe's live mode reads the battle the game itself runs; BattleEnvV1 and the V4 policy see it
exactly as in FirstLight's own offline matches (``offline_agent``); the plays become adb taps on
the MuMu screen.  Play against it from your phone: a friendly between your two accounts.

    python -m native_runner.live_diagnose start         # once: lift the offline block, game online
    python -m native_runner.live_play shot --account-id N   # in a battle: screenshot + tap layout drawn on it
    python -m native_runner.live_play run --account-id N    # play every battle the MuMu account enters

``--account-id`` is the MuMu account's id (``run`` prints both players' ids; it is remembered in
runs/live/seat.json).  The tap layout lives in runs/live/layout.json; ``shot`` draws it over a
screenshot (arena grid, towers where the layout puts them, hand slots, ability button) so it can
be checked and corrected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping

from .battle_env import BattleEnvError
from .cr_native_env import NativeClashEnv, RunnerError, _decode_observed_hand_card, _observed_command_card_id
from .live_diagnose import CONTROL_HOST, CONTROL_PORT, _adb
from .live_match import LiveCaptureNative
from .paths import WORKSPACE_ROOT
from .play_timing import DEFAULT_PLAY_GAP_TICKS, PlayGapClock, parse_play_gap_ticks

LIVE_ROOT = WORKSPACE_ROOT / "runs" / "live"
LAYOUT_PATH = LIVE_ROOT / "layout.json"
SEAT_PATH = LIVE_ROOT / "seat.json"

# Touch timing on the stock client (Jason-XII/clash-royale-battle-engine, docs/GAME_INTEGRATION.md,
# measured on MuMu): no touch is taken before tick 94; a play is issued 0-4 ticks after its tile
# touch and executes 21 ticks later, when the next card is dealt into its slot; that slot ignores
# touches until ~100 ms (2 ticks) after the deal.
FIRST_TOUCH_TICK = 94
LAND_TICKS = 26  # tile touch -> execution, with margin (issue +0..4, then +21)
DEAL_GUARD_TICKS = 3
# Keep card input on the short-touch path verified against native deployments.
# Extra UI sleeps must not become part of the policy's measured execution latency.
LOST_TAP_TICKS = 60  # a tapped card still in its slot this long after the tap was never taken
MAX_CONTRACT_REBUILDS = 4  # deck-contract corrections within one battle


@dataclass(frozen=True)
class ScreenLayout:
    """Screen pixels of the battle UI, for the player at the bottom of the screen."""

    width: int
    height: int
    arena: tuple[float, float, float, float]  # left, top, right, bottom of the 18x32 tile arena
    hand: tuple[tuple[float, float], ...]  # the four hand slots, left to right (handIndex 0..3)
    ability: tuple[float, float]  # hero / champion ability button
    calibrated: bool = False

    @classmethod
    def default(cls, width: int, height: int) -> "ScreenLayout":
        """A layout measured on this resolution, else Jason's Pixel 9 one (1080x2424) scaled: a guess."""

        if (width, height) in MEASURED_LAYOUTS:
            return MEASURED_LAYOUTS[width, height]
        sx, sy = width / 1080, height / 2424
        tile_h = (1714 - 295) / 30
        return cls(
            width=width,
            height=height,
            arena=(16 * sx, (295 - tile_h) * sy, 1053 * sx, (1714 + tile_h) * sy),
            hand=tuple(((335 + 200 * slot) * sx, 2220 * sy) for slot in range(4)),
            ability=(973 * sx, 1954 * sy),
        )

    @classmethod
    def load(cls, path: Path, width: int, height: int) -> "ScreenLayout":
        """``path`` when it is calibrated; otherwise the default (a measured one where there is one)."""

        if not path.is_file():
            return cls.default(width, height)
        data = json.loads(path.read_text("utf-8"))
        layout = cls(
            width=int(data["width"]),
            height=int(data["height"]),
            arena=tuple(map(float, data["arena"])),
            hand=tuple(tuple(map(float, slot)) for slot in data["hand"]),
            ability=tuple(map(float, data["ability"])),
            calibrated=bool(data.get("calibrated", False)),
        )
        if (layout.width, layout.height) != (width, height):
            raise SystemExit(f"{path} is for {layout.width}x{layout.height}, the screen is {width}x{height}")
        if not layout.calibrated and (width, height) in MEASURED_LAYOUTS:
            return MEASURED_LAYOUTS[width, height]  # an uncalibrated file is only the old guess
        return layout

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", "utf-8")

    def world_point(self, x: float, y: float, seat: int) -> tuple[int, int]:
        """Native world units -> screen pixels; ``seat`` is at the bottom (owner 0 sees x mirrored)."""

        cells_x, cells_y = (18 - x / 1000, y / 1000) if seat == 0 else (x / 1000, 32 - y / 1000)
        left, top, right, bottom = self.arena
        return round(left + (right - left) * cells_x / 18), round(bottom - (bottom - top) * cells_y / 32)


# Measured from MuMu battle screenshots (2026-10-12): the arena edges fitted to the grass
# checkerboard's tile boundaries (68.3 x 55.35 px tiles, 1-2 px residuals, river rows 15-16 at
# y 1026-1137), the hand card centres and the hero ability button.
MEASURED_LAYOUTS = {
    (1440, 2560): ScreenLayout(
        width=1440,
        height=2560,
        arena=(101.2, 195.7, 1330.1, 1966.9),
        hand=((446, 2270), (720, 2270), (990, 2270), (1260, 2270)),
        ability=(1295, 1940),
        calibrated=True,
    ),
}


def screen_size() -> tuple[int, int]:
    out = _adb("shell", "wm", "size").stdout
    sizes = [line.split(":")[1].strip() for line in out.splitlines() if ":" in line]
    width, height = (int(v) for v in sizes[-1].split("x"))  # an override size comes last
    return width, height


class AdbTaps:
    """One persistent adb shell; a play is `tap card; tap tile`, acknowledged."""

    def __init__(self) -> None:
        from .local_config import setting

        serial = setting("CR_ADB_SERIAL")
        command = [setting("CR_ADB", "adb") or "adb", *(["-s", serial] if serial else []), "shell"]
        self.shell = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def _run(self, line: str) -> None:
        assert self.shell.stdin is not None and self.shell.stdout is not None
        self.shell.stdin.write(f"{line}; echo done:$?\n")
        self.shell.stdin.flush()
        reply = self.shell.stdout.readline().strip()
        if reply != "done:0":
            raise RuntimeError(f"adb tap failed: {reply!r}")

    def tap(self, x: int, y: int) -> None:
        self._run(f"input tap {x} {y}")

    def play(self, card: tuple[int, int], tile: tuple[int, int]) -> None:
        # Keep the two taps in one shell command; this is the short path used by
        # the recordings whose card_play events were successfully attested.
        self._run(f"input tap {card[0]} {card[1]} && input tap {tile[0]} {tile[1]}")

    def close(self) -> None:
        self.shell.kill()


class LiveTapNative(LiveCaptureNative):
    """Live captures in, adb taps out: BattleEnvV1's native queue calls become screen touches.

    ``observe()`` (BattleEnvV1's polling while a decision interval passes) is a cheap
    ``live status``; every observation the policy sees is still one full game-thread capture.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        layout: ScreenLayout,
        taps: AdbTaps | None,
        log: Callable[[dict[str, Any]], None],
        timeout: float = 10.0,
        min_play_gap_ticks: tuple[int, int] = DEFAULT_PLAY_GAP_TICKS,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self.layout = layout
        self.taps = taps
        # Only real input can be reconciled with the game's deployment receipts.
        self.reconcile_card_play_receipts = taps is not None
        self.public_card_play_events_from_combat_ring = self.reconcile_card_play_receipts
        self.log = log
        self.seat: int | None = None
        self.generation: int | None = None
        self.ordinary: Mapping[str, Any] | None = None
        self.capture_micros: int | None = None  # game-thread time of the probe's last capture
        self.play_gap = PlayGapClock(min_play_gap_ticks, seed="live-taps")
        self.capture_sink = None
        self.last_capture_id = None
        self._slots: dict[int, tuple[int, int, int]] = {}  # handIndex -> (cardId, deckSlot, tick first seen)
        self._tapped: dict[int, tuple[int, int, int]] = {}  # handIndex -> (cardId, deckSlot, tap tick)
        self.tap_counts = {"sent": 0, "hand_changed": 0, "lost": 0, "skipped": 0}

    # -- reading ---------------------------------------------------------------------------
    def observe_atomic(self) -> dict[str, Any]:
        result = super().observe_atomic()
        self.ordinary = result["ordinary"]
        if self.generation is None:
            self.generation = int(result["generation"])
        if self.seat is not None:
            self._track_hand(int(result["tick"]))
        return result

    def observe(self) -> dict[str, Any]:
        status = self._request("live status")
        self.capture_micros = status.get("captureMicros")
        same = self.generation is None or int(status.get("generation", -1)) == self.generation
        return {"tick": int(status.get("tick", -1)), "ended": not (status.get("attached") and same)}

    def _hand(self) -> list[Mapping[str, Any]]:
        assert self.ordinary is not None and self.seat is not None
        player = next(p for p in self.ordinary["players"] if int(p["owner"]) == self.seat)
        return list(player.get("hand") or ())

    def _track_hand(self, tick: int) -> None:
        for card in self._hand():
            slot, identity = int(card["handIndex"]), (int(card["cardId"]), int(card["deckSlot"]))
            seen = self._slots.get(slot)
            if seen is None or seen[:2] != identity:
                self._slots[slot] = (*identity, tick)  # dealt (or first seen) at this tick
            tapped = self._tapped.get(slot)
            if tapped is not None and (tapped[:2] != identity or tick - tapped[2] > LOST_TAP_TICKS):
                if tapped[:2] == identity:
                    self.tap_counts["lost"] += 1
                    self.log({"event": "tap_lost", "tick": tick, "slot": slot, "card": identity[0]})
                    print(f"  警告：tick {tick} 槽位 {slot} 的点击后手牌未变化，可能未生效。", flush=True)
                else:
                    self.tap_counts["hand_changed"] += 1
                    self.log({"event": "tap_hand_changed", "tick": tick, "slot": slot,
                              "card": tapped[0], "tap_tick": tapped[2], "new_card": identity[0],
                              "execution_attested": False})
                    print(f"  tick {tick}: 点击后手牌已轮换（累计 {self.tap_counts['hand_changed']} 次）；"
                          "实际部署以游戏事件为准。", flush=True)
                del self._tapped[slot]

    # -- acting ----------------------------------------------------------------------------
    def _tick(self) -> int:
        return int(self.observe()["tick"])

    def _blocked(self, slot: int, tick: int) -> str | None:
        if tick < FIRST_TOUCH_TICK:
            return "before the client takes touches"
        if slot in self._tapped:
            return "its last play has not been dealt over yet"
        seen = self._slots.get(slot)
        if seen is not None and tick - seen[2] < DEAL_GUARD_TICKS:
            return "dealt too recently to be touched"
        return None

    def _wait_for_play_gap(self, owner: int, tick: int) -> int:
        target = self.play_gap.next_tick(owner, tick)
        deadline = time.monotonic() + max(5.0, (target - tick) * 0.05 + 2.0)
        if tick < target:
            self.log({"event": "play_gap_wait", "tick": tick, "until_tick": target, "owner": owner})
        while tick < target:
            status = self.observe()
            if status["ended"]:
                raise RunnerError("battle ended while waiting for the next tap")
            tick = int(status["tick"])
            if tick >= target:
                break
            if time.monotonic() >= deadline:
                raise RunnerError("live battle clock stopped while waiting for the next tap")
            time.sleep(0.01)
        return tick

    def _play(self, action: Any) -> dict[str, Any]:
        assert self.seat is not None and action.owner == self.seat
        tick = self._tick()
        card = next((c for c in self._hand() if int(c["handIndex"]) == action.hand_index), None)
        if card is None:  # BattleEnvV1 only needs the execute tick of a canceled play
            self.tap_counts["skipped"] += 1
            self.log({"event": "tap_skipped", "tick": tick, "slot": action.hand_index, "reason": "slot is empty"})
            return {"terminalCanceled": True, "executeTick": tick}
        descriptor = _decode_observed_hand_card(card)
        receipt = {
            "owner": action.owner,
            "handIndex": action.hand_index,
            "cardId": int(card["cardId"]),
            "commandCardId": _observed_command_card_id(card),
            "cardParameter": descriptor.packed,
            "deckSlot": descriptor.deck_slot,
            "cost": descriptor.cost,
            "formCode": descriptor.form_code,
            "formName": descriptor.form_name,
            "queuedAtTick": tick,
        }
        reason = self._blocked(action.hand_index, tick)
        if reason is not None:
            self.tap_counts["skipped"] += 1
            self.log({"event": "tap_skipped", "tick": tick, "slot": action.hand_index, "card": receipt["cardId"],
                      "reason": reason})
            return {**receipt, "terminalCanceled": True, "executeTick": tick}
        tick = self._wait_for_play_gap(action.owner, tick)
        receipt["queuedAtTick"] = tick
        card_point = tuple(round(v) for v in self.layout.hand[action.hand_index])
        tile_point = self.layout.world_point(action.x, action.y, self.seat)
        started = time.perf_counter()
        if self.taps is not None:
            self.taps.play(card_point, tile_point)
        after = self._tick()
        self.play_gap.commit(action.owner, after)
        self._tapped[action.hand_index] = (receipt["cardId"], receipt["deckSlot"], after)
        self.tap_counts["sent"] += 1
        self.log({"event": "tap", "tick": tick, "after_tick": after, "slot": action.hand_index,
                  "card": receipt["cardId"], "world": [action.x, action.y], "screen": [card_point, tile_point],
                  "input_ms": round((time.perf_counter() - started) * 1000), "dry_run": self.taps is None,
                  "execute_tick_estimate": after + LAND_TICKS, "execution_attested": False})
        return {**receipt, "executeTick": after + LAND_TICKS}

    def queue_hand_action_next_tick(self, action: Any) -> dict[str, Any]:
        return self._play(action)

    def queue_hand_action_at(self, action: Any, **_: Any) -> dict[str, Any]:
        return self._play(action)

    def queue_ability_action_at(self, action: Any, *, execute_in_ticks: int = 1) -> dict[str, Any]:
        tick = self._tick()
        tick = self._wait_for_play_gap(action.owner, tick)
        point = tuple(round(v) for v in self.layout.ability)
        if self.taps is not None and tick >= FIRST_TOUCH_TICK:
            self.taps.tap(*point)
        after = self._tick()
        self.play_gap.commit(action.owner, after)
        self.log({"event": "ability_tap", "tick": tick, "after_tick": after, "screen": point,
                  "dry_run": self.taps is None, "execute_tick_estimate": after + LAND_TICKS,
                  "execution_attested": False})
        return {
            "tick": tick,
            "owner": action.owner,
            "sourceEntityKey": [action.owner, action.object_index, action.secondary_index],
            "queuedAtTick": tick,
            "executeTick": after + LAND_TICKS,
        }


# -- seats -----------------------------------------------------------------------------------
def players_of(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"owner": int(p["owner"]), "accountId": int(p["accountId"]),
         "deck": [int(c["cardId"]) for c in sorted(p.get("deck") or (), key=lambda c: int(c["deckSlot"]))]}
        for p in capture["ordinary"]["players"]
    ]


def choose_seat(capture: Mapping[str, Any], owner: int | None, account_id: int | None) -> int:
    players = players_of(capture)
    if owner is not None:
        return owner
    if account_id is None and SEAT_PATH.is_file():
        account_id = int(json.loads(SEAT_PATH.read_text("utf-8"))["accountId"])
    for player in players:
        if account_id is not None and player["accountId"] == account_id:
            return player["owner"]
    listing = "\n".join(f"  owner {p['owner']}: accountId {p['accountId']}  deck {p['deck']}" for p in players)
    raise SystemExit(f"认不出 MuMu 账号是哪一方。本局双方：\n{listing}\n用 --account-id <MuMu 账号的 accountId> 再运行一次。")


def remember_seat(capture: Mapping[str, Any], seat: int) -> None:
    account = next(p["accountId"] for p in players_of(capture) if p["owner"] == seat)
    SEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEAT_PATH.write_text(json.dumps({"accountId": account}) + "\n", "utf-8")


# -- running ---------------------------------------------------------------------------------
def _rendered_action(action: Any) -> Any:
    """As offline_agent: a play is touched now; the client adds its own 21-tick delay."""

    from .contracts import ActionKind

    if action.kind != ActionKind.PLAY_CARD:
        return action
    metadata = dict(action.metadata)
    metadata.pop("native_command_age_ticks", None)
    metadata.update(execute_offset_ticks=1, model_delay_ignored=True)
    return replace(action, execute_offset_ticks=1, metadata=metadata)


def wait_for_battle(
    control: NativeClashEnv,
    skip_generation: int | None,
    *,
    periodic_action: Callable[[], None] | None = None,
    periodic_interval_s: float = 18.0,
    timeout_s: float = 90.0,
) -> int:
    """Wait for a validated running battle, with a bounded startup timeout."""

    last_periodic = time.monotonic()
    deadline = last_periodic + timeout_s
    last_error = None
    while True:
        if time.monotonic() >= deadline:
            raise RunnerError(f"{timeout_s:g} 秒内未取得可用对局观测；已停止等待且未点击界面。最近错误：{last_error}")
        first = control._request("live status")
        time.sleep(0.5)
        second = control._request("live status")
        running = second.get("attached") and int(second.get("tick", -1)) > int(first.get("tick", -1))
        generation = int(second.get("generation", -1))
        if running and int(second.get("tick", -1)) >= FIRST_TOUCH_TICK and generation == int(first.get("generation", -2)) and generation != skip_generation:
            # A controller can advance during loading/teardown with no readable
            # players. Only a validated battle capture permits model startup.
            try:
                capture = control._validate_atomic_result(control._request("live observe-atomic"))
                if int(capture["generation"]) == generation:
                    return generation
            except RunnerError as error:
                if str(error) != last_error:
                    print(f"等待完整对局观测（不点击界面）：{error}")
                last_error = str(error)
        if periodic_action is not None and (time.monotonic() - last_periodic) >= periodic_interval_s:
            periodic_action()
            last_periodic = time.monotonic()
        time.sleep(1.0)


def auto_return_to_lobby_and_queue(taps: AdbTaps | None, width: int, height: int, control: NativeClashEnv) -> None:
    if taps is None:
        return
    # 结算界面底部的确定/OK按钮：实测中心 (720, 2322) -> (0.5000, 0.9070)
    ok_x, ok_y = round(width * 0.5000), round(height * 0.9070)
    # 底栏正中央的【对战 Tab】：实测中心 (720, 2400) -> (0.5000, 0.9375)
    tab_x, tab_y = round(width * 0.5000), round(height * 0.9375)
    # 大厅正中巨大的黄色【对战】按钮：实测中心 (720, 1955) -> (0.5000, 0.7637)
    battle_x, battle_y = round(width * 0.5000), round(height * 0.7637)
    # 弹窗确认按钮（如“宝箱位已满继续”）：(0.6500, 0.6000)
    confirm_x, confirm_y = round(width * 0.6500), round(height * 0.6000)

    print("【全自动】对局已结束，正在跳过结算画面返回大厅……")
    time.sleep(3.5)

    # 1. 连续点击赛后确定/OK按钮，跳过结算画面
    for _ in range(4):
        taps.tap(ok_x, ok_y)
        time.sleep(0.8)

    time.sleep(1.0)
    # 2. 点击底栏中间的【对战Tab】，确保百分之百切回大厅主界面（避免误入竞技场/活动详情）
    for _ in range(2):
        taps.tap(tab_x, tab_y)
        time.sleep(0.8)

    time.sleep(1.5)
    # 3. 点击大厅黄色「对战」按钮，开始匹配
    print("【全自动】已到达大厅，点击「对战」按钮开始下一局匹配……")
    taps.tap(battle_x, battle_y)
    time.sleep(1.2)
    # 4. 若有“宝箱位已满是否继续”确认框，轻点确认按钮
    taps.tap(confirm_x, confirm_y)


def play_battle(
    model: Any,
    args: argparse.Namespace,
    layout: ScreenLayout,
    taps: AdbTaps | None,
    *,
    native_factory: Callable[..., LiveTapNative] | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    from .battle_env import BattleEnvError
    from .contracts import ActionKind, ActionV1
    from .live_match import LiveEpisode, is_contract_miss, learn_contract_miss, observe_forms
    from .training.v4.expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
    from .live_recording import LiveBattleRecording
    from .training.v4.deployment_timing import DeploymentTimingRecorder

    recorded = log_dir is not None
    log_dir = log_dir or LIVE_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{'replay-' if recorded else ''}{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    stream = path.open("w", encoding="utf-8", buffering=1)
    recording = None
    if getattr(args, "record", False):
        from .local_config import setting
        import hashlib
        sources = (Path(__file__), Path(__file__).with_name("live_match.py"),
                   Path(__file__).with_name("live_recording.py"))
        recording = LiveBattleRecording(getattr(args, "record_dir", LIVE_ROOT / "recordings"), identity={
            "checkpoint": args.checkpoint, "checkpoint_identity": getattr(args, "checkpoint_identity", None),
            "layout": asdict(layout), "resolution": [layout.width, layout.height],
            "dry_run": taps is None, "sample": not getattr(args, "deterministic", False),
            "gate_temperature": getattr(model, "ppo_gate_temperature", None),
            "card_tile_temperature": getattr(model, "inference_card_tile_temperature", None),
            "min_play_gap_ticks": getattr(args, "min_play_gap_ticks", DEFAULT_PLAY_GAP_TICKS),
            "input_method": "input_tap",
            "card_selection_gap_seconds": 0.0,
            "touch_hold_ms": 0,
            "adb_serial": setting("CR_ADB_SERIAL"),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        })

    def log(record: dict[str, Any]) -> None:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if recording is not None:
            recording.event(record)

    def screenshot(name):
        if recording is not None and getattr(args, "record_screenshots", False):
            try:
                png = subprocess.run([*_adb_prefix(), "exec-out", "screencap", "-p"],
                                     capture_output=True, check=True, timeout=15).stdout
                recording.screenshot(name, png)
            except (OSError, subprocess.SubprocessError) as error:
                log(dict(event="screenshot_failed", name=name, error=str(error)))

    factory = native_factory or (lambda **kw: LiveTapNative(CONTROL_HOST, CONTROL_PORT, **kw))
    native = factory(layout=layout, taps=taps, log=log)
    if recording is not None:
        native.capture_sink = recording.capture
    native.play_gap = PlayGapClock(getattr(args, "min_play_gap_ticks", DEFAULT_PLAY_GAP_TICKS), seed="live-taps")
    try:
        first = native.observe_atomic()
        seat = choose_seat(first, args.owner, args.account_id)
        if recording is not None:
            recording.update_identity(owner=seat, players=players_of(first), start_tick=first["tick"],
                                      generation=first["generation"])
            screenshot("start")
        if not recorded:
            remember_seat(first, seat)
    except BaseException as error:
        try:
            if recording is not None:
                recording.close({"result": "setup_failed", "error": str(error)})
        finally:
            stream.close()
            native.close_transport()
        raise
    native.seat = seat
    log({"event": "start", "tick": first["tick"], "seat": seat, "players": players_of(first),
         "checkpoint": args.checkpoint, "layout": asdict(layout), "min_play_gap_ticks": native.play_gap.bounds})
    print(f"对局开始（tick {first['tick']}），模型操作 owner {seat}；日志 {path}")
    summary: dict[str, Any] = {"log": str(path), "decisions": 0, "taps": 0, "rebuilds": 0}
    if recording is not None:
        summary["recording"] = str(recording.path)
        print(f"完整录制：{recording.path}")
    timing = DeploymentTimingRecorder(log) if recording is not None else None

    def play(episode: LiveEpisode) -> None:
        env, observation = episode.env, episode.observation
        while True:
            if recording is not None:
                recording.submit("policy_frame", observation.to_dict(), tick=observation.tick,
                                 capture_id=native.last_capture_id, recurrent_episode=summary["rebuilds"],
                                 next_decision=summary["decisions"] + 1)
            commands = {o: (ActionV1.wait(o, ticks=POLICY_DECISION_TICKS),) for o in (0, 1)}
            if observation.tick >= FIRST_POLICY_DECISION_TICK:
                decision = episode.session.decide(observation)
                chosen = tuple(replace(_rendered_action(a), action_id=(a.action_id or
                               f"live-{summary['rebuilds']}-{summary['decisions'] + 1}-{i}"))
                               for i, a in enumerate(decision.decoded.actions))
                commands[seat] = chosen or commands[seat]
                summary["decisions"] += 1
                acting = [a for a in chosen if a.kind != ActionKind.WAIT]
                summary["taps"] += len(acting)
                log({"event": "decision", "tick": observation.tick, "inference_ms": round(decision.inference_ms, 1),
                     "capture_us": native.capture_micros,
                     "decision_id": summary["decisions"], "capture_id": native.last_capture_id,
                     "actions": [a.to_dict() for a in chosen]})
                if timing is not None:
                    timing.submit(summary["decisions"], observation.tick,
                                  [dict(kind=a.kind.value, cardId=a.card_id) for a in chosen])
                for action in acting:
                    print(f"  tick {observation.tick}: {action.kind.value} card {action.card_id} -> {action.target_grid}")
            else:
                episode.session.tensorizer.tensorize(observation, validate=False)
            try:
                observations, _rewards, terms, truncs, _info = env.step(commands, record_trace=False)
            except BattleEnvError as error:
                if is_contract_miss(str(error)):
                    raise
                # Validated before anything is sent: skip this decision and let the interval pass.
                log({"event": "step_rejected", "tick": observation.tick, "error": str(error)})
                waits = {o: (ActionV1.wait(o, ticks=POLICY_DECISION_TICKS),) for o in (0, 1)}
                observations, _rewards, terms, truncs, _info = env.step(waits, record_trace=False)
            observation = observations[seat]
            if recording is not None:
                log({"event": "step_result", "tick": observation.tick, "capture_id": native.last_capture_id,
                     "decision_id": summary["decisions"], "recurrent_episode": summary["rebuilds"],
                     "rewards": _rewards, "terminated": terms, "truncated": truncs,
                     "terminal": observation.terminal.to_dict(),
                     "events": [e.to_dict() for e in observation.events]})
                timing.observe(env.deployment_timing_events(seat), observation.tick)
            if all(terms.values()) or all(truncs.values()):
                summary["result"] = "terminated"
                summary["terminal"] = observation.terminal.to_dict()
                if recording is not None:
                    recording.submit("policy_frame", observation.to_dict(), tick=observation.tick,
                                     capture_id=native.last_capture_id, recurrent_episode=summary["rebuilds"],
                                     terminal=True)
                return

    # The deck contract is read from the battle; a form it missed (a hero or an evolution the
    # capture did not show yet) is learned from BattleEnvV1's refusal and the episode rebuilt.
    forms_seen: dict[int, dict[int, int]] = {}
    episode: LiveEpisode | None = None
    try:
        while True:
            try:
                episode = LiveEpisode(
                    native, model, seat, device=args.device, forms_seen=forms_seen,
                    sample=not getattr(args, "deterministic", False),
                )
                observe_forms({"ordinary": native.ordinary}, forms_seen)
                log({"event": "contract", "tick": episode.observation.tick,
                     "forms": [list(episode.match.deck0_form_availability),
                               list(episode.match.deck1_form_availability)]})
                play(episode)
                break
            except BattleEnvError as error:
                learned = summary["rebuilds"] < MAX_CONTRACT_REBUILDS and learn_contract_miss(
                    str(error), {"ordinary": native.ordinary}, forms_seen
                )
                if not learned:
                    raise
                summary["rebuilds"] += 1
                log({"event": "contract_rebuild", "error": str(error), "forms_seen": forms_seen})
                print(f"  牌组形态补充后重建：{error}")
                if episode is not None:
                    episode.close()
                episode = None
    except (RunnerError, BattleEnvError) as error:
        summary["result"] = "error"
        summary["error"] = str(error)
    except BaseException as error:
        summary["result"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        summary["error"] = str(error)
        raise
    finally:
        try:
            if episode is not None:
                episode.close()
            if timing is not None:
                timing.finish()
            summary["tap_counts"] = dict(native.tap_counts)
            log({"event": "end", **summary})
            screenshot("end")
        finally:
            try:
                if recording is not None:
                    recording.close(summary)
            finally:
                stream.close()
                native.close_transport()
    print(f"对局结束：{summary}")
    return summary


class RecordedTapNative(LiveTapNative):
    """Saved live captures played back as a battle: each ``live status`` moves time to the next one.

    Captures are trimmed to the requested event cursors as the probe does (``full_rings`` sends
    every retained event instead), and nothing is tapped.
    """

    def __init__(self, captures: list[Mapping[str, Any]], *, full_rings: bool = False, **kwargs: Any) -> None:
        super().__init__("127.0.0.1", 0, **kwargs)
        self.captures = captures
        self.full_rings = full_rings
        self.index = 0
        self.moved = False

    def _wait_for_play_gap(self, owner: int, tick: int) -> int:
        # Recorded time advances by captures, never by wall-clock sleep.
        target = self.play_gap.next_tick(owner, tick)
        while tick < target:
            self.index += 1
            self.moved = False
            tick = self._tick()
        return tick

    def _request(self, command: str, **_: Any) -> dict[str, Any]:
        from .live_match import trim_event_rings

        if command == "live status":
            if self.index + 1 >= len(self.captures):
                raise RunnerError("recording ended")
            self.moved = True
            upcoming = self.captures[self.index + 1]
            return {"live": True, "attached": True, "generation": upcoming["generation"], "tick": upcoming["tick"]}
        if command.startswith("live observe-atomic"):
            if self.moved:
                self.index += 1
                self.moved = False
            floors = [0, 0, 0, 0] if self.full_rings else [int(v) for v in command.split()[2:]] or [0, 0, 0, 0]
            return trim_event_rings(self.captures[self.index], floors)
        raise RunnerError(f"recorded captures cannot answer {command!r}")


def replay(directory: Path, checkpoint: str, owner: int, *, full_rings: bool = False) -> list[dict[str, Any]]:
    """Play saved ``live_diagnose`` captures back through ``play_battle`` (dry run); the decisions."""

    import gzip

    from .training.v4.policy_session import load_policy_v4

    captures = [json.load(gzip.open(path, "rt", encoding="utf-8")) for path in sorted(directory.glob("sample-*.json.gz"))]
    if not captures:
        raise SystemExit(f"no sample-*.json.gz in {directory}")
    args = argparse.Namespace(owner=owner, account_id=None, checkpoint=checkpoint, device="cpu", deterministic=True)
    summary = play_battle(
        load_policy_v4(checkpoint, device="cpu").model,
        args,
        ScreenLayout.default(1080, 2424),
        None,
        native_factory=lambda **kw: RecordedTapNative(captures, full_rings=full_rings, **kw),
        log_dir=directory,
    )
    records = [json.loads(line) for line in Path(summary["log"]).read_text("utf-8").splitlines()]
    return [record for record in records if record["event"] == "decision"]


def run(args: argparse.Namespace) -> None:
    from .training.v4.policy_session import load_policy_v4

    width, height = screen_size()
    layout = ScreenLayout.load(Path(args.layout), width, height)
    if not layout.calibrated:
        print(f"注意：{args.layout} 还没校准（屏幕 {width}x{height}），点击位置可能不准。先用 shot 检查。")
    taps = None if args.dry_run else AdbTaps()
    loaded = load_policy_v4(args.checkpoint, device=args.device)
    model = loaded.model
    args.checkpoint_identity = dict(path=str(loaded.checkpoint_path), sha256=loaded.checkpoint_sha256,
                                    checkpoint_id=loaded.checkpoint_id, update=loaded.update_step)
    # A training checkpoint keeps the act/wait temperature of the update that saved it (0.5 early in a run);
    # sampling with it plays noticeably more randomly than the released models' 0.2.
    if getattr(args, "gate_temperature", None) is not None:
        model.set_ppo_gate_temperature(args.gate_temperature)
        print(f"门控温度已设定为: {model.ppo_gate_temperature}")
    if getattr(args, "card_tile_temperature", None) is not None:
        model.set_inference_card_tile_temperature(args.card_tile_temperature)
        print(f"选牌和落点温度已设定为: {model.inference_card_tile_temperature}（出不出牌仍用门控温度）")
    control = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=10.0)
    control._request("live on")
    played, last = 0, None
    auto_queue = getattr(args, "auto_queue", False)
    tab_x, tab_y = round(width * 0.5000), round(height * 0.9375)
    battle_x, battle_y = round(width * 0.5000), round(height * 0.7637)

    # Queue only once. A timeout or a failed observation is never evidence that
    # the game is in the lobby; periodic blind clicks can interrupt a real match.
    try:
        initial_status = control._request("live status")
        if auto_queue and taps is not None and not initial_status.get("attached"):
            print("【单次匹配】点击大厅对战按钮；等待期间不会重复点击。")
            taps.tap(tab_x, tab_y)
            time.sleep(0.8)
            taps.tap(battle_x, battle_y)
        elif auto_queue and initial_status.get("attached"):
            try:
                control._validate_atomic_result(control._request("live observe-atomic"))
            except RunnerError as error:
                raise RunnerError("接口报告 attached，但完整对局观测不可读，可能是大厅动画。"
                                  "为避免误点已停止；请确认大厅后单次点击对战，用不带 --auto-queue 的命令接管。") from error
            print("已验证完整对局观测，直接接管，不点击大厅或匹配按钮。")

        while args.games == 0 or played < args.games:
            print("等待对局……（等待期间不发送界面点击）")
            last = wait_for_battle(control, last)
            try:
                summary = play_battle(model, args, layout, taps)
            except (RunnerError, BattleEnvError) as error:
                print(f"对局初始化失败：{error}。已停止自动点击，不返回大厅、不重新匹配。")
                raise
            terminal = summary.get("terminal") or {}
            if summary.get("result") != "terminated" or not terminal.get("ended"):
                raise RunnerError(f"对局控制失败，停止自动点击并保留录制：{summary}")
            played += 1
            if auto_queue and taps is not None and (args.games == 0 or played < args.games):
                auto_return_to_lobby_and_queue(taps, width, height, control)
    finally:
        try:
            control._request("live off")
        except (RunnerError, OSError):
            pass
        if taps is not None:
            taps.close()
        control.close_transport()


def shot(args: argparse.Namespace) -> Path:
    """Screenshot with the tap layout (and, in a battle, towers and units where it puts them) drawn on it."""

    import gzip
    from io import BytesIO

    from PIL import Image, ImageDraw

    width, height = screen_size()
    layout = ScreenLayout.load(Path(args.layout), width, height)
    if not Path(args.layout).is_file() or (
        layout.calibrated and not json.loads(Path(args.layout).read_text("utf-8")).get("calibrated")
    ):
        layout.save(Path(args.layout))  # never overwrites a layout calibrated by hand
    raw = subprocess.run(
        [*_adb_prefix(), "exec-out", "screencap", "-p"], capture_output=True, check=True, timeout=30
    ).stdout
    image = Image.open(BytesIO(raw)).convert("RGB")
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    image.save(LIVE_ROOT / f"shot-{stamp}.png")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = layout.arena
    for column in range(19):
        x = left + (right - left) * column / 18
        draw.line([(x, top), (x, bottom)], fill=(0, 255, 255), width=1)
    for row in range(33):
        y = bottom - (bottom - top) * row / 32
        draw.line([(left, y), (right, y)], fill=(0, 255, 255), width=1)
    radius = max(6, width // 60)
    for slot, (x, y) in enumerate(layout.hand):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 0, 255), width=4)
        draw.text((x + radius, y - radius), str(slot), fill=(255, 0, 255))
    x, y = layout.ability
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 255, 0), width=4)
    control = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=10.0)
    was_on = False
    try:
        was_on = bool(control._request("live status").get("live"))
        control._request("live on")
        capture = control._request("live observe-atomic")
        seat = choose_seat(capture, args.owner, args.account_id)
        with gzip.open(LIVE_ROOT / f"shot-{stamp}.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(capture, stream)
        font = _font(max(16, width // 40))
        # Towers (orange) carry their hp: it must match the hp bar drawn next to the same tower,
        # or the layout mirrors the arena.  Units: blue ours, red theirs.
        for item in capture["ordinary"]["objects"]:
            tx, ty = layout.world_point(item["x"], item["y"], seat)
            if int(item.get("cardId", 0)) == -1:
                draw.ellipse([tx - radius, ty - radius, tx + radius, ty + radius], outline=(255, 64, 0), width=5)
                if item.get("hp") is not None:
                    draw.text((tx + radius + 4, ty - radius), str(item["hp"]), fill=(255, 64, 0), font=font,
                              stroke_width=3, stroke_fill=(0, 0, 0))
            else:
                color = (0, 120, 255) if int(item.get("owner", -1)) == seat else (255, 0, 0)
                draw.ellipse([tx - radius // 2, ty - radius // 2, tx + radius // 2, ty + radius // 2],
                             outline=color, width=4)
    except (RunnerError, OSError, SystemExit) as error:
        print(f"没有画塔的位置（{error}）")
    finally:
        try:
            if not was_on:  # leave a running `run` alone
                control._request("live off")
        except (RunnerError, OSError):
            pass
    out = LIVE_ROOT / f"shot-{stamp}-layout.png"
    image.save(out)
    print(f"截图：{LIVE_ROOT / f'shot-{stamp}.png'}\n叠加布局：{out}\n布局文件：{args.layout}")
    return out


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _adb_prefix() -> list[str]:
    from .local_config import setting

    serial = setting("CR_ADB_SERIAL")
    return [setting("CR_ADB", "adb") or "adb", *(["-s", serial] if serial else [])]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "shot"):
        command = sub.add_parser(name)
        command.add_argument("--owner", type=int, choices=(0, 1), help="the MuMu account's seat in this battle")
        command.add_argument("--account-id", type=int, help="the MuMu account's accountId (remembered)")
        command.add_argument("--layout", default=str(LAYOUT_PATH))
        if name == "run":
            command.add_argument("--checkpoint", default="checkpoints/General/checkpoint-step-00000460.pt")
            command.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
            command.add_argument("--games", type=int, default=0, help="battles to play (0: until stopped)")
            command.add_argument("--dry-run", action="store_true", help="decide and log, but do not tap")
            command.add_argument("--record", action="store_true", help="save all captures, policy observations and actions")
            command.add_argument("--record-dir", type=Path, default=LIVE_ROOT / "recordings")
            command.add_argument("--record-screenshots", action="store_true",
                                 help="save start/end debug screenshots with --record (adds ADB latency)")
            command.add_argument("--min-play-gap-ticks", type=parse_play_gap_ticks, default=DEFAULT_PLAY_GAP_TICKS,
                                 help="minimum gap between plays/abilities, in game ticks (default: 6-10 = 0.3-0.5 s)")
            command.add_argument("--deterministic", action="store_true",
                                 help="always take the most likely action (rarely plays; default samples)")
            command.add_argument("--gate-temperature", type=float, default=None,
                                 help="override model ppo_gate_temperature (e.g. 0.2)")
            command.add_argument("--card-tile-temperature", type=float, default=None,
                                 help="sample the card and tile at this temperature (e.g. 0.1); when to play "
                                      "keeps the gate temperature (default: the model's action temperature)")
            command.add_argument("--auto-queue", action="store_true",
                                 help="automatically return to lobby and queue for matches")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "shot":
        shot(args)
        return 0
    if args.record_screenshots and not args.record:
        parser.error("--record-screenshots requires --record")
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
