from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest

from native_runner import phone_touch
from native_runner.phone_touch import (
    DEFAULT_LAYOUT,
    CalibrationPoints,
    apply_touch_settings,
    layout_from_calibration,
    load_layout,
    save_layout,
    validate_layout,
)


def _project(layout, world_x, world_y, *, lift=0.0):
    _, _, _, _, _, _, wl, wt, wr, wb, *_ = layout
    return (
        round(wl + world_x / 18_000 * (wr - wl)),
        round(wt + world_y / 32_000 * (wb - wt) - lift),
    )


def _points_for(layout, screen, *, tower_lift=0.0):
    centers = layout[14:18]
    return CalibrationPoints(
        screen=screen,
        bottom_first_card=(centers[0], layout[11]),
        bottom_last_card=(centers[3], layout[11]),
        top_card=(centers[1], layout[10]),
        left_bridge=_project(layout, 3_500, 16_000),
        right_bridge=_project(layout, 14_500, 16_000),
        top_left_tower=_project(layout, 3_500, 6_500, lift=tower_lift),
        bottom_right_tower=_project(layout, 14_500, 25_500, lift=tower_lift),
    )


def test_calibration_reproduces_the_measured_mumu_layout() -> None:
    # Tower sprites stand above their ground point; only their difference is used.
    layout = layout_from_calibration(_points_for(DEFAULT_LAYOUT, (1080, 1920), tower_lift=40))

    for index in (0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        assert abs(layout[index] - DEFAULT_LAYOUT[index]) <= 2, index
    for index in (2, 3, 4, 5):
        assert abs(layout[index] - DEFAULT_LAYOUT[index]) <= 3, index


def test_calibration_scales_to_a_tall_phone() -> None:
    # A 1440x3200 phone whose HUD is the MuMu HUD scaled by 4/3 and moved down.
    scale = 1440 / 1080
    shifted = list(DEFAULT_LAYOUT)
    shifted[0:2] = [1440, 3200]
    for index in (2, 4, 6, 8, 12, 13, 14, 15, 16, 17):
        shifted[index] = round(DEFAULT_LAYOUT[index] * scale)
    for index in (3, 5, 7, 9, 10, 11):
        shifted[index] = round(DEFAULT_LAYOUT[index] * scale) + 160
    layout = layout_from_calibration(_points_for(tuple(shifted), (1440, 3200)))

    validate_layout(layout)
    for index, expected in enumerate(shifted):
        assert abs(layout[index] - expected) <= 4, index


def test_calibration_rejects_swapped_taps() -> None:
    points = _points_for(DEFAULT_LAYOUT, (1080, 1920))
    with pytest.raises(ValueError):
        layout_from_calibration(replace(points, left_bridge=points.right_bridge, right_bridge=points.left_bridge))
    with pytest.raises(ValueError):
        layout_from_calibration(replace(points, bottom_first_card=points.bottom_last_card,
                                        bottom_last_card=points.bottom_first_card))


def test_default_layout_matches_the_probe_constant() -> None:
    header = (Path(__file__).resolve().parents[1] / "probe" / "native_touch_layout.h").read_text(encoding="utf-8")
    body = re.search(r"kNativeRender1080x1920 = \{(.*?)\};", header, re.S).group(1)
    assert tuple(int(value) for value in re.findall(r"-?\d+", body)) == DEFAULT_LAYOUT


def test_layout_round_trip_and_validation(tmp_path) -> None:
    path = tmp_path / "layout.json"
    assert load_layout(path) is None
    save_layout(DEFAULT_LAYOUT, path)
    assert load_layout(path) == DEFAULT_LAYOUT
    broken = list(DEFAULT_LAYOUT)
    broken[15] = broken[14] + 5
    with pytest.raises(ValueError):
        validate_layout(broken)


def test_apply_touch_settings_locks_owner_and_loads_layout(tmp_path) -> None:
    class StubEnv:
        def __init__(self) -> None:
            self.calls = []

        def set_touch_layout(self, layout):
            self.calls.append(("layout", layout))
            return {"ok": True}

        def set_touch_calibrating(self, enabled):
            self.calls.append(("calibrate", enabled))
            return {"ok": True}

        def set_touch_ability_tap(self, enabled):
            self.calls.append(("ability", enabled))
            return {"ok": True}

        def set_touch_owner(self, owner):
            self.calls.append(("owner", owner))
            return {"ok": True, "ownerFilter": -1 if owner is None else owner}

    path = tmp_path / "layout.json"
    env = StubEnv()
    apply_touch_settings(env, owner=1, path=path)
    assert env.calls == [("calibrate", False), ("ability", True), ("owner", 1)]

    save_layout(DEFAULT_LAYOUT, path)
    env = StubEnv()
    apply_touch_settings(env, owner=None, path=path)
    assert env.calls[0] == ("layout", DEFAULT_LAYOUT)
    assert env.calls[-1] == ("owner", None)


def test_native_env_touch_commands_use_probe_wire_format() -> None:
    from native_runner.cr_native_env import NativeClashEnv

    class Stub(NativeClashEnv):
        def __init__(self) -> None:
            super().__init__()
            self.commands = []

        def _request(self, command: str):
            self.commands.append(command)
            return {"ok": True}

    env = Stub()
    env.set_touch_layout(DEFAULT_LAYOUT)
    env.set_touch_layout(None)
    env.set_touch_owner(None)
    env.set_touch_owner(0)
    env.set_touch_ability_tap(False)
    env.set_touch_calibrating(True)
    assert env.commands == [
        "touch layout " + " ".join(str(value) for value in DEFAULT_LAYOUT),
        "touch layout default",
        "touch owner -1",
        "touch owner 0",
        "touch ability off",
        "touch calibrate on",
    ]
    with pytest.raises(ValueError):
        env.set_touch_layout(DEFAULT_LAYOUT[:-1])
    with pytest.raises(ValueError):
        env.set_touch_owner(2)


def test_phone_mode_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(phone_touch, "setting", lambda name, default=None: default)
    assert not phone_touch.phone_mode_enabled()
    monkeypatch.setattr(phone_touch, "setting", lambda name, default=None: " Phone " if name == "CR_DEVICE_MODE" else default)
    assert phone_touch.phone_mode_enabled()
