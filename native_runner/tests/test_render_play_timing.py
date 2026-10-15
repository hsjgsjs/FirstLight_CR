"""Exercise command receipts and live game-clock spacing, without a running engine."""

import pytest

from native_runner.cr_native_env import AbilityAction, HandAction, NativeClashEnv
from native_runner.play_timing import DEFAULT_PLAY_GAP_TICKS, PlayGapClock
from native_runner.tests.test_live_play import _Native
from native_runner.training.v4.render_timing import SpacedRenderNativeV4


def test_gap_clock_preserves_first_play_and_independent_owners():
    clock = PlayGapClock((8, 8))
    assert DEFAULT_PLAY_GAP_TICKS == (8, 12)
    assert clock.next_tick(0, 101) == 101
    clock.commit(0, 101)
    assert clock.next_tick(0, 102) == 109
    assert clock.next_tick(1, 102) == 102
    assert clock.next_tick(0, 150) == 150
    variable = PlayGapClock(seed=7)
    variable.commit(0, 100)
    assert {variable.next_tick(0, 101) for _ in range(100)} == set(range(108, 113))


def test_renderer_spaces_compound_and_later_commands_using_receipts(monkeypatch):
    native = SpacedRenderNativeV4(min_play_gap_ticks=(8, 8))
    now = {"tick": 100, "paused": False}
    queued = []
    monkeypatch.setattr(native, "observe", lambda: {"tick": now["tick"]})
    monkeypatch.setattr(native, "pause", lambda: now.update(paused=True))
    monkeypatch.setattr(native, "resume", lambda: now.update(paused=False))

    def queue(_self, action, *, execute_in_ticks):
        assert now["paused"]
        tick = now["tick"] + execute_in_ticks
        # Native may clamp an opening play; its receipt is authoritative.
        if not queued:
            tick += 2
        queued.append((action.owner, tick))
        return {"executeTick": tick}

    monkeypatch.setattr(NativeClashEnv, "queue_hand_action_at", queue)
    monkeypatch.setattr(NativeClashEnv, "queue_ability_action_at", queue)
    first = native.queue_hand_action_next_tick(HandAction(0, 0, 3000, 6000))
    assert first["executeTick"] == 103 and not now["paused"]
    native.queue_hand_action_next_tick(HandAction(0, 1, 5000, 6000))
    native.queue_hand_action_next_tick(HandAction(1, 0, 3000, 26000))
    now["tick"] = 105
    native.queue_ability_action_at(AbilityAction(0, 1, 1))
    now["tick"] = 160  # inference/render time already exceeds the gap
    native.queue_hand_action_next_tick(HandAction(0, 2, 5000, 6000))
    assert queued == [(0, 103), (0, 111), (1, 101), (0, 119), (0, 161)]


def test_rejected_renderer_command_does_not_advance_clock_and_resumes(monkeypatch):
    native = SpacedRenderNativeV4(min_play_gap_ticks=(8, 8))
    resumed = []
    monkeypatch.setattr(native, "observe", lambda: {"tick": 100})
    monkeypatch.setattr(native, "pause", lambda: None)
    monkeypatch.setattr(native, "resume", lambda: resumed.append(True))
    monkeypatch.setattr(NativeClashEnv, "queue_hand_action_at",
                        lambda *_a, **_kw: {"executeTick": 101, "terminalCanceled": True})
    native.queue_hand_action_next_tick(HandAction(0, 0, 3000, 6000))
    assert native.play_gap.last == {} and len(resumed) == 1

    def fail(*_a, **_kw):
        raise ValueError("slot disappeared")

    monkeypatch.setattr(NativeClashEnv, "queue_hand_action_at", fail)
    with pytest.raises(ValueError, match="slot disappeared"):
        native.queue_hand_action_next_tick(HandAction(0, 0, 3000, 6000))
    assert native.play_gap.last == {} and len(resumed) == 2


def test_live_cards_and_abilities_wait_in_game_ticks(monkeypatch):
    native = _Native(200)
    native.set_hand(26000084)
    native.tick = 203  # hand deal guard elapsed
    sleeps = []

    def advance(_seconds):
        sleeps.append(True)
        native.tick += 1

    monkeypatch.setattr("native_runner.live_play.time.sleep", advance)
    first = native.queue_hand_action_next_tick(HandAction(1, 0, 3500, 25500))
    assert not sleeps
    native.tick += 2
    second = native.queue_ability_action_at(AbilityAction(1, 1, 1))
    assert second["executeTick"] - first["executeTick"] == 8
    native.set_hand(26000030)
    native.tick += 3
    third = native.queue_hand_action_next_tick(HandAction(1, 0, 3500, 25500))
    assert third["executeTick"] - second["executeTick"] == 8
    assert len(sleeps) == 11


def test_live_wait_stops_when_battle_ends(monkeypatch):
    from native_runner.cr_native_env import RunnerError

    native = _Native(200)
    native.play_gap.commit(1, 200)
    monkeypatch.setattr(native, "observe", lambda: {"tick": 201, "ended": True})
    with pytest.raises(RunnerError, match="battle ended"):
        native._wait_for_play_gap(1, 201)


def test_recording_advances_virtual_time_instead_of_sleeping(monkeypatch):
    from native_runner.live_play import RecordedTapNative
    from native_runner.tests.test_live_play import _layout

    captures = [{"generation": 9, "tick": tick} for tick in (100, 105, 110, 115, 130)]
    native = RecordedTapNative(captures, layout=_layout(), taps=None, log=lambda _r: None,
                               min_play_gap_ticks=(8, 8))
    native.generation = 9
    native.play_gap.commit(1, 105)
    assert native._tick() == 105

    def no_sleep(_seconds):
        pytest.fail("recording must advance virtual time")

    monkeypatch.setattr("native_runner.live_play.time.sleep", no_sleep)
    assert native._wait_for_play_gap(1, 105) == 115
    assert native._request("live observe-atomic")["tick"] == 115
