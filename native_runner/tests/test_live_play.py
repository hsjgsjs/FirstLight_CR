from __future__ import annotations

from pathlib import Path

import pytest

from native_runner.live_match import trim_event_rings
from native_runner.live_play import FIRST_TOUCH_TICK, LAND_TICKS, LiveTapNative, ScreenLayout
from native_runner.tests.cpp_source import cpp_function

PROBE = Path(__file__).resolve().parents[1] / "probe"


def _layout() -> ScreenLayout:
    return ScreenLayout(1000, 2000, arena=(0, 0, 1800, 3200), hand=((1, 1), (2, 2), (3, 3), (4, 4)), ability=(9, 9))


def test_world_points_put_the_seat_at_the_bottom() -> None:
    layout = _layout()
    # Owner 0's back-left corner (world 0,0) is bottom-right on its screen (x mirrored).
    assert layout.world_point(0, 0, 0) == (1800, 3200)
    assert layout.world_point(18000, 32000, 0) == (0, 0)
    # Owner 1 sees the arena turned around: its back line (y 32000) at the bottom.
    assert layout.world_point(0, 32000, 1) == (0, 3200)
    assert layout.world_point(9000, 16000, 1) == layout.world_point(9000, 16000, 0) == (900, 1600)


def test_layout_round_trip_and_screen_check(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    default = ScreenLayout.default(1080, 1920)
    assert ScreenLayout.load(path, 1080, 1920) == default and not default.calibrated
    default.save(path)
    assert ScreenLayout.load(path, 1080, 1920) == default
    with pytest.raises(SystemExit):
        ScreenLayout.load(path, 1440, 2560)


def test_the_measured_mumu_layout_replaces_an_uncalibrated_guess(tmp_path: Path) -> None:
    from dataclasses import replace

    from native_runner.live_play import MEASURED_LAYOUTS

    measured = MEASURED_LAYOUTS[1440, 2560]
    assert measured.calibrated and ScreenLayout.default(1440, 2560) == measured
    path = tmp_path / "layout.json"
    guess = replace(measured, arena=(21.3, 261.6, 1404.0, 1860.1), calibrated=False)
    guess.save(path)
    assert ScreenLayout.load(path, 1440, 2560) == measured
    own = replace(measured, ability=(1300, 1950))  # a layout calibrated by hand wins
    own.save(path)
    assert ScreenLayout.load(path, 1440, 2560) == own
    # Owner 1 (MuMu) sees its princess towers (world y 25500) above its king tower (29000).
    assert measured.world_point(3500, 25500, 1)[1] < measured.world_point(9000, 29000, 1)[1]


def test_event_rings_are_trimmed_like_the_probe() -> None:
    ring = {"oldestRetainedSequence": 3, "nextSequence": 8, "events": [{"sequence": n} for n in range(3, 8)]}
    capture = {"rich": {"phaseRuntime": dict(ring), "combatEvents": dict(ring)}}
    trimmed = trim_event_rings(capture, (0, 6, 0, 0))
    assert trimmed["rich"]["combatEvents"]["events"] == ring["events"]  # floor below oldest: everything
    assert trimmed["rich"]["phaseRuntime"]["transmitFromSequence"] == 6
    assert [e["sequence"] for e in trimmed["rich"]["phaseRuntime"]["events"]] == [6, 7]
    beyond = trim_event_rings(capture, (99, 0, 0, 0))["rich"]["combatEvents"]
    assert beyond["events"] == [] and beyond["transmitFromSequence"] == 8


class _Native(LiveTapNative):
    def __init__(self, tick: int) -> None:
        self.records: list[dict] = []
        super().__init__("127.0.0.1", 1, layout=_layout(), taps=None, log=self.records.append,
                         min_play_gap_ticks=(8, 8))
        self.tick = tick
        self.seat = 1
        self.generation = 7

    def _request(self, command: str, **_: object) -> dict:
        assert command == "live status"
        return {"live": True, "attached": True, "generation": 7, "tick": self.tick, "captureMicros": 1234}

    def set_hand(self, card_id: int) -> None:
        card = {"handIndex": 0, "deckSlot": 4, "cardId": card_id, "commandCardId": card_id,
                "cardParameter": 289417216, "cost": 1}
        self.ordinary = {"players": [{"owner": 1, "hand": [card]}]}
        self._track_hand(self.tick)


def test_taps_follow_the_clients_touch_rules() -> None:
    from native_runner.cr_native_env import HandAction

    native = _Native(FIRST_TOUCH_TICK - 1)
    native.set_hand(26000084)
    play = HandAction(owner=1, hand_index=0, x=3500, y=25500)
    early = native.queue_hand_action_next_tick(play)
    assert early["terminalCanceled"] and native.records[-1]["event"] == "tap_skipped"

    native.tick = 200
    receipt = native.queue_hand_action_next_tick(play)
    assert not receipt.get("terminalCanceled") and receipt["executeTick"] == 200 + LAND_TICKS
    assert receipt["cardId"] == receipt["commandCardId"] == 26000084 and receipt["cost"] == 1
    assert native.records[-1]["screen"][1] == _layout().world_point(3500, 25500, 1)

    native.tick = 205  # not dealt yet: the slot is empty on the screen
    assert native.queue_hand_action_next_tick(play)["terminalCanceled"]
    native.set_hand(26000030)  # dealt at 205
    native.tick = 206  # touches right after a deal are refused
    assert native.queue_hand_action_next_tick(play)["terminalCanceled"]
    native.tick = 205 + 3
    assert not native.queue_hand_action_next_tick(play).get("terminalCanceled")
    assert native.observe() == {"tick": 208, "ended": False} and native.capture_micros == 1234


def test_probe_sends_only_events_after_the_live_cursors() -> None:
    source = (PROBE / "cr_replay_probe.cpp").read_text(encoding="utf-8")
    command = cpp_function(source, "handle_live_command")
    assert '"%llu %llu %llu %llu"' in command
    # The cursors apply to one capture only and are cleared under the same lock.
    assert command.index("g_event_transmit_floor[ring] = floors[ring]") < command.index("g_live_capture_requested = true")
    assert "std::memset(g_event_transmit_floor, 0" in command
    assert "captureMicros" in command
    for name, ring in (
        ("combat_event_telemetry.inc", "kTransmitCombat"),
        ("phase_runtime_telemetry.inc", "kTransmitPhase"),
        ("visibility_runtime_telemetry.inc", "kTransmitVisibility"),
        ("remaining_runtime_telemetry.inc", "kTransmitRemaining"),
    ):
        assert f"transmit_from = event_transmit_from({ring}, retained_oldest" in (PROBE / name).read_text("utf-8")


def test_recorded_captures_play_back_as_a_battle() -> None:
    from native_runner.cr_native_env import RunnerError
    from native_runner.live_play import RecordedTapNative

    ring = {"oldestRetainedSequence": 1, "nextSequence": 3, "events": [{"sequence": 1}, {"sequence": 2}]}
    captures = [{"generation": 9, "tick": tick, "rich": {"combatEvents": dict(ring)}} for tick in (100, 160)]
    native = RecordedTapNative(captures, layout=_layout(), taps=None, log=lambda _r: None)
    assert native._request("live observe-atomic 2 0 0 0")["rich"]["combatEvents"]["events"] == [{"sequence": 2}]
    assert native._request("live status")["tick"] == 160  # time moves on to the next capture
    assert native._request("live observe-atomic")["tick"] == 160
    with pytest.raises(RunnerError):
        native._request("live status")
    assert native.pause() == {"ok": True}


def _deck_capture(forms: dict[int, int]) -> dict:
    # Real packed words: form in bits 0..3, deck slot + 1 in bits 22..27, cost in 28..31.
    deck = [{"deckSlot": slot, "cardId": 26000000 + slot, "cardParameter": (3 << 28) | ((slot + 1) << 22) | forms.get(slot, 0)}
            for slot in range(8)]
    evolvable = [{"deckSlot": slot, "evolvable": True} for slot in range(4)]
    return {"ordinary": {"players": [{"owner": 1, "deck": deck, "hand": []}]},
            "rich": {"players": [{"owner": 1, "evolutionRuntime": evolvable}]}}


def test_forms_the_battle_shows_outrank_the_slot_guess() -> None:
    from native_runner.live_match import form_availability_of, observe_forms

    guessed = _deck_capture({})
    assert form_availability_of(guessed, 1) == (1, 1, 0, 0, 0, 0, 0, 0)
    hero = _deck_capture({1: 2})  # a hero in the wild slot is no evolution
    seen: dict = {}
    observe_forms(hero, seen)
    assert seen == {1: {1: 2}}
    assert form_availability_of(hero, 1, seen) == (1, 2, 0, 0, 0, 0, 0, 0)
    # Two seen evolutions push the guessed ones out (<= 2 evolutions, <= 3 forms in all).
    assert form_availability_of(guessed, 1, {1: {2: 1, 3: 1}}) == (0, 0, 1, 1, 0, 0, 0, 0)
    with pytest.raises(Exception):
        form_availability_of(guessed, 1, {1: {2: 2, 3: 2, 4: 2}})


def test_contract_refusals_name_the_missing_form() -> None:
    from native_runner.live_match import is_contract_miss, learn_contract_miss

    capture = _deck_capture({})
    seen: dict = {}
    play = "native policy form is not enabled by the episode deck contract: owner=1, card=26000005, form=1, enabled=0"
    ability = "native explicit-Hero ability is not enabled by the episode deck contract: owner=1, card=26000006, enabled=0"
    assert is_contract_miss(play) and is_contract_miss(ability) and not is_contract_miss("compound decision overspends elixir")
    assert learn_contract_miss(play, capture, seen) and learn_contract_miss(ability, capture, seen)
    assert seen == {1: {5: 1, 6: 2}}
    assert not learn_contract_miss(play, capture, seen)  # already known: no endless rebuilds


def test_live_play_samples_its_actions_unless_told_otherwise() -> None:
    import inspect

    from native_runner.live_match import LiveEpisode

    # A greedy argmax of a decision every 0.25 s is nearly always "wait": sample like offline_agent.
    assert inspect.signature(LiveEpisode).parameters["sample"].default is True
    source = inspect.getsource(__import__("native_runner.live_play", fromlist=["play_battle"]).play_battle)
    assert 'sample=not getattr(args, "deterministic", False)' in source
