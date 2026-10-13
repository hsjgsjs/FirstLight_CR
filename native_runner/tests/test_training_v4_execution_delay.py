from __future__ import annotations

import argparse

import pytest

from native_runner.contracts import ActionV1
from native_runner.training.v4.async_cluster_self_play import (
    apply_execution_delay_v4,
    parse_play_gap_ticks,
    space_plays_v4,
)
from native_runner.training.v4.train_ppo_self_play_cluster import parse_execution_delay_ticks


def test_execution_delay_replaces_play_offsets_and_leaves_waits() -> None:
    actions = (ActionV1.play(0, 1, (3, 20), execute_offset_ticks=1), ActionV1.wait(0, ticks=5))
    assert apply_execution_delay_v4(actions, None, "key") is actions
    delayed = apply_execution_delay_v4(actions, (18, 24), "7-0-300")
    assert 18 <= delayed[0].execute_offset_ticks <= 24
    assert delayed[1] == actions[1]
    # The same decision replays the same delay; draws cover the whole inclusive range.
    assert apply_execution_delay_v4(actions, (18, 24), "7-0-300") == delayed
    draws = {apply_execution_delay_v4(actions, (18, 24), f"7-0-{tick}")[0].execute_offset_ticks for tick in range(200)}
    assert draws == set(range(18, 25))
    waits = (ActionV1.wait(1, ticks=5),)
    assert apply_execution_delay_v4(waits, (21, 21), "key") is waits


def test_execution_delay_option_parses_a_range_or_a_single_value() -> None:
    assert parse_execution_delay_ticks("18-24") == (18, 24)
    assert parse_execution_delay_ticks("21") == (21, 21)
    for value in ("0", "24-18", "x", "1-70000"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_execution_delay_ticks(value)


def test_play_gap_spaces_two_cards_in_one_decision_and_across_decisions() -> None:
    two = (ActionV1.play(0, 1, (3, 20)), ActionV1.play(0, 2, (5, 20)), ActionV1.wait(0, ticks=5))
    delayed = apply_execution_delay_v4(two, (21, 21), "k")
    assert space_plays_v4(delayed, None, "k", decision_tick=100, last_landing_tick=None) == (delayed, None)
    spaced, last = space_plays_v4(delayed, (4, 8), "k", decision_tick=100, last_landing_tick=None)
    assert spaced[0].execute_offset_ticks == 21
    assert 25 <= spaced[1].execute_offset_ticks <= 29
    assert spaced[2] == delayed[2]
    assert last == 100 + spaced[1].execute_offset_ticks
    # The next decision 5 ticks later draws a shorter delay; it still lands a gap after the previous card.
    one = apply_execution_delay_v4((ActionV1.play(0, 3, (3, 20)),), (18, 18), "k2")
    later, last2 = space_plays_v4(one, (4, 8), "k2", decision_tick=105, last_landing_tick=last)
    assert last + 4 <= 105 + later[0].execute_offset_ticks == last2 <= last + 8
    # A play already far enough after the last one keeps its offset.
    far, _ = space_plays_v4(one, (4, 8), "k3", decision_tick=500, last_landing_tick=last)
    assert far == one
    waits = (ActionV1.wait(0, ticks=5),)
    assert space_plays_v4(waits, (4, 8), "k", decision_tick=0, last_landing_tick=7) == (waits, 7)


def test_play_gap_option_parses_a_range_or_a_single_value() -> None:
    assert parse_play_gap_ticks("4-8") == (4, 8)
    assert parse_play_gap_ticks("6") == (6, 6)
    for value in ("0", "8-4", "x", "1-500"):
        with pytest.raises(ValueError):
            parse_play_gap_ticks(value)
