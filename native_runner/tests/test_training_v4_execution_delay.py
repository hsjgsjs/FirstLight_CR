from __future__ import annotations

import argparse

import pytest

from native_runner.contracts import ActionV1
from native_runner.training.v4.async_cluster_self_play import apply_execution_delay_v4
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
