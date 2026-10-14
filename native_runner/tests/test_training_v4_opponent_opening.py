from __future__ import annotations

import pytest

from native_runner.training.v4.async_cluster_self_play import (
    _frozen_opponent_opening_force_owner_v4,
    opening_force_seconds_v4,
    parse_seconds_range,
)


def test_opening_force_time_is_drawn_per_game_within_the_range() -> None:
    assert opening_force_seconds_v4(None, 1) is None
    draws = [opening_force_seconds_v4((8.0, 20.0), sequence) for sequence in range(400)]
    assert all(8.0 <= value <= 20.0 for value in draws)
    assert min(draws) < 9.0 and max(draws) > 19.0
    # The same game always draws the same time.
    assert opening_force_seconds_v4((8.0, 20.0), 17) == opening_force_seconds_v4((8.0, 20.0), 17)
    assert opening_force_seconds_v4((12.0, 12.0), 5) == 12.0


def test_frozen_opponent_is_forced_only_after_its_drawn_time_and_only_before_it_acts() -> None:
    def force(tick: int, opponent_actions: int = 0) -> int | None:
        return _frozen_opponent_opening_force_owner_v4(
            current_owners=(0,), active_actions=(3, opponent_actions), tick=tick, after_seconds=12.5
        )

    assert force(249) is None
    assert force(250) == 1
    assert force(400, opponent_actions=1) is None
    assert (
        _frozen_opponent_opening_force_owner_v4(
            current_owners=(0, 1), active_actions=(0, 0), tick=400, after_seconds=12.5
        )
        is None
    )


def test_seconds_range_parses_a_range_or_a_single_value() -> None:
    assert parse_seconds_range("8-20") == (8.0, 20.0)
    assert parse_seconds_range("15") == (15.0, 15.0)
    for value in ("0", "20-8", "x", "1-500"):
        with pytest.raises(ValueError):
            parse_seconds_range(value)
