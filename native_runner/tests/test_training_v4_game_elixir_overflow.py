from __future__ import annotations

import pytest

from native_runner.training.v4.async_cluster_self_play import (
    AsyncResidentEngineSpecV4,
    game_elixir_overflow_penalty_v4,
    late_opening_penalty_v4,
)


def step(state, newly_wasted, counting=True, cap=0.1):
    wasted, accrued, penalty = game_elixir_overflow_penalty_v4(
        wasted=state[0],
        accrued=state[1],
        newly_wasted=newly_wasted,
        counting=counting,
        coefficient=0.01,
        grace=3.0,
        step_cap=cap,
    )
    return (wasted, accrued), penalty


def test_opening_waste_before_first_card_is_free() -> None:
    state, penalty = step((0.0, 0.0), 8.0, counting=False)
    assert state == (0.0, 0.0) and penalty == 0.0


def test_whole_game_waste_is_charged_only_beyond_grace() -> None:
    state = (0.0, 0.0)
    charged = []
    for _ in range(5):  # separate overflow stretches still add up over the game
        state, penalty = step(state, 1.0)
        charged.append(penalty)
    assert charged[:3] == [0.0, 0.0, 0.0]
    assert charged[3:] == pytest.approx([0.01, 0.01])
    assert state == pytest.approx((5.0, 0.02))


def test_cost_above_the_step_cap_carries_over() -> None:
    state, first = step((0.0, 0.0), 23.0, cap=0.1)
    assert first == pytest.approx(0.1)
    state, second = step(state, 0.0, cap=0.1)
    assert second == pytest.approx(0.1)
    assert state[1] == pytest.approx(0.2)


def test_spec_defaults_to_disabled() -> None:
    spec = AsyncResidentEngineSpecV4(engine_index=0, cpu=0)
    assert spec.game_elixir_overflow is None and spec.late_opening is None


def late(state, newly_wasted, counting=True, step_cap=0.1):
    wasted, accrued, penalty = late_opening_penalty_v4(
        wasted=state[0],
        accrued=state[1],
        newly_wasted=newly_wasted,
        counting=counting,
        coefficient=0.005,
        grace=10.0,
        cap=0.1,
        step_cap=step_cap,
    )
    return (wasted, accrued), penalty


def test_late_opening_leaves_ordinary_openings_free() -> None:
    state, penalty = late((0.0, 0.0), 10.0)
    assert penalty == 0.0 and state == (10.0, 0.0)
    state, penalty = late(state, 4.0, counting=False)  # after the first card nothing more is counted
    assert penalty == 0.0 and state == (10.0, 0.0)


def test_late_opening_charges_a_standoff_up_to_the_cap() -> None:
    state, penalty = late((10.0, 0.0), 4.0)
    assert penalty == pytest.approx(0.02)
    total = penalty
    for _ in range(30):  # sitting at full until double elixir
        state, penalty = late(state, 1.0)
        total += penalty
    assert total == pytest.approx(0.1) and state[1] == pytest.approx(0.1)
