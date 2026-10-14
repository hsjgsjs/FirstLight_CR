from __future__ import annotations

import pytest

from native_runner.training.v4.async_cluster_self_play import early_first_play_penalty_v4
from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK


def penalty(tick: int, *, landed: bool = False) -> float:
    return early_first_play_penalty_v4(tick=tick, opponent_card_landed=landed, coefficient=0.1, free_seconds=14.0)


def test_opening_penalty_falls_linearly_to_zero_at_the_free_time() -> None:
    assert penalty(FIRST_POLICY_DECISION_TICK) == pytest.approx(0.1)
    # Halfway between the first decision (4.5 s) and 14 s.
    assert penalty((FIRST_POLICY_DECISION_TICK + 280) // 2) == pytest.approx(0.05, abs=1e-3)
    assert penalty(7 * 20) == pytest.approx(0.1 * (280 - 140) / (280 - FIRST_POLICY_DECISION_TICK))
    assert penalty(280) == 0.0
    assert penalty(400) == 0.0
    assert penalty(FIRST_POLICY_DECISION_TICK) > penalty(150) > penalty(250) > 0.0


def test_answering_a_landed_card_is_free() -> None:
    assert penalty(FIRST_POLICY_DECISION_TICK, landed=True) == 0.0
    assert penalty(150, landed=True) == 0.0
