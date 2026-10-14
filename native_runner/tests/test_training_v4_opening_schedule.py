import random

import pytest

from native_runner.training.v4.opening_schedule import (
    frozen_opponent_opening_plan_v4,
    opening_force_seconds_v4,
    parse_seconds_range,
)


def test_natural_games_have_no_deadline_and_selection_is_stable():
    key = next(key for key in range(100) if frozen_opponent_opening_plan_v4(
        (6., 20.), key, natural_percent=20,
    )[0])
    for _ in range(10):
        assert frozen_opponent_opening_plan_v4((6., 20.), key, natural_percent=20) == (True, None)


def test_twenty_percent_mix_retains_both_early_and_passive_opponents():
    plans = [frozen_opponent_opening_plan_v4((6., 20.), key, natural_percent=20)
             for key in range(10000)]
    fraction = sum(exempt for exempt, _ in plans) / len(plans)
    assert .18 < fraction < .22
    deadlines = [deadline for exempt, deadline in plans if not exempt]
    assert all(6. <= deadline <= 20. for deadline in deadlines)
    assert min(deadlines) < 6.1 and max(deadlines) > 19.9


def test_disabled_exemption_preserves_previous_deadline_draws():
    for key in range(20):
        assert frozen_opponent_opening_plan_v4((6., 20.), key) == (
            False, opening_force_seconds_v4((6., 20.), key),
        )


def test_exemption_also_bypasses_legacy_il_timeout():
    assert frozen_opponent_opening_plan_v4((6., 20.), 17, natural_percent=100,
                                         il_after_seconds=15) == (True, None)
    assert frozen_opponent_opening_plan_v4((6., 20.), 17,
                                         il_after_seconds=15) == (False, 15.)


def test_no_timeout_does_not_claim_an_exemption():
    assert frozen_opponent_opening_plan_v4(None, 17, natural_percent=20) == (False, None)


@pytest.mark.parametrize('percent', [-1, 101, 20.5, True])
def test_invalid_natural_percentage_is_rejected(percent):
    with pytest.raises(ValueError):
        frozen_opponent_opening_plan_v4((6., 20.), 17, natural_percent=percent)


def test_schedule_does_not_change_global_random_generator():
    state = random.getstate()
    frozen_opponent_opening_plan_v4((6., 20.), 17, natural_percent=20)
    assert random.getstate() == state


def test_new_range_and_fixed_deadlines_are_supported():
    assert parse_seconds_range('6-20') == (6., 20.)
    assert frozen_opponent_opening_plan_v4((12., 12.), 17) == (False, 12.)
