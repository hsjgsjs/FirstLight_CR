"""Advantage scaling pooled over updates keeps game phases in proportion."""

import math

import pytest
import torch

from native_runner.training.v4.distributed_ppo import RunningAdvantageScaleV4, _global_normalize_advantages


def _normalize(values, scale=None):
    advantages = torch.tensor(values, dtype=torch.float32)
    mask = torch.ones_like(advantages, dtype=torch.bool)
    return _global_normalize_advantages(advantages, mask, device=torch.device("cpu"), running_scale=scale)


def test_batch_scaling_blows_small_opening_advantages_up_to_unit_spread() -> None:
    normalized, scale = _normalize([-0.06, 0.06] * 50)
    assert scale == pytest.approx(0.06) and float(normalized.std(unbiased=False)) == pytest.approx(1.0, rel=1e-4)


def test_running_scaling_keeps_openings_smaller_than_endings() -> None:
    scale = RunningAdvantageScaleV4(0.95)
    scaled = {}
    for _ in range(4):  # several rounds of eight segments, small at the opening, large at the end
        for spread in (0.06, 0.09, 0.12, 0.15, 0.2, 0.23, 0.28, 0.6):
            normalized, used = _normalize([-spread, spread] * 50, scale)
            scaled[spread] = float(normalized.std(unbiased=False))
    assert scaled[0.06] < 0.5 and scaled[0.6] > 2.0
    assert 0.06 < used < 0.6 and math.isfinite(used)


def test_running_scale_state_round_trips() -> None:
    scale = RunningAdvantageScaleV4(0.9)
    scale.update(count=10.0, total=1.0, square_total=2.0)
    restored = RunningAdvantageScaleV4(0.9)
    restored.load_state_dict(scale.state_dict())
    assert restored.state_dict() == scale.state_dict()
    with pytest.raises(ValueError):
        RunningAdvantageScaleV4(0.8).load_state_dict(scale.state_dict())
    with pytest.raises(ValueError):
        RunningAdvantageScaleV4(1.0)
