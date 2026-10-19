"""Accumulating several segments into one PPO update must not change what each segment teaches."""

from dataclasses import fields, is_dataclass

import pytest
import torch

from native_runner.tests.test_training_v4_ppo_direction import _actions, _production_like_model, _states
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4, concatenate_rollout_lanes_v4
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT


def _rollout(model, lanes: int, steps: int, *, seed: int, policy: str = "behavior") -> StoredRolloutV4:
    generator = torch.Generator().manual_seed(seed)
    gates = [GATE_ACT if (lane + seed) % 2 else GATE_WAIT for lane in range(lanes)]
    actions = _actions(gates, [lane % 2 for lane in range(lanes)], [0] * lanes)
    shape = (steps, lanes)
    terminated = torch.zeros(shape, dtype=torch.bool)
    terminated[-1, ::2] = True
    hidden = model.initial_state(lanes)
    return StoredRolloutV4(
        observations=(_states(lanes),) * steps,
        actions=(actions,) * steps,
        episode_start=torch.cat((torch.ones(1, lanes, dtype=torch.bool), torch.zeros(steps - 1, lanes, dtype=torch.bool))),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        behavior_log_prob=-torch.rand(shape, generator=generator),
        behavior_value=torch.randn(shape, generator=generator),
        next_value=torch.randn(shape, generator=generator),
        rewards=torch.randn(shape, generator=generator),
        terminated=terminated,
        truncated=torch.zeros(shape, dtype=torch.bool),
        initial_state=type(hidden)(
            hidden=torch.randn(hidden.hidden.shape, generator=generator),
            cell=torch.randn(hidden.cell.shape, generator=generator),
        ),
        gate_temperature=model.ppo_gate_temperature,
        action_temperature=model.ppo_action_temperature,
        continue_temperature=model.ppo_continue_temperature,
        policy_state_id=policy,
    )


def _assert_same(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    elif is_dataclass(left):
        for item in fields(left):
            _assert_same(getattr(left, item.name), getattr(right, item.name))
    else:
        assert left == right


def test_lanes_keep_their_data_and_halves_mix_the_segments() -> None:
    model = _production_like_model()
    whole = _rollout(model, 8, 3, seed=1)
    first, second = whole.select_lanes(torch.arange(0, 3)), whole.select_lanes(torch.arange(3, 8))
    joined, split = concatenate_rollout_lanes_v4((first, second))
    # First halves of both segments (lane 0; lanes 3-4), then their second halves (1-2; 5-7).
    _assert_same(joined, whole.select_lanes(torch.tensor([0, 3, 4, 1, 2, 5, 6, 7])))
    assert split == 3


def test_a_shorter_segment_is_padded_with_steps_that_teach_nothing() -> None:
    model = _production_like_model()
    long, short = _rollout(model, 4, 3, seed=2), _rollout(model, 3, 1, seed=3)
    joined, _ = concatenate_rollout_lanes_v4((long, short))
    assert joined.time_steps == 3 and joined.batch_size == 7
    order = [0, 1, 4, 2, 3, 5, 6]  # joined lane -> lane of long (0-3) or short (4-6)
    short_lanes = [index for index, lane in enumerate(order) if lane >= 4]
    assert not joined.valid_mask[1:, short_lanes].any() and joined.valid_mask[0, short_lanes].all()
    joined_advantages, _ = joined.advantages_and_returns(gamma=0.99, gae_lambda=0.95)
    long_advantages, _ = long.advantages_and_returns(gamma=0.99, gae_lambda=0.95)
    short_advantages, _ = short.advantages_and_returns(gamma=0.99, gae_lambda=0.95)
    for index, lane in enumerate(order):
        if lane < 4:
            torch.testing.assert_close(joined_advantages[:, index], long_advantages[:, lane])
        else:
            torch.testing.assert_close(joined_advantages[0, index], short_advantages[0, lane - 4])


def test_rollouts_of_different_policies_are_refused() -> None:
    model = _production_like_model()
    with pytest.raises(ValueError, match="one behavior policy"):
        concatenate_rollout_lanes_v4((_rollout(model, 2, 1, seed=4), _rollout(model, 2, 1, seed=5, policy="other")))


def _update(rollout: StoredRolloutV4, **options) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    model = _production_like_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-5, weight_decay=0.0)
    metrics = distributed_ppo_update_v4(
        model,
        optimizer,
        rollout,
        epochs=2,
        lane_minibatch_size=rollout.batch_size,
        sequence_chunk_steps=2,
        config=PPOConfigV4(clip_range=0.1, value_clip_range=0.1, entropy_coefficient=0.0, max_grad_norm=0.5),
        autocast_dtype=None,
        gradient_noise=True,
        **options,
    )
    return metrics, model.state_dict()


def test_the_noise_split_chooses_the_halves_and_leaves_the_update_alone() -> None:
    model = _production_like_model()
    rollout = _rollout(model, 6, 2, seed=6)
    default_metrics, default_state = _update(rollout)
    middle_metrics, middle_state = _update(rollout, gradient_noise_split=3)
    other_metrics, other_state = _update(rollout, gradient_noise_split=2)
    assert middle_metrics[0]["gradient_half_cosine"] == default_metrics[0]["gradient_half_cosine"]
    assert other_metrics[0]["gradient_half_cosine"] != default_metrics[0]["gradient_half_cosine"]
    for name, value in default_state.items():
        torch.testing.assert_close(middle_state[name], value, rtol=0.0, atol=0.0)
        torch.testing.assert_close(other_state[name], value, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="outside the rollout lanes"):
        _update(rollout, gradient_noise_split=7)
