"""PPO must move the policy toward actions with positive advantage, head by head.

A wrong sign, a misaligned reward or a log-probability mismatch between rollout and update makes training
drift the wrong way while every health statistic (KL, clip fraction, explained variance) still looks normal.
These tests catch that in seconds instead of hours of training.
"""

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT, ActionSequenceV4


def _actions(gates, candidates, delays):
    count = len(gates)
    act = torch.tensor([gate == GATE_ACT for gate in gates])
    return ActionSequenceV4(
        gate=torch.tensor(gates, dtype=torch.long),
        micro_action_count=act.long(),
        candidate_index=torch.stack((torch.where(act, torch.tensor(candidates), -1), torch.full((count,), -1)), 1),
        candidate_uid=torch.stack((torch.where(act, torch.tensor(candidates) + 101, -1), torch.full((count,), -1)), 1),
        target_cell=torch.stack((torch.where(act, 0, -1), torch.full((count,), -1)), 1),
        delay_offset_bin=torch.stack((torch.where(act, torch.tensor(delays), -1), torch.full((count,), -1)), 1),
    )


def _evaluate(model, batch, actions):
    starts = torch.ones(1, batch.batch_size, dtype=torch.bool)
    return evaluate_recurrent_sequence(
        model,
        (batch,),
        (actions,),
        starts,
        initial_state=model.initial_state(batch.batch_size),
        gate_temperature=model.ppo_gate_temperature,
        action_temperature=model.ppo_action_temperature,
        continue_temperature=model.ppo_continue_temperature,
    )


def _train(model, batch, actions, rewards, *, updates=6):
    """On-policy PPO updates on one-step episodes that end with the given rewards."""
    lanes = batch.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-5, weight_decay=0.0)
    for _ in range(updates):
        with torch.no_grad():
            behavior = _evaluate(model, batch, actions)
        rollout = StoredRolloutV4(
            observations=(batch,),
            actions=(actions,),
            episode_start=torch.ones(1, lanes, dtype=torch.bool),
            valid_mask=torch.ones(1, lanes, dtype=torch.bool),
            behavior_log_prob=behavior.log_prob,
            behavior_value=behavior.value,
            next_value=torch.zeros(1, lanes),
            rewards=torch.tensor([rewards], dtype=torch.float32),
            terminated=torch.ones(1, lanes, dtype=torch.bool),
            truncated=torch.zeros(1, lanes, dtype=torch.bool),
            initial_state=model.initial_state(lanes),
            gate_temperature=model.ppo_gate_temperature,
            action_temperature=model.ppo_action_temperature,
            continue_temperature=model.ppo_continue_temperature,
            policy_state_id="direction",
        )
        distributed_ppo_update_v4(
            model,
            optimizer,
            rollout,
            epochs=4,
            lane_minibatch_size=lanes,
            sequence_chunk_steps=1,
            config=PPOConfigV4(clip_range=0.1, value_clip_range=0.1, entropy_coefficient=0.0, max_grad_norm=0.5),
            autocast_dtype=None,
        )


def _states(count):
    """Copies of the fixture's playable row: full elixir, cards 0 and 1 placeable anywhere."""
    return _batch(2).index_select(torch.zeros(count, dtype=torch.long))


def _production_like_model():
    torch.manual_seed(17)
    model = _model().eval()
    # Rollouts sample the gate at a low temperature; the update must score it at the same one.
    model.set_ppo_gate_temperature(0.33)
    return model


@pytest.mark.parametrize("act_reward", [1.0, -1.0])
def test_gate_moves_toward_the_rewarded_choice(act_reward: float) -> None:
    model = _production_like_model()
    batch = _states(8)
    gates = [GATE_ACT] * 4 + [GATE_WAIT] * 4
    actions = _actions(gates, [0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 2, 3, 0, 0, 0, 0])
    rewards = [act_reward] * 4 + [-act_reward] * 4
    with torch.no_grad():
        before = _evaluate(model, batch, actions).log_prob[0]
    _train(model, batch, actions, rewards)
    with torch.no_grad():
        after = _evaluate(model, batch, actions).log_prob[0]
    change = after - before
    # Rows with positive advantage gain probability, rows with negative advantage lose it.
    assert torch.all(change[:4] * act_reward > 0), change
    assert torch.all(change[4:] * -act_reward > 0), change


def test_card_choice_moves_toward_the_rewarded_card() -> None:
    model = _production_like_model()
    batch = _states(8)
    # Same states, both cards played; only card 0 is rewarded.
    candidates = [0, 1] * 4
    actions = _actions([GATE_ACT] * 8, candidates, [0] * 8)
    rewards = [1.0 if candidate == 0 else -1.0 for candidate in candidates]
    card_zero = _actions([GATE_ACT] * 8, [0] * 8, [0] * 8)
    card_one = _actions([GATE_ACT] * 8, [1] * 8, [0] * 8)
    with torch.no_grad():
        gap_before = _evaluate(model, batch, card_zero).log_prob - _evaluate(model, batch, card_one).log_prob
    _train(model, batch, actions, rewards)
    with torch.no_grad():
        gap_after = _evaluate(model, batch, card_zero).log_prob - _evaluate(model, batch, card_one).log_prob
    assert torch.all(gap_after > gap_before), (gap_before, gap_after)


def test_unscored_delay_offset_stays_out_of_the_ratio_and_gets_no_gradient() -> None:
    model = _production_like_model()
    batch = _states(8)
    delays = [0, 1, 2, 3] * 2
    actions = _actions([GATE_ACT] * 8, [0] * 8, delays)
    with torch.no_grad():
        scored = _evaluate(model, batch, actions).log_prob
        model.set_ppo_score_delay_offset(False)
        unscored = _evaluate(model, batch, actions).log_prob
        model.set_ppo_score_delay_offset(True)
        other_delay = _evaluate(model, batch, _actions([GATE_ACT] * 8, [0] * 8, [3, 2, 1, 0] * 2)).log_prob
        model.set_ppo_score_delay_offset(False)
        unscored_other_delay = _evaluate(model, batch, _actions([GATE_ACT] * 8, [0] * 8, [3, 2, 1, 0] * 2)).log_prob
    # The delay term is what separates the two scored values; without it the delay choice no longer matters.
    assert not torch.allclose(scored, unscored)
    assert not torch.allclose(scored, other_delay)
    torch.testing.assert_close(unscored, unscored_other_delay)
    head = [parameter.detach().clone() for parameter in model.delay_offset_head.parameters()]
    # Rewards follow the delay alone, which the environment would have replaced: nothing to learn there.
    _train(model, batch, actions, [1.0 if delay < 2 else -1.0 for delay in delays], updates=2)
    for before, parameter in zip(head, model.delay_offset_head.parameters(), strict=True):
        torch.testing.assert_close(parameter.detach(), before, rtol=0.0, atol=0.0)
