"""Recent own actions: tensorizer tracking, zero-contribution checkpoint upgrade, optimizer mapping."""

from dataclasses import replace

import pytest
import torch

from native_runner.contracts import EventV1
from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.tests.test_training_v4_tensorizer import _observation, _semantic_catalog, _tensorizer
from native_runner.training.v4.checkpoint import (
    RECENT_ACTION_PARAMETER_PREFIXES,
    RECENT_ACTION_WIDENED,
    checkpoint_contract,
    zero_start_parameter_names,
    legacy_checkpoint_contract,
    load_actor_critic_checkpoint,
    save_actor_critic_checkpoint,
)
from native_runner.training.v4.tensors import GATE_ACT, ActionSequenceV4, RecentActionsV4


def _deploy_action(batch, *, target_cell: int) -> tuple[ActionSequenceV4, int]:
    row = int(((batch.candidates.variant[0] == 0) & batch.candidates.mask[0]).nonzero(as_tuple=False)[0])
    action = ActionSequenceV4(
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([1]),
        candidate_index=torch.tensor([[row, -1]]),
        candidate_uid=torch.tensor([[int(batch.candidates.uid[0, row]), -1]]),
        target_cell=torch.tensor([[target_cell, -1]]),
        delay_offset_bin=torch.tensor([[0, -1]]),
    )
    return action, int(batch.candidates.native_visible_card_id[0, row])


def _at(tick: int, *events: EventV1):
    return replace(_observation(), tick=tick, events=tuple(events))


def test_recent_action_is_queued_then_executed_then_expires() -> None:
    tensorizer = _tensorizer(*_semantic_catalog())
    batch = tensorizer.tensorize(_at(20))
    assert bool(batch.recent_actions.known[0]) and not bool(batch.recent_actions.mask.any())
    action, card_id = _deploy_action(batch, target_cell=9 * 18 + 5)
    tensorizer.record_action(action, batch)

    queued = tensorizer.tensorize(_at(25)).recent_actions
    assert bool(queued.mask[0, 0]) and int(queued.mask.sum()) == 1
    assert int(queued.target_cell[0, 0]) == 9 * 18 + 5
    assert queued.features[0, 0].tolist() == pytest.approx([5 / 60, 0.0, 0.0, 0.0])

    executed_event = EventV1(tick=44, event_type="action_executed", owner=0, card_id=card_id)
    executed = tensorizer.tensorize(_at(45, executed_event)).recent_actions
    assert executed.features[0, 0].tolist() == pytest.approx([25 / 60, 1.0, 1 / 60, 0.0])
    # The same play reported again (command receipt and public play) is not a second execution.
    again = tensorizer.tensorize(_at(50, replace(executed_event, tick=46))).recent_actions
    assert again.features[0, 0, 1] == 1.0 and int(again.mask.sum()) == 1

    still = tensorizer.tensorize(_at(104)).recent_actions
    assert bool(still.mask[0, 0])
    assert not bool(tensorizer.tensorize(_at(105)).recent_actions.mask.any())


def test_rejected_and_other_owner_events_are_handled() -> None:
    tensorizer = _tensorizer(*_semantic_catalog())
    batch = tensorizer.tensorize(_at(20))
    action, card_id = _deploy_action(batch, target_cell=100)
    tensorizer.record_action(action, batch)
    enemy = EventV1(tick=30, event_type="action_executed", owner=1, card_id=card_id)
    assert tensorizer.tensorize(_at(30, enemy)).recent_actions.features[0, 0, 1] == 0.0
    rejected = EventV1(tick=35, event_type="action_rejected", owner=0, card_id=card_id)
    assert not bool(tensorizer.tensorize(_at(35, rejected)).recent_actions.mask.any())

    batch = tensorizer.tensorize(_at(40))
    action, _ = _deploy_action(batch, target_cell=100)
    tensorizer.record_action(action, batch)
    # A command never resolved is dropped three horizons after its decision.
    assert bool(tensorizer.tensorize(_at(220)).recent_actions.mask.any())
    assert not bool(tensorizer.tensorize(_at(221)).recent_actions.mask.any())


def test_episode_reset_clears_recent_actions() -> None:
    tensorizer = _tensorizer(*_semantic_catalog())
    batch = tensorizer.tensorize(_at(20))
    action, _ = _deploy_action(batch, target_cell=100)
    tensorizer.record_action(action, batch)
    tensorizer.end_episode()
    tensorizer.start_episode(_observation(skeleton_count=0))
    assert not bool(tensorizer.tensorize(_at(20)).recent_actions.mask.any())


def _with_recent(batch, config):
    recent = RecentActionsV4.empty(batch.batch_size, config, known=True)
    recent.mask[:, :2] = True
    recent.visible_card_vocab_id[:, :2] = 1
    recent.effective_card_vocab_id[:, :2] = 1
    recent.target_cell[:, 0] = 200
    recent.target_cell[:, 1] = 300
    recent.features[:, 0] = torch.tensor([0.3, 0.0, 0.0, 0.0])
    recent.features[:, 1] = torch.tensor([0.6, 1.0, 0.2, 0.0])
    return replace(batch, recent_actions=recent)


def _legacy_payload(model, path, *, optimizer=None) -> dict:
    """Write what a checkpoint from before recent actions contains."""

    save_actor_critic_checkpoint(path, model, optimizer=optimizer, training_stage="test", gamma_per_decision=0.99)
    payload = torch.load(path, weights_only=False)
    state = {}
    for name, value in payload["model_state_dict"].items():
        if ".card_encoder." in name and name.startswith(RECENT_ACTION_PARAMETER_PREFIXES):
            continue
        if name.startswith(RECENT_ACTION_PARAMETER_PREFIXES) or name in zero_start_parameter_names(model):
            continue
        if name in RECENT_ACTION_WIDENED:
            axis = RECENT_ACTION_WIDENED[name]
            extra = (
                model.config.recent_action_dim
                if name.startswith("input_projection")
                else model.config.recent_action_spatial_channels
            )
            value = value.narrow(axis, 0, value.shape[axis] - extra).clone()
        state[name] = value
    payload["model_state_dict"] = state
    payload["contract"] = legacy_checkpoint_contract(model)
    torch.save(payload, path)
    return payload


def test_legacy_checkpoint_upgrade_is_output_identical_and_trainable(tmp_path) -> None:
    torch.manual_seed(1)
    source = _model().eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if not name.startswith(RECENT_ACTION_PARAMETER_PREFIXES) and name not in zero_start_parameter_names(source):
                parameter.add_(0.01 * torch.randn_like(parameter))
        # A legacy checkpoint has no recent-action columns at all.
        source.input_projection[0].weight[:, -source.config.recent_action_dim :].zero_()
        source.spatial_encoder.input.weight[:, -source.config.recent_action_spatial_channels :].zero_()
    path = tmp_path / "legacy.pt"
    _legacy_payload(source, path)

    torch.manual_seed(2)
    upgraded = _model().eval()
    payload = load_actor_critic_checkpoint(path, upgraded, map_location="cpu")
    assert payload["contract"] == checkpoint_contract(upgraded)
    assert payload["upgraded_from_contract"] == legacy_checkpoint_contract(upgraded)

    plain = _batch()
    with_recent = _with_recent(plain, upgraded.config)
    unknown = replace(plain, recent_actions=RecentActionsV4.empty(plain.batch_size, upgraded.config, known=False))
    state = source.initial_state(plain.batch_size)
    with torch.no_grad():
        expected = source(plain, state)
        for batch in (plain, with_recent, unknown):
            actual = upgraded(batch, state)
            torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=1e-6)
            torch.testing.assert_close(actual.policy_context, expected.policy_context, rtol=0, atol=1e-6)
            torch.testing.assert_close(
                actual.encoded.spatial_memory, expected.encoded.spatial_memory, rtol=0, atol=1e-6
            )

    upgraded.train()
    context = upgraded(with_recent, state)
    (context.value.sum() + context.policy_context.square().sum()).backward()
    assert upgraded.recent_action_policy_skip.weight.grad.abs().sum() > 0
    assert upgraded.input_projection[0].weight.grad[:, -upgraded.config.recent_action_dim :].abs().sum() > 0
    assert upgraded.spatial_encoder.input.weight.grad[:, -2:].abs().sum() > 0


def test_legacy_optimizer_moments_follow_parameter_names(tmp_path) -> None:
    torch.manual_seed(3)
    source = _model()
    legacy_names = [
        name
        for name, _ in source.named_parameters()
        if not name.startswith(RECENT_ACTION_PARAMETER_PREFIXES) and name not in zero_start_parameter_names(source)
    ]
    parameters = dict(source.named_parameters())
    optimizer = torch.optim.AdamW([parameters[name] for name in legacy_names], lr=1e-3)
    context = source(_batch(), source.initial_state(2))
    context.value.sum().backward()
    optimizer.step()
    # Shrink the moments of widened tensors to their legacy width, as an old run saved them.
    legacy_optimizer = optimizer.state_dict()
    for index, name in enumerate(legacy_names):
        if name in RECENT_ACTION_WIDENED:
            extra = source.config.recent_action_dim if name.startswith("input_projection") else 2
            moments = legacy_optimizer["state"][index]
            for key in ("exp_avg", "exp_avg_sq"):
                moments[key] = moments[key].narrow(1, 0, moments[key].shape[1] - extra).clone()
    path = tmp_path / "legacy.pt"
    payload = _legacy_payload(source, path)
    payload["optimizer_state_dict"] = legacy_optimizer
    torch.save(payload, path)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    load_actor_critic_checkpoint(path, target, optimizer=target_optimizer, map_location="cpu")
    names = {id(parameter): name for name, parameter in target.named_parameters()}
    restored = {names[id(parameter)]: state for parameter, state in target_optimizer.state.items()}
    original_state = optimizer.state_dict()["state"]
    stepped = {legacy_names[index] for index in original_state}
    assert "input_projection.0.weight" in stepped
    assert set(restored) == stepped
    for index, name in enumerate(legacy_names):
        if index not in original_state:
            continue
        original = original_state[index]["exp_avg"]
        actual = restored[name]["exp_avg"]
        if name in RECENT_ACTION_WIDENED:
            actual = actual.narrow(1, 0, original.shape[1])
        torch.testing.assert_close(actual, original)
    widened = restored["input_projection.0.weight"]["exp_avg"]
    assert not bool(widened[:, -target.config.recent_action_dim :].any())
    # The upgraded optimizer can keep training every parameter, new ones included.
    target(_with_recent(_batch(), target.config), target.initial_state(2)).value.sum().backward()
    target_optimizer.step()


def test_frozen_recent_actions_keep_the_legacy_function_while_training(tmp_path) -> None:
    torch.manual_seed(4)
    model = _model()
    model.freeze_recent_actions()
    card_parameter = next(model.card_encoder.parameters())
    assert card_parameter.requires_grad
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = _with_recent(_batch(), model.config)
    for _ in range(3):
        optimizer.zero_grad()
        context = model(batch, model.initial_state(2))
        (context.value.sum() + context.policy_context.square().sum()).backward()
        optimizer.step()
    assert not bool(model.input_projection[0].weight[:, -model.config.recent_action_dim :].any())
    assert not bool(model.spatial_encoder.input.weight[:, -2:].any())
    assert not any(bool(parameter.any()) for parameter in model.recent_action_policy_skip.parameters())
    model.eval()
    plain = _batch()
    with torch.no_grad():
        expected = model(plain, model.initial_state(2))
        actual = model(_with_recent(plain, model.config), model.initial_state(2))
    torch.testing.assert_close(actual.policy_context, expected.policy_context, rtol=0, atol=0)
    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)

    trained = _model()
    with torch.no_grad():
        trained.recent_action_policy_skip.weight.fill_(0.1)
    with pytest.raises(ValueError, match="already trained"):
        trained.freeze_recent_actions()
