"""--overlap-collection: the collector plays with its own copy of the policy while the learner steps."""

import sys

import pytest
import torch

from native_runner.tests.test_training_v4_model import _model
from native_runner.training.v4 import train_ppo_self_play_cluster as trainer


def test_actor_sync_copies_weights_and_sampling_settings_in_place() -> None:
    torch.manual_seed(11)
    learner, actor = _model(), _model()
    with torch.no_grad():
        for parameter in learner.parameters():
            parameter.add_(torch.randn_like(parameter))
    learner.set_ppo_gate_temperature(0.27)
    learner.set_ppo_action_temperature(0.9)
    learner.set_ppo_continue_temperature(1.7)
    learner.set_ppo_rollout_sampling(False)
    learner.set_ppo_score_delay_offset(False)
    learner.set_ppo_location_head_fp32(True)
    learner.set_inference_delay_offset_max_ms(learner.config.delay_offset_ms[1])
    storage = [parameter.data_ptr() for parameter in actor.parameters()]

    trainer._sync_actor_model(actor, learner)

    for (name, copied), (_, source) in zip(actor.named_parameters(), learner.named_parameters()):
        assert torch.equal(copied, source), name
    for (name, copied), (_, source) in zip(actor.named_buffers(), learner.named_buffers()):
        assert torch.equal(copied, source), name
    # Same storage: CUDA graphs captured on the actor keep reading its current weights.
    assert [parameter.data_ptr() for parameter in actor.parameters()] == storage
    assert actor.ppo_gate_temperature == 0.27
    assert actor.ppo_action_temperature == 0.9
    assert actor.ppo_continue_temperature == 1.7
    assert actor.ppo_rollout_sampling is False
    assert actor._ppo_score_delay_offset is False
    assert actor._ppo_location_head_fp32 is True
    assert actor._inference_delay_offset_max_ms == learner.config.delay_offset_ms[1]

    # Later learner steps do not reach the actor until the next sync.
    with torch.no_grad():
        next(learner.parameters()).add_(1.0)
    assert not torch.equal(next(actor.parameters()), next(learner.parameters()))


def test_actor_sync_refuses_a_model_of_another_shape() -> None:
    with pytest.raises((RuntimeError, ValueError)):
        trainer._sync_actor_model(_model(), torch.nn.Linear(2, 2))  # type: ignore[arg-type]


def test_overlap_needs_accumulated_batches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    arguments = ["--deck-pool", "decks.json", "--checkpoint", "start.pt", "--output-dir", str(tmp_path / "out")]
    arguments += ["--engine-ready", "ready.json", "--overlap-collection"]
    monkeypatch.setattr(sys, "argv", ["train_ppo_self_play_cluster", *arguments])

    with pytest.raises(ValueError, match="--overlap-collection requires --accumulate-batches"):
        trainer.main()
