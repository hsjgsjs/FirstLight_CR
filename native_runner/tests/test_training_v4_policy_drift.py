"""The drift report must read zero for identical policies and point at the head that actually moved."""

import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.policy_drift import HEADS, Tally, compare_sequence, report
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT, ActionSequenceV4


def _human_actions(gates):
    actions = []
    for gate in gates:
        act = gate == GATE_ACT
        actions.append(
            ActionSequenceV4(
                gate=torch.tensor([gate]),
                micro_action_count=torch.tensor([int(act)]),
                candidate_index=torch.tensor([[0 if act else -1, -1]]),
                candidate_uid=torch.tensor([[101 if act else -1, -1]]),
                target_cell=torch.tensor([[0 if act else -1, -1]]),
                delay_offset_bin=torch.tensor([[0 if act else -1, -1]]),
            )
        )
    return tuple(actions)


def test_identical_policy_reads_zero_and_a_moved_delay_head_reads_only_there() -> None:
    torch.manual_seed(3)
    base, same, moved = _model().eval(), _model().eval(), _model().eval()
    same.load_state_dict(base.state_dict())
    moved.load_state_dict(base.state_dict())
    with torch.no_grad():
        moved.delay_offset_head[-1].bias.add_(torch.linspace(-2.0, 2.0, moved.delay_offset_head[-1].bias.numel()))
    gates = [GATE_WAIT, GATE_ACT, GATE_WAIT, GATE_ACT, GATE_WAIT, GATE_ACT]
    sequence = ILSequenceV4(
        observations=(_batch(1),) * len(gates),
        actions=_human_actions(gates),
        episode_start=torch.tensor([[True]] + [[False]] * (len(gates) - 1)),
        valid_mask=torch.ones(len(gates), 1, dtype=torch.bool),
        returns=torch.zeros(len(gates), 1),
    )
    tallies = [Tally(), Tally()]
    with torch.no_grad():
        compare_sequence(
            base,
            [same, moved],
            sequence,
            tallies,
            temperatures=(0.33, 1.0, 2.0),
            width=base.config.board_width,
            generator=torch.Generator().manual_seed(1),
        )
    unchanged, delay_only = tallies
    assert unchanged.steps == delay_only.steps == len(gates)
    assert unchanged.gate_kl == 0.0
    assert all(value == 0.0 for value in unchanged.head_kl.values())
    assert unchanged.human["new"] == unchanged.human["base"]
    assert delay_only.gate_kl < 1e-9
    assert delay_only.head_kl["delay"] / delay_only.sampled_acts > 0.05
    # The card and placement for a play come before its delay, so they do not move.
    assert abs(delay_only.head_kl["candidate"]) < 1e-6 and abs(delay_only.head_kl["target"]) < 1e-6
    text = report("moved", delay_only)
    assert "per-play KL" in text and all(head in text for head in HEADS)
