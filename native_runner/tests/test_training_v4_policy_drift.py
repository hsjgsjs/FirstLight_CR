"""The drift report must read zero for identical policies and point at the head that actually moved."""

from dataclasses import replace

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


def test_repeated_candidate_uids_in_recorded_games_do_not_stop_the_report() -> None:
    torch.manual_seed(4)
    base, moved = _model().eval(), _model().eval()
    batch = _batch(1)
    # Recorded games can carry the same UID on two candidates; scoring must not depend on it.
    batch = replace(batch, candidates=replace(batch.candidates, uid=torch.tensor([[101, 101, 101, 101, 0]])))
    gates = [GATE_ACT, GATE_WAIT, GATE_ACT, GATE_ACT]
    sequence = ILSequenceV4(
        observations=(batch,) * len(gates),
        actions=_human_actions(gates),
        episode_start=torch.tensor([[True]] + [[False]] * (len(gates) - 1)),
        valid_mask=torch.ones(len(gates), 1, dtype=torch.bool),
        returns=torch.zeros(len(gates), 1),
    )
    tally = Tally()
    with torch.no_grad():
        compare_sequence(
            base,
            [moved],
            sequence,
            [tally],
            temperatures=(0.33, 1.0, 2.0),
            width=base.config.board_width,
            generator=torch.Generator().manual_seed(2),
        )
    assert tally.steps == len(gates) and tally.sampled_acts == len(gates)
    # A WAIT names no candidate, so only the plays are left out of the human comparison.
    assert tally.human_dropped == gates.count(GATE_ACT)
    assert "repeated card UID" in report("moved", tally)


def test_states_without_a_legal_play_are_measured_on_the_gate_only() -> None:
    torch.manual_seed(5)
    base, moved = _model().eval(), _model().eval()
    playable = _batch(1)
    # The fixture's second row has no legal candidate, like a recorded state short of elixir.
    unplayable = _batch(2).index_select(torch.tensor([1]))
    observations = (playable, unplayable, unplayable, playable)
    gates = [GATE_ACT, GATE_WAIT, GATE_WAIT, GATE_ACT]
    sequence = ILSequenceV4(
        observations=observations,
        actions=_human_actions(gates),
        episode_start=torch.tensor([[True]] + [[False]] * (len(gates) - 1)),
        valid_mask=torch.ones(len(gates), 1, dtype=torch.bool),
        returns=torch.zeros(len(gates), 1),
    )
    tally = Tally()
    with torch.no_grad():
        compare_sequence(
            base,
            [moved],
            sequence,
            [tally],
            temperatures=(0.33, 1.0, 2.0),
            width=base.config.board_width,
            generator=torch.Generator().manual_seed(3),
        )
    assert tally.steps == len(gates)
    assert tally.sampled_acts == 2


def test_a_decision_that_cannot_be_scored_is_counted_not_fatal(monkeypatch) -> None:
    from native_runner.training.v4 import policy_drift

    def broken(*args, **kwargs):
        raise ValueError("odd recorded state")

    monkeypatch.setattr(policy_drift, "measure_step", broken)
    base = _model().eval()
    sequence = ILSequenceV4(
        observations=(_batch(1),) * 3,
        actions=_human_actions([GATE_WAIT] * 3),
        episode_start=torch.tensor([[True], [False], [False]]),
        valid_mask=torch.ones(3, 1, dtype=torch.bool),
        returns=torch.zeros(3, 1),
    )
    tally = Tally()
    with torch.no_grad():
        compare_sequence(
            base,
            [base],
            sequence,
            [tally],
            temperatures=(0.33, 1.0, 2.0),
            width=base.config.board_width,
            generator=torch.Generator().manual_seed(4),
        )
    assert tally.skipped == 3
    assert "3 decisions could not be scored" in report("x", tally)
