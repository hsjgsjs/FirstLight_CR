import json
import random

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.tests.test_training_v4_ppo_expert_bc import _one_action, _wait_actions
from native_runner.training.v4.build_expert_manifest import (
    audit_selection, build_manifests, cache_status, selected_sides, split_games,
)
from native_runner.training.v4.cache import write_il_cache_index, write_il_cache_shard
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.ppo_expert_bc import (
    GeneralExpertSamplerV4, distributed_expert_bc_update_v4, expert_bc_contract_v4,
    general_window_mask, load_expert_sampler_v4,
)
from native_runner.training.v4.producer import ReplayProductionResultV4


def _dataset(tmp_path, *, failed=False):
    batch = _batch(batch_size=1)
    batch.candidates.native_visible_card_id[batch.candidates.mask] = 26000000  # no Hog labels
    results, selections = [], []
    for n in range(4):
        tag = f"game-{n}"
        sequences = []
        for owner in (0, 1):
            valid = torch.ones(12, 1, dtype=torch.bool)
            valid[6:8] = False
            gate = valid.clone()
            gate[2] = False
            sequences.append(ILSequenceV4(
                (batch,) * 12,
                tuple(_one_action(1) if t in (1, 8, 10) else _wait_actions(1) for t in range(12)),
                torch.tensor([[t == 0] for t in range(12)]), valid, torch.zeros(12, 1),
                gate_loss_mask=gate, value_loss_mask=torch.zeros_like(valid),
                sequence_id=f"{tag}:owner-{owner}", natural_time=True,
            ))
        is_failed = failed and n == 0
        results.append(ReplayProductionResultV4(
            tag, tuple(sequences), not is_failed, "RuntimeError: rejected action" if is_failed else None,
            1200 if is_failed else None, 1200, 1200, 6, 6, True, True, 12,
        ))
        selections.append(dict(replay_tag=tag, learn=[
            dict(owner=0, outcome="win", weight=1.0), dict(owner=1, outcome="close_loss", weight=0.5),
        ]))
    root = tmp_path / "cache"
    root.mkdir()
    data = root / "shard.cril.zst"
    summary = write_il_cache_shard(data, results, shard_id="shard", contract={})
    summary["summary_file"] = data.with_suffix(".json").name
    write_il_cache_index(root / "index.json", [summary], contract={})
    selection = tmp_path / "selection.jsonl"
    selection.write_text("\n".join(json.dumps(r) for r in selections))
    return root, selection


def test_general_cache_audit_split_sampler_and_real_bc_step(tmp_path):
    root, selection = _dataset(tmp_path)
    output = tmp_path / "manifests"
    summary = build_manifests(selection, [root], output, validation_percent=25)
    assert summary["train"]["games"] == 3 and summary["validation"]["games"] == 1
    train = json.loads((output / "train.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    assert not ({r["replay_tag"] for r in train["rows"]} & {r["replay_tag"] for r in validation["rows"]})
    assert all(r["cached_hog_count"] == 0 for r in train["rows"])
    assert all(r["cached_micro_actions"] == 3 for r in train["rows"])
    sampler = load_expert_sampler_v4(output / "train.json", seed=7, records_per_rank=1, window_steps=4)
    assert isinstance(sampler, GeneralExpertSamplerV4)
    item = sampler.sample(4, 0, 1)[0]
    assert not item.sequence.valid_mask[6:8].any()
    assert not item.sequence.gate_loss_mask[2]
    assert not item.sequence.value_loss_mask.any()
    assert bool(item.sequence.episode_start[0, 0])
    assert any(int(action.micro_action_count.sum()) for action, valid in
               zip(item.sequence.actions, item.sequence.valid_mask) if valid.any())
    before_rng = random.getstate()
    assert sampler.rows_for_update(4, 0, 1) == sampler.rows_for_update(4, 0, 1)
    assert random.getstate() == before_rng
    with pytest.raises(ValueError, match="validation"):
        load_expert_sampler_v4(output / "validation.json", seed=7)
    with pytest.raises(ValueError, match="Hog-specific"):
        expert_bc_contract_v4(sampler, hog_gate_weight=1)
    assert expert_bc_contract_v4(sampler, every_updates=4)["every_updates"] == 4
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    value_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("value_head.")}
    metrics = distributed_expert_bc_update_v4(model, optimizer, sampler, update_step=4,
                                             sequence_chunk_steps=2, coefficient=0.1, validate=True)
    assert metrics["bc_gate_act_weight"] == 1 and metrics["bc_optimizer_steps"] == 1
    assert metrics["bc_hog_labels"] == 0 and metrics["bc_loss"] > 0
    for n, p in model.named_parameters():
        if n in value_before:
            torch.testing.assert_close(p, value_before[n], rtol=0, atol=0)
    # Audited counts/hash must still match when a cache is used by the trainer.
    sampler.rows[0]["cached_micro_actions"] = 100
    sampler.rows = sampler.rows[:1]
    with pytest.raises(ValueError, match="labels changed"):
        sampler.sample(8, 0, 1)


def test_audit_excludes_failed_and_missing_sides_and_detects_duplicate_cache(tmp_path):
    root, selection = _dataset(tmp_path, failed=True)
    extra = dict(replay_tag="missing", learn=[dict(owner=0, outcome="win", weight=1)])
    with selection.open("a") as handle:
        handle.write("\n" + json.dumps(extra))
    rows, excluded = audit_selection(selection, [root])
    assert len(rows) == 6 and len(excluded) == 3
    assert sum(r["reason"] == "selected owner missing from cache" for r in excluded) == 1
    import shutil
    other = tmp_path / "duplicate"
    shutil.copytree(root, other)
    with pytest.raises(ValueError, match="duplicate cache"):
        audit_selection(selection, [root, other])


def test_whole_game_split_is_order_independent():
    rows = [dict(replay_tag=str(i), owner=owner) for i in range(100) for owner in (0, 1)]
    forward = split_games(rows, seed=3)
    reverse = split_games(list(reversed(rows)), seed=3)
    for name in forward:
        assert {(r["replay_tag"], r["owner"]) for r in forward[name]} == {
            (r["replay_tag"], r["owner"]) for r in reverse[name]}
    assert len(forward["validation"]) == 20


@pytest.mark.parametrize("reason,allowed", [
    ("terminal fidelity mismatch: winner", True),
    ("battle ended before all expert actions executed", True),
    ("RuntimeError: rejected action", False),
])
def test_trusted_prefix_is_explicit_and_narrow(reason, allowed):
    result = dict(completed=False, failure_reason=reason, decision_frame_count=40,
                  simulated_end_tick=1200, first_untrusted_tick=1200)
    assert cache_status(result) == "excluded"
    assert (cache_status(result, allow_trusted_prefix=True) == "trusted_prefix") == allowed
    result["simulated_end_tick"] = 700
    assert cache_status(result, allow_trusted_prefix=True) == "excluded"


def test_general_windows_keep_wait_and_invalid_gaps_in_long_empty_opening():
    valid = torch.ones(1000, 1, dtype=torch.bool)
    valid[100:120] = False
    selected = general_window_mask(valid, [900], window_steps=32, rng=random.Random(2))
    assert selected[:32].all() and selected[900]
    assert not (selected & ~valid).any()


@pytest.mark.parametrize("percent", [0, 100, float("nan")])
def test_invalid_split_percent(percent):
    with pytest.raises(ValueError, match="validation percent"):
        split_games([dict(replay_tag="a"), dict(replay_tag="b")], validation_percent=percent)


def test_duplicate_or_invalid_source_side_is_rejected(tmp_path):
    path = tmp_path / "selection.jsonl"
    item = dict(replay_tag="a", learn=[dict(owner=True, outcome="win", weight=1)])
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="native owner"):
        selected_sides(path)
    item["learn"][0]["owner"] = 0
    path.write_text(json.dumps(item) + "\n" + json.dumps(item))
    with pytest.raises(ValueError, match="duplicate"):
        selected_sides(path)


def test_prepared_trial_pins_optimizer_and_continues_temperature(tmp_path):
    from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
    from native_runner.training.v4.prepare_expert_experiment import prepare_experiment, continuation_options
    from native_runner.training.v4.train_ppo_self_play_cluster import _gate_temperature_for_update

    root, selection = _dataset(tmp_path)
    manifests = tmp_path / "manifests"
    build_manifests(selection, [root], manifests)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.set_ppo_gate_temperature(_gate_temperature_for_update(1209, start=0.5, end=0.2, anneal_updates=3000))
    checkpoint = tmp_path / "source.pt"
    save_actor_critic_checkpoint(checkpoint, model, optimizer=optimizer, update_step=1208,
                                training_stage="ppo-distributed", gamma_per_decision=0.9997,
                                extra={"resume_contract": dict(gate_temperature_start=0.5,
                                       gate_temperature_end=0.2, gate_temperature_anneal_updates=3000)})
    options = tmp_path / "options.txt"
    options.write_text("--execution-delay-ticks 18-24 --gate-temperature-start 0.5 "
                       "--gate-temperature-end=0.2 --gate-temperature-anneal-updates 3000 "
                       "--expert-bc-manifest old.json --expert-bc-hog-gate-weight 0.5")
    output = tmp_path / "trial"
    plan = prepare_experiment(checkpoint, options, manifests / "train.json", output)
    assert plan["source_update"] == 1208 and plan["warm_start_optimizer"]
    assert (output / "baseline.pt").read_bytes() == checkpoint.read_bytes()
    control = (output / "ppo-options.txt").read_text().split()
    human = (output / "human-options.txt").read_text().split()
    assert not any(token.startswith("--expert-bc-") for token in control)
    assert "--warm-start-optimizer" in control
    assert human[:len(control)] == control and "--expert-bc-hog-gate-weight" not in human
    remaining = int(control[control.index("--gate-temperature-anneal-updates") + 1])
    start = float(control[control.index("--gate-temperature-start") + 1])
    assert remaining == 1792
    for step in (1, 2, 200, 1792):
        assert _gate_temperature_for_update(step, start=start, end=0.2, anneal_updates=remaining) == pytest.approx(
            _gate_temperature_for_update(1208 + step, start=0.5, end=0.2, anneal_updates=3000), abs=1e-12)
    with pytest.raises(FileExistsError):
        prepare_experiment(checkpoint, options, manifests / "train.json", output)
    with pytest.raises(ValueError, match="train manifest"):
        prepare_experiment(checkpoint, options, manifests / "validation.json", tmp_path / "invalid")
    with pytest.raises(ValueError, match="whitespace"):
        continuation_options('--fixed-opponent-checkpoint "path with spaces.pt"',
                             torch.load(checkpoint, weights_only=False))
