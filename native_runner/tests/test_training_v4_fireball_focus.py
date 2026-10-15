import json

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.tests.test_training_v4_ppo_expert_bc import _one_action, _wait_actions
from native_runner.training.v4.build_expert_manifest import build_manifests
from native_runner.training.v4.cache import load_il_cache_shard, write_il_cache_index, write_il_cache_shard
from native_runner.training.v4.fireball_focus import build_focus_manifest, focus_frames
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.ppo_expert_bc import load_expert_sampler_v4, distributed_expert_bc_update_v4
from native_runner.training.v4.producer import ReplayProductionResultV4


def dataset(tmp_path):
    results, selections = [], []
    for n in range(4):
        batch = _batch(batch_size=1)
        batch.candidates.native_visible_card_id[0, 0] = 28000000
        batch.groups.child_features[0, 2, 9] = 0.1  # visible enemy moves 1 tile/sec
        batch.groups.child_type[0, 2] = 0
        actions = []
        for t in range(100):
            action = _one_action(1) if t in (10, 70) else _wait_actions(1)
            if t in (10, 70):
                action.target_cell[0, 0] = 20 * 18 + (13 if n % 2 == 0 else 0)
            actions.append(action)
        valid = torch.ones(100, 1, dtype=torch.bool)
        sequence = ILSequenceV4((batch,) * 100, tuple(actions),
                                torch.tensor([[t == 0] for t in range(100)]), valid, torch.zeros(100, 1),
                                gate_loss_mask=valid.clone(), value_loss_mask=torch.zeros_like(valid),
                                sequence_id=f"game-{n}:owner-0", natural_time=True)
        results.append(ReplayProductionResultV4(f"game-{n}", (sequence,), True, None, None,
                                               1200, 1200, 2, 2, True, True, 100))
        selections.append(dict(replay_tag=f"game-{n}", learn=[dict(owner=0, outcome="win", weight=1)]))
    root = tmp_path / "cache"
    root.mkdir()
    path = root / "part.cril.zst"
    summary = write_il_cache_shard(path, results, shard_id="part", contract={})
    summary["summary_file"] = path.with_suffix(".json").name
    write_il_cache_index(root / "index.json", [summary], contract={})
    selection = tmp_path / "selection.jsonl"
    selection.write_text("\n".join(json.dumps(x) for x in selections))
    manifests = tmp_path / "manifests"
    build_manifests(selection, [root], manifests, validation_percent=25)
    return manifests, path.with_suffix(".json")


def test_focus_filters_real_actions_moving_visible_enemies_and_preserves_splits(tmp_path):
    manifests, shard_path = dataset(tmp_path)
    source = json.loads((manifests / "train.json").read_bytes())
    output = tmp_path / "focus.json"
    summary = build_focus_manifest(manifests / "train.json", output)
    focused = json.loads(output.read_bytes())
    assert {r["replay_tag"] for r in source["rows"]} == {r["replay_tag"] for r in focused["rows"]}
    assert summary["focused_sides"] > 0 and summary["focused_sides"] < summary["all_sides"]
    assert all(row["fireball_focus_frames"] in ([], [10, 70]) for row in focused["rows"])
    shard = load_il_cache_shard(shard_path, verify=True)
    config = focused["focus"]
    assert focus_frames(shard, 0, config) == [10, 70]
    # Friendly, stationary, far-away, projectile, untrusted gate/action cases do not qualify.
    shard.observations.groups.owner_type[:, 1] = 0
    assert not focus_frames(shard, 0, config)
    shard.observations.groups.owner_type[:, 1] = 1
    shard.observations.groups.child_features[:, 2, 9] = 0
    assert not focus_frames(shard, 0, config)
    shard.observations.groups.child_features[:, 2, 9] = 0.1
    shard.observations.groups.child_type[:, 2] = 3
    assert not focus_frames(shard, 0, config)
    shard.observations.groups.child_type[:, 2] = 0
    shard.gate_loss_mask[10, 0] = False
    shard.valid_mask[70, 0] = False
    assert not focus_frames(shard, 0, config)
    with pytest.raises(FileExistsError):
        build_focus_manifest(manifests / "train.json", output)


def test_focus_sampler_retains_wait_burnin_general_examples_and_real_update(tmp_path):
    manifests, _ = dataset(tmp_path)
    path = tmp_path / "focus.json"
    build_focus_manifest(manifests / "train.json", path, percent=80)
    sampler = load_expert_sampler_v4(path, seed=7, records_per_rank=2, window_steps=8)
    flags = [r["sampled_fireball_focus"] for update in range(1, 101)
             for r in sampler.rows_for_update(update, 0, 1)]
    assert True in flags and False in flags
    assert sampler.rows_for_update(3, 0, 1) == sampler.rows_for_update(3, 0, 1)
    selected_update = next(u for u in range(1, 101) if any(r["sampled_fireball_focus"]
                             for r in sampler.rows_for_update(u, 0, 1)))
    items = sampler.sample(selected_update, 0, 1)
    for item in items:
        if item.metadata["sampled_fireball_focus"]:
            assert any(t < item.sequence.time_steps and item.sequence.valid_mask[t, 0]
                       for t in item.metadata["fireball_focus_frames"])
        assert item.sequence.episode_start[0, 0] and not item.sequence.value_loss_mask.any()
        assert item.sequence.valid_mask[:8].all()
    assert sampler.contract()["focus"]["percent"] == 80
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    metrics = distributed_expert_bc_update_v4(model, optimizer, sampler, update_step=selected_update,
                                            coefficient=0.05, sequence_chunk_steps=8, validate=False)
    assert metrics["bc_fireball_focus_records"] > 0 and metrics["bc_fireball_moving_frames"] > 0
    assert metrics["bc_optimizer_steps"] == 1
    # A changed annotation cannot silently train a different context.
    payload = json.loads(path.read_bytes())
    for row in payload["rows"]:
        if row["fireball_focus_frames"]:
            row["fireball_focus_frames"] = [11]
    path.write_text(json.dumps(payload))
    wrong = load_expert_sampler_v4(path, seed=7, records_per_rank=2, window_steps=8)
    with pytest.raises(ValueError, match="annotations changed"):
        wrong.sample(selected_update, 0, 1)


def test_invalid_focus_configuration_is_rejected(tmp_path):
    manifests, _ = dataset(tmp_path)
    for percent in (0, 100, float("nan")):
        with pytest.raises(ValueError, match="percent"):
            build_focus_manifest(manifests / "train.json", tmp_path / "invalid.json", percent=percent)


def test_parallel_focus_annotation_matches_serial_and_validation_stays_held_out(tmp_path):
    manifests, _ = dataset(tmp_path)
    serial = tmp_path / "serial.json"
    parallel = tmp_path / "parallel.json"
    build_focus_manifest(manifests / "train.json", serial)
    build_focus_manifest(manifests / "train.json", parallel, workers=2)
    assert serial.read_bytes() == parallel.read_bytes()
    heldout = json.loads((manifests / "validation.json").read_bytes())
    heldout["focus"] = json.loads(serial.read_bytes())["focus"]
    heldout["rows"][0]["fireball_focus_frames"] = [10, 70]
    validation = tmp_path / "validation-focus.json"
    validation.write_text(json.dumps(heldout))
    with pytest.raises(ValueError, match="validation"):
        load_expert_sampler_v4(validation, seed=7)
