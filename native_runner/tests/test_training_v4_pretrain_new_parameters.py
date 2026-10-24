"""Offline imitation of the new (zero-start) layers only: everything else frozen, PPO state carried through."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from native_runner.tests.test_training_v4_general_expert_bc import _dataset
from native_runner.tests.test_training_v4_structural_inputs import CHILD, _alive_towers, _model
from native_runner.training.v4 import factory, ppo_expert_bc
from native_runner.training.v4.build_expert_manifest import build_manifests
from native_runner.training.v4.checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from native_runner.training.v4.components import zero_start_parameters
from native_runner.training.v4.ppo_expert_bc import load_expert_sampler_v4
from native_runner.training.v4.pretrain_new_parameters import (
    _share_shard_cache,
    batch_sequences,
    imitation_step,
    main,
    pretrain_new_parameters,
)


def _rich(batch):
    """Standing towers and a projectile in flight, so the new tower and timing paths get gradients."""

    features = batch.groups.child_features.clone()
    features[:, 0, CHILD["projectile_in_flight"]] = 1.0
    features[:, 0, CHILD["projectile_target_position_known"]] = 1.0
    features[:, 0, CHILD["projectile_target_y_over_board_height"]] = 0.6
    features[:, 0, CHILD["velocity_y_over_10"]] = 0.02
    return replace(
        batch,
        towers=_alive_towers(batch, [1.0, 0.8, 1.0, 1.0, 0.25, 0.6]),
        groups=replace(batch.groups, child_features=features),
    )


@pytest.fixture
def manifests(tmp_path, monkeypatch):
    from native_runner.tests import test_training_v4_general_expert_bc as general

    plain = general._batch
    monkeypatch.setattr(general, "_batch", lambda batch_size=2: _rich(plain(batch_size)))
    root, selection = _dataset(tmp_path)
    output = tmp_path / "manifests"
    build_manifests(selection, [root], output, validation_percent=25)
    return output


def _samplers(manifests):
    train = load_expert_sampler_v4(manifests / "train.json", seed=3, records_per_rank=1, window_steps=4)
    validation = load_expert_sampler_v4(
        manifests / "validation.json", seed=5, records_per_rank=1, window_steps=4, split="validation"
    )
    return train, validation


def test_validation_games_load_only_when_asked_for(manifests) -> None:
    train, validation = _samplers(manifests)
    assert {row["replay_tag"] for row in train.rows}.isdisjoint(row["replay_tag"] for row in validation.rows)
    with pytest.raises(ValueError, match="must not be used for training"):
        load_expert_sampler_v4(manifests / "validation.json", seed=1)
    with pytest.raises(ValueError, match="expected a validation"):
        load_expert_sampler_v4(manifests / "train.json", seed=1, split="validation")
    with pytest.raises(ValueError, match="train or validation"):
        load_expert_sampler_v4(manifests / "train.json", seed=1, split="test")


def test_only_new_layers_learn_and_the_best_validation_step_is_kept(manifests) -> None:
    torch.manual_seed(2)
    model = _model()
    new_names = {name for name, _parameter in zero_start_parameters(model)}
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    train, validation = _samplers(manifests)
    result = pretrain_new_parameters(
        model,
        train,
        steps=4,
        learning_rate=1e-2,
        chunk_steps=4,
        validation=validation,
        validation_batches=1,
        eval_every=1,
    )
    rows = result["rows"]
    assert [row["step"] for row in rows] == [0, 1, 2, 3, 4]
    losses = [row["validation_total"] for row in rows]
    assert result["chosen_step"] == losses.index(min(losses)) > 0
    assert min(losses) < losses[0]
    assert all("train_loss" in row for row in rows[1:])
    for name, parameter in model.named_parameters():
        if name in new_names:
            continue
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    moved = {name for name, parameter in zero_start_parameters(model) if bool(parameter.any())}
    assert {"spatial_encoder.tower_input.weight", "child_encoder.timing.weight"} <= moved
    # The kept layers are the chosen step's, and every parameter is trainable again for later use.
    rms = {
        key.removeprefix("rms/"): value for key, value in rows[result["chosen_step"]].items() if key.startswith("rms/")
    }
    assert rms["spatial_encoder.tower_input"] == pytest.approx(
        float(dict(zero_start_parameters(model))["spatial_encoder.tower_input.weight"].square().mean().sqrt())
    )
    assert all(parameter.requires_grad for parameter in model.parameters())


def _source_checkpoint(path):
    torch.manual_seed(4)
    model = _model()
    model.set_ppo_gate_temperature(0.32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-5, weight_decay=0.0)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    ages = {name: 21 for name, _parameter in zero_start_parameters(model)}
    save_actor_critic_checkpoint(
        path,
        model,
        optimizer=optimizer,
        update_step=14,
        training_stage="ppo",
        gamma_per_decision=0.99,
        parameter_ages=ages,
        extra={"resume_contract": {"kept": True}},
    )
    return model, optimizer


def test_command_writes_a_warm_start_that_differs_only_in_the_new_layers(tmp_path, manifests, monkeypatch) -> None:
    monkeypatch.setattr(factory, "build_production_model_v4", lambda device=None: _model().to(device))
    source = tmp_path / "update-00000014.pt"
    source_model, source_optimizer = _source_checkpoint(source)
    output = tmp_path / "pretrained.pt"
    common = [
        "--manifest",
        str(manifests / "train.json"),
        "--records",
        "1",
        "--window-steps",
        "4",
        "--chunk-steps",
        "4",
        "--validation-batches",
        "1",
        "--device",
        "cpu",
    ]
    main(
        [str(source), "--output", str(output), "--steps", "3", "--eval-every", "1", "--learning-rate", "1e-2", *common]
    )

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-5, weight_decay=0.0)
    payload = load_actor_critic_checkpoint(output, model, optimizer=optimizer)
    assert payload["update_step"] == 14 and payload["training_stage"] == "ppo"
    assert model.ppo_gate_temperature == pytest.approx(0.32)
    assert payload["parameter_ages"] == {name: 21 for name, _parameter in zero_start_parameters(model)}
    summary = payload["extra"]["new_parameter_pretraining"]
    assert payload["extra"]["resume_contract"] == {"kept": True}
    assert summary["chosen_step"] > 0 and summary["validation_loss_chosen"] < summary["validation_loss_start"]
    assert summary["validation_manifest"].endswith("validation.json")
    new_names = {name for name, _parameter in zero_start_parameters(model)}
    expected = dict(source_model.named_parameters())
    for name, parameter in model.named_parameters():
        if name not in new_names:
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    # The PPO optimizer state is the source's, untouched by the imitation steps.
    saved_state = optimizer.state_dict()["state"]
    for index, state in source_optimizer.state_dict()["state"].items():
        torch.testing.assert_close(saved_state[index]["exp_avg"], state["exp_avg"], rtol=0, atol=0)
    log = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert [row["step"] for row in log if row["event"] == "eval"] == [0, 1, 2, 3]
    assert log[-1]["event"] == "saved"

    with pytest.raises(SystemExit):
        main([str(source), "--output", str(output), "--steps", "0", *common])  # refuses to overwrite
    with pytest.raises(SystemExit):
        main(["", "--output", str(tmp_path / "unset.pt"), "--steps", "0", *common])  # an unset $SRC

    ablation = tmp_path / "zeroed.pt"
    main([str(output), "--output", str(ablation), "--steps", "0", "--reset-new-parameters", *common])
    payload = load_actor_critic_checkpoint(ablation, model)
    assert not any(bool(parameter.any()) for _name, parameter in zero_start_parameters(model))
    assert set(payload["parameter_ages"].values()) == {0}


def test_shards_are_read_once_when_they_fit_in_memory(tmp_path, monkeypatch) -> None:
    root, selection = _dataset(tmp_path, multiple_shards=True)
    build_manifests(selection, [root], tmp_path / "split", validation_percent=25)
    train, validation = _samplers(tmp_path / "split")
    loads = []
    real = ppo_expert_bc.load_il_cache_shard
    monkeypatch.setattr(
        ppo_expert_bc, "load_il_cache_shard", lambda path, verify=True: loads.append(path) or real(path, verify=verify)
    )
    _share_shard_cache(train, validation, cache_gb=100)
    assert train.cached_shards == validation.cached_shards == 2 and train._shards is validation._shards
    for update in range(1, 9):
        train.sample(update, 0, 1)
        validation.sample(update, 0, 1)
    assert len(loads) == len(set(loads)) == 2
    # PPO's default keeps a single shard: memory never holds more than that.
    ppo, _validation = _samplers(tmp_path / "split")
    assert ppo.cached_shards == 1
    for update in range(1, 9):
        ppo.sample(update, 0, 1)
        assert len(ppo._shards) == 1


def _rank_main(rank: int, port: int, argv: list[str]) -> None:
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    factory.build_production_model_v4 = lambda device=None: _model().to(device)
    torch.set_num_threads(1)
    main(argv)


def test_two_processes_train_together_and_only_the_first_saves(tmp_path, manifests) -> None:
    import multiprocessing
    import socket

    source = tmp_path / "update-00000014.pt"
    source_model, _optimizer = _source_checkpoint(source)
    output = tmp_path / "pretrained.pt"
    argv = [
        str(source),
        "--output",
        str(output),
        "--manifest",
        str(manifests / "train.json"),
        "--steps",
        "2",
        "--eval-every",
        "1",
        "--learning-rate",
        "1e-2",
        "--records",
        "1",
        "--window-steps",
        "4",
        "--chunk-steps",
        "4",
        "--validation-batches",
        "1",
        "--load-workers",
        "2",
        "--device",
        "cpu",
    ]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_rank_main, args=(rank, port, argv)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=600)
    assert [process.exitcode for process in processes] == [0, 0]

    model = _model()
    payload = load_actor_critic_checkpoint(output, model)
    summary = payload["extra"]["new_parameter_pretraining"]
    assert summary["world_size"] == 2 and summary["chosen_step"] > 0
    new_names = {name for name, _parameter in zero_start_parameters(model)}
    expected = dict(source_model.named_parameters())
    for name, parameter in model.named_parameters():
        if name not in new_names:
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    log = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert [row["step"] for row in log if row["event"] == "eval"] == [0, 1, 2]  # one log, written by rank 0


def test_games_side_by_side_train_exactly_like_the_ppo_imitation_step(manifests) -> None:
    train = load_expert_sampler_v4(manifests / "train.json", seed=3, records_per_rank=3, window_steps=4)
    items = train.sample(5, 0, 1)
    assert len({item.sequence.time_steps for item in items}) > 1  # lanes of different lengths get padded
    batch = batch_sequences([item.sequence for item in items])
    assert batch.batch_size == 3 and batch.time_steps == max(item.sequence.time_steps for item in items)
    torch.manual_seed(7)
    reference = _model()
    with torch.no_grad():
        for _name, parameter in zero_start_parameters(reference):
            parameter.normal_(std=0.05)
    batched = _model()
    batched.load_state_dict(reference.state_dict())
    # Plain SGD moves each parameter in proportion to its gradient, so this compares the gradients themselves.
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    batched_optimizer = torch.optim.SGD(batched.parameters(), lr=0.1)
    expected = ppo_expert_bc.distributed_expert_bc_update_v4(
        reference, reference_optimizer, train, update_step=5, sequence_chunk_steps=4, coefficient=1.0, max_grad_norm=1.0
    )
    actual = imitation_step(
        batched, batched_optimizer, items, chunk_steps=4, max_grad_norm=1.0, gate_act_weight=train.gate_act_weight
    )
    for name in ("bc_loss", "bc_gradient_norm", *(f"bc_{head}_loss" for head in ppo_expert_bc.HEADS)):
        assert actual[name] == pytest.approx(expected[name], rel=1e-5, abs=1e-7), name
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in batched.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name], rtol=1e-5, atol=1e-6)


class _Stopped(Exception):
    pass


def test_a_stopped_run_resumes_from_its_last_evaluation_and_ends_where_an_uninterrupted_one_does(
    tmp_path, manifests, monkeypatch
) -> None:
    from native_runner.training.v4 import pretrain_new_parameters as tool

    monkeypatch.setattr(factory, "build_production_model_v4", lambda device=None: _model().to(device))
    source = tmp_path / "update-00000014.pt"
    _source_checkpoint(source)

    def command(output, *extra):
        return [
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifests / "train.json"),
            "--steps",
            "5",
            "--eval-every",
            "2",
            "--learning-rate",
            "1e-2",
            "--records",
            "1",
            "--window-steps",
            "4",
            "--chunk-steps",
            "4",
            "--validation-batches",
            "1",
            "--device",
            "cpu",
            *extra,
        ]

    straight = tmp_path / "straight.pt"
    main(command(straight))
    assert not tool._progress_path(straight).exists()  # removed once the output is written

    # Stop the second run partway through step 3, after the evaluation at step 2 saved its progress.
    stopped = tmp_path / "stopped.pt"
    real = tool.imitation_step
    calls = []

    def stop_at_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise _Stopped
        return real(*args, **kwargs)

    monkeypatch.setattr(tool, "imitation_step", stop_at_third)
    with pytest.raises(_Stopped):
        main(command(stopped))
    monkeypatch.setattr(tool, "imitation_step", real)
    progress = tool._progress_path(stopped)
    assert torch.load(progress, weights_only=False)["step"] == 2 and not stopped.exists()

    with pytest.raises(SystemExit):
        main(command(stopped))  # an unfinished run is never overwritten by accident
    with pytest.raises(SystemExit, match="learning_rate"):
        main([*command(stopped, "--resume"), "--learning-rate", "1e-3"])
    with pytest.raises(SystemExit):
        main(command(tmp_path / "never-started.pt", "--resume"))

    main(command(stopped, "--resume"))
    assert not progress.exists()
    log = [json.loads(line) for line in stopped.with_suffix(".jsonl").read_text().splitlines()]
    assert [row["step"] for row in log if row["event"] == "eval"] == [0, 2, 4, 5]
    straight_log = [json.loads(line) for line in straight.with_suffix(".jsonl").read_text().splitlines()]
    for resumed_row, straight_row in zip(log[:-1], straight_log[:-1], strict=True):
        assert resumed_row["validation_total"] == pytest.approx(straight_row["validation_total"], rel=1e-6)
    expected, actual = _model(), _model()
    straight_payload = load_actor_critic_checkpoint(straight, expected)
    resumed_payload = load_actor_critic_checkpoint(stopped, actual)
    summary = resumed_payload["extra"]["new_parameter_pretraining"]
    assert summary["chosen_step"] == straight_payload["extra"]["new_parameter_pretraining"]["chosen_step"]
    for (name, parameter), (_name, reference) in zip(
        zero_start_parameters(actual), zero_start_parameters(expected), strict=True
    ):
        torch.testing.assert_close(parameter, reference, rtol=1e-5, atol=1e-7, msg=name)
