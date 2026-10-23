"""Train only the new (zero-start) input layers on human games, with every other parameter frozen.

In PPO these layers learn from a gradient that is mostly noise (``snr`` about 0.05 per update), so a larger step
size grows them about as fast in noise as in signal. Human games carry a label at every decision, so here the
same layers learn from a much cleaner signal, and the frozen rest of the policy cannot drift.

The output is the source checkpoint with only the new layers changed. Optimizer state, behaviour temperatures,
update step and parameter ages are carried over, so it is a drop-in PPO warm start. The layers kept are the ones
with the lowest loss on held-out human games (step 0 included), so the result never predicts humans worse than
the source. ``--steps 0 --reset-new-parameters`` writes the source with its new layers zeroed: the ablation copy
for checking in an evaluation whether the new layers help.

Several GPUs: ``torchrun --nproc-per-node 2 -m native_runner.training.v4.pretrain_new_parameters ...``. Each GPU
takes ``--records`` games per step and the gradients are averaged, as in the PPO trainer's imitation step.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from . import ppo_expert_bc
from .checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from .components import zero_start_parameters
from .imitation import ILLossWeightsV4, imitation_loss, slice_il_sequence
from .learning import evaluate_recurrent_sequence
from .ppo_expert_bc import COUNT_NAMES, HEADS, _sum, _world, distributed_expert_bc_update_v4, load_expert_sampler_v4

# Same as the PPO imitation step.
DELAY_NEIGHBOR_WEIGHT = 0.2
# Validation windows depend only on this seed, so runs with different --seed values compare on the same games.
VALIDATION_SEED = 20261024


def freeze_all_but_new_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    new = zero_start_parameters(model)
    if not new:
        raise ValueError("the model has no zero-start parameters to train")
    model.requires_grad_(False)
    for _name, parameter in new:
        parameter.requires_grad_(True)
    return new


def reset_new_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for _name, parameter in zero_start_parameters(model):
            parameter.zero_()


def new_parameter_rms(model: nn.Module) -> dict[str, float]:
    return {
        name.removesuffix(".weight"): float(parameter.detach().float().square().mean().sqrt())
        for name, parameter in zero_start_parameters(model)
    }


@torch.no_grad()
def imitation_losses(model: nn.Module, sampler, *, batches: int, chunk_steps: int) -> dict[str, float]:
    """Per-head imitation loss on a fixed set of human windows: the same windows on every call.

    Each window replays its game from the real start for the recurrent state, exactly as the training step does.
    With several GPUs each takes its share of the windows and the sums are combined.
    """

    device = next(model.parameters()).device
    rank, world_size = _world()
    sums = torch.zeros(len(HEADS), dtype=torch.float64, device=device)
    counts = torch.zeros(len(HEADS), dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    for batch in range(1, batches + 1):
        for item in sampler.sample(batch, rank, world_size):
            sequence, state = item.sequence, None
            for start in range(0, sequence.time_steps, chunk_steps):
                chunk = slice_il_sequence(sequence, start, min(start + chunk_steps, sequence.time_steps)).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    evaluation = evaluate_recurrent_sequence(
                        model,
                        chunk.observations,
                        chunk.actions,
                        chunk.episode_start,
                        initial_state=state,
                        gate_temperature=1.0,
                        action_temperature=1.0,
                        continue_temperature=1.0,
                        validate=False,
                        preencode_observations=True,
                    )
                    state = evaluation.final_state
                    if bool(chunk.valid_mask.any()):
                        loss = imitation_loss(
                            chunk,
                            evaluation,
                            weights=ILLossWeightsV4(value=0.0),
                            gate_act_weight=sampler.gate_act_weight,
                            delay_neighbor_weight=DELAY_NEIGHBOR_WEIGHT,
                        )
                        head_counts = torch.stack([getattr(loss, name) for name in COUNT_NAMES]).double()
                        sums += torch.stack([getattr(loss, name) for name in HEADS]).double() * head_counts
                        counts += head_counts
    model.train(was_training)
    sums, counts = _sum(sums).cpu(), _sum(counts).cpu()
    means = sums / counts.clamp_min(1)
    result = {name: float(value) for name, value in zip(HEADS, means.tolist(), strict=True)}
    result["total"] = float(means.sum())
    result.update({name: float(value) for name, value in zip(COUNT_NAMES, counts.tolist(), strict=True)})
    return result


def pretrain_new_parameters(
    model: nn.Module,
    sampler,
    *,
    steps: int,
    learning_rate: float,
    chunk_steps: int,
    max_grad_norm: float = 1.0,
    validation=None,
    validation_batches: int = 8,
    eval_every: int = 250,
    log: Callable[[Mapping[str, object]], None] = lambda row: None,
    progress: Callable[[str], None] = lambda text: None,
    progress_every: int = 10,
) -> dict[str, object]:
    """Imitation steps on the new parameters only; keeps the ones with the lowest validation loss.

    Without a validation sampler the last step is kept. Returns the evaluation rows and the chosen step.
    """

    if steps < 0 or min(eval_every, validation_batches, chunk_steps, progress_every) <= 0:
        raise ValueError("steps must be nonnegative, and evaluation and chunk sizes positive")
    if not learning_rate > 0:
        raise ValueError("learning rate must be positive")
    new = freeze_all_but_new_parameters(model)
    optimizer = torch.optim.AdamW([parameter for _name, parameter in new], lr=learning_rate, eps=1e-5, weight_decay=0.0)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    best: tuple[float, int, dict[str, torch.Tensor]] | None = None

    def evaluate(step: int, train: Mapping[str, float]) -> None:
        nonlocal best
        row: dict[str, object] = {"event": "eval", "step": step, "seconds": round(time.perf_counter() - started, 1)}
        row.update({f"train_{name}": value for name, value in train.items()})
        if validation is not None:
            games = validation_batches * validation.records_per_rank * _world()[1]
            progress(f"step {step}: evaluating on {games} held-out games")
            losses = imitation_losses(model, validation, batches=validation_batches, chunk_steps=chunk_steps)
            row.update({f"validation_{name}": value for name, value in losses.items()})
            if best is None or losses["total"] < best[0]:
                best = (losses["total"], step, {name: parameter.detach().clone() for name, parameter in new})
        row.update({f"rms/{name}": value for name, value in new_parameter_rms(model).items()})
        rows.append(row)
        log(row)

    try:
        evaluate(0, {})
        window: dict[str, float] = {}
        window_steps = 0
        recent_loss, recent_steps, recent_started = 0.0, 0, time.perf_counter()
        for step in range(1, steps + 1):
            metrics = distributed_expert_bc_update_v4(
                model,
                optimizer,
                sampler,
                update_step=step,
                sequence_chunk_steps=chunk_steps,
                coefficient=1.0,
                max_grad_norm=max_grad_norm,
            )
            for name in ("loss", *(f"{head}_loss" for head in HEADS), "gradient_norm"):
                window[name] = window.get(name, 0.0) + metrics[f"bc_{name}"]
            window_steps += 1
            recent_loss += metrics["bc_loss"]
            recent_steps += 1
            if step % progress_every == 0:
                seconds = (time.perf_counter() - recent_started) / recent_steps
                progress(
                    f"step {step}/{steps}: {seconds:.1f} s/step, train loss {recent_loss / recent_steps:.4f}, "
                    f"about {seconds * (steps - step) / 60:.0f} min left before evaluations"
                )
                recent_loss, recent_steps, recent_started = 0.0, 0, time.perf_counter()
            if step % eval_every == 0 or step == steps:
                evaluate(step, {name: value / window_steps for name, value in window.items()})
                window, window_steps = {}, 0
        chosen_step = steps
        if best is not None:
            chosen_step = best[1]
            with torch.no_grad():
                for name, parameter in new:
                    parameter.copy_(best[2][name])
    finally:
        model.requires_grad_(True)
    return {"chosen_step": chosen_step, "rows": rows}


def _share_shard_cache(
    sampler,
    validation,
    *,
    cache_gb: float,
    workers: int = 1,
    validation_batches: int = 0,
    say: Callable[[str], None] = lambda text: print(text, flush=True),
) -> None:
    """Keep as many unpacked shards in memory as ``cache_gb`` holds, shared by training and validation.

    Each record otherwise reads, verifies and unpacks its whole shard again, which is most of the time per step.
    The cache is filled up front by ``workers`` threads (hashing and unpacking release the interpreter lock):
    this GPU's validation shards first, then the training shards most likely to be drawn.
    """

    sizes: dict[str, int] = {}
    for name, source in (("train", sampler), ("validation", validation)):
        if source is None:
            continue
        paths = {row["summary_path"] for row in source.rows}
        for path in paths - sizes.keys():
            sizes[path] = int(json.loads(Path(path).read_text(encoding="utf-8"))["container_bytes"])
        unpacked = sum(sizes[path] for path in paths) / 1e9
        say(f"{name}: {len(source.rows)} sides in {len(paths)} shards, {unpacked:.1f} GB unpacked")
    largest = max(sizes.values())
    count = min(len(sizes), max(1, int(cache_gb * 1e9 // largest)))
    say(f"keeping up to {count} of {len(sizes)} shards in memory (--cache-gb {cache_gb:g})")
    cache = sampler._shards
    for source in (sampler, validation):
        if source is not None:
            source.cached_shards = count
            source._shards = cache

    rank, world_size = _world()
    wanted = []
    if validation is not None:
        for batch in range(1, validation_batches + 1):
            wanted += [row["summary_path"] for row in validation.rows_for_update(batch, rank, world_size)]
    weights: Counter[str] = Counter()
    for row in sampler.rows:
        weights[row["summary_path"]] += row["weight"]
    wanted += [path for path, _weight in weights.most_common()]
    order = [path for path in dict.fromkeys(wanted) if path not in cache][: count - len(cache)]
    if not order:
        return
    started = time.perf_counter()
    report_every = max(1, len(order) // 10)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(ppo_expert_bc.load_il_cache_shard, path, verify=True): path for path in order}
        for done, future in enumerate(as_completed(futures), 1):
            cache[futures[future]] = future.result()
            if done % report_every == 0 or done == len(order):
                say(f"loaded {done}/{len(order)} shards in {time.perf_counter() - started:.0f}s")


def _describe(row: Mapping[str, object]) -> str:
    parts = [f"step {row['step']:>6}", f"{row['seconds']:>8.0f}s"]
    if "train_loss" in row:
        parts.append(f"train {row['train_loss']:.4f}")
    if "validation_total" in row:
        heads = " ".join(f"{head} {row[f'validation_{head}']:.4f}" for head in HEADS)
        parts.append(f"validation {row['validation_total']:.4f} ({heads})")
    rms = " ".join(f"{value:.4f}" for key, value in row.items() if key.startswith("rms/"))
    parts.append(f"rms {rms}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="PPO checkpoint to start from")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="human train.json (the run's --expert-bc-manifest)"
    )
    parser.add_argument(
        "--validation-manifest", type=Path, help="held-out human games; default: validation.json beside --manifest"
    )
    parser.add_argument("--output", type=Path, required=True, help="checkpoint to write; a .jsonl log goes beside it")
    parser.add_argument("--steps", type=int, default=3000, help="imitation steps (0: no training)")
    parser.add_argument("--records", type=int, default=8, help="human games per step on each GPU")
    parser.add_argument("--window-steps", type=int, default=128, help="supervised decisions per game window")
    parser.add_argument("--chunk-steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=8, help="validation size, in batches of --records")
    parser.add_argument("--reset-new-parameters", action="store_true", help="zero the new layers before training")
    parser.add_argument(
        "--cache-gb", type=float, default=16, help="memory per GPU process for unpacked shards, each then read once"
    )
    parser.add_argument("--load-workers", type=int, default=8, help="threads reading and unpacking shards")
    parser.add_argument("--progress-every", type=int, default=10, help="print a progress line every N steps")
    parser.add_argument("--seed", type=int, default=1, help="which training windows are drawn")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} already exists; pick another --output")
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must differ from the source checkpoint")
    validation_path = args.validation_manifest or args.manifest.parent / "validation.json"
    if args.validation_manifest is not None and not validation_path.is_file():
        parser.error(f"no validation manifest at {validation_path}")

    from .factory import build_production_model_v4

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if args.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        else:
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device(args.device)
    try:
        _run(args, validation_path, device, build_production_model_v4)
    finally:
        if distributed:
            dist.destroy_process_group()


def _run(args: argparse.Namespace, validation_path: Path, device: torch.device, build_model) -> None:
    rank, world_size = _world()
    lead = rank == 0

    def say(text: str) -> None:
        if lead:
            print(text, flush=True)

    torch.manual_seed(args.seed)
    model = build_model(device=device)
    # As in the PPO trainer, whose imitation step this reuses.
    model.set_ppo_location_head_fp32(True)
    # Carries the source's PPO optimizer state through unchanged, for --warm-start-optimizer.
    ppo_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-5, weight_decay=0.0)
    payload = load_actor_critic_checkpoint(args.checkpoint, model, optimizer=ppo_optimizer, map_location=device)
    ages = payload.get("parameter_ages")
    if args.reset_new_parameters:
        reset_new_parameters(model)
        ages = {name: 0 for name, _parameter in zero_start_parameters(model)}

    sampler = load_expert_sampler_v4(
        args.manifest, seed=args.seed, records_per_rank=args.records, window_steps=args.window_steps
    )
    validation = None
    if validation_path.is_file():
        validation = load_expert_sampler_v4(
            validation_path,
            seed=VALIDATION_SEED,
            records_per_rank=args.records,
            window_steps=args.window_steps,
            split="validation",
        )
    else:
        say(f"no validation manifest at {validation_path}; keeping the last step")
    if world_size > 1:
        say(f"{world_size} processes: {args.records} games per step on each, {args.records * world_size} in all")
    _share_shard_cache(
        sampler,
        validation,
        cache_gb=args.cache_gb,
        workers=args.load_workers,
        validation_batches=args.validation_batches,
        say=say,
    )

    log_path = args.output.with_suffix(".jsonl")
    if lead:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log_path.unlink(missing_ok=True)

    def log(row: Mapping[str, object]) -> None:
        if not lead:
            return
        print(_describe(row), flush=True)
        with log_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(row, sort_keys=True) + "\n")

    say(f"training {', '.join(name for name, _ in zero_start_parameters(model))}; everything else frozen")
    result = pretrain_new_parameters(
        model,
        sampler,
        steps=args.steps,
        learning_rate=args.learning_rate,
        chunk_steps=args.chunk_steps,
        max_grad_norm=args.max_grad_norm,
        validation=validation,
        validation_batches=args.validation_batches,
        eval_every=args.eval_every,
        log=log,
        progress=say,
        progress_every=args.progress_every,
    )
    if not lead:
        return
    rows = result["rows"]
    chosen = next(row for row in rows if row["step"] == result["chosen_step"])
    summary = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_id": payload["checkpoint_id"],
        "manifest": str(args.manifest.resolve()),
        "validation_manifest": None if validation is None else str(validation_path.resolve()),
        "reset_new_parameters": args.reset_new_parameters,
        "steps": args.steps,
        "chosen_step": result["chosen_step"],
        "records": args.records,
        "world_size": world_size,
        "window_steps": args.window_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "validation_loss_start": rows[0].get("validation_total"),
        "validation_loss_chosen": chosen.get("validation_total"),
        "new_parameter_rms": new_parameter_rms(model),
    }
    save_actor_critic_checkpoint(
        args.output,
        model,
        optimizer=ppo_optimizer,
        update_step=int(payload["update_step"]),
        training_stage=str(payload["training_stage"]),
        gamma_per_decision=float(payload["gamma_per_decision"]),
        gae_lambda=payload.get("gae_lambda"),
        extra={**dict(payload.get("extra") or {}), "new_parameter_pretraining": summary},
        parameter_ages=ages,
    )
    with log_path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps({"event": "saved", "path": str(args.output), **summary}, sort_keys=True) + "\n")
    print(
        f"saved {args.output}: kept step {result['chosen_step']}, validation loss "
        f"{summary['validation_loss_start']} -> {summary['validation_loss_chosen']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
