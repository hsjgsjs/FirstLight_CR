"""Prepare a pinned baseline and matched PPO-only/PPO+human trial options.

Does not start training. Both new runs warm-start the same weights and Adam
state. Their temperature schedules continue from the source checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil


def continuation_options(text, payload):
    contract = payload.get("extra", {}).get("resume_contract", {})
    if "gate_temperature_anneal_updates" not in contract:
        raise ValueError("source must be a strict PPO checkpoint with a saved temperature schedule")
    temperature = float(payload["ppo_gate_temperature"])
    end = float(contract["gate_temperature_end"])
    remaining = max(1, int(contract["gate_temperature_anneal_updates"]) - int(payload["update_step"]))
    if not 0 < end <= temperature:
        raise ValueError("invalid source temperature schedule")
    removed = {"--gate-temperature-start", "--gate-temperature-end", "--gate-temperature-anneal-updates"}
    tokens, kept = shlex.split(text), []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        flag, separator, _ = token.partition("=")
        if flag in removed or flag.startswith("--expert-bc-"):
            if not separator:
                if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                    raise ValueError(f"missing option value: {flag}")
                i += 1
        else:
            kept.append(token)
        i += 1
    if "--warm-start-optimizer" not in kept:
        kept.append("--warm-start-optimizer")
    kept.extend([
        "--gate-temperature-start", str(temperature), "--gate-temperature-end", str(end),
        "--gate-temperature-anneal-updates", str(remaining),
    ])
    # lab_ppo's launcher deliberately splits options on whitespace.
    if any(any(c.isspace() for c in token) for token in kept):
        raise ValueError("lab trainer option paths must not contain whitespace; use a symlink")
    return kept


def prepare_experiment(checkpoint, trainer_options, manifest, output, *, coefficient=0.1, every_updates=4):
    import torch
    from .checkpoint import CHECKPOINT_SCHEMA_V4

    output = Path(output).resolve()
    manifest = Path(manifest).resolve()
    if any(c.isspace() for c in str(manifest)) or any(c.isspace() for c in str(output)):
        raise ValueError("lab experiment paths must not contain whitespace; use a symlink")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"experiment output must be empty: {output}")
    from math import isfinite
    if not isfinite(coefficient) or coefficient <= 0 or every_updates <= 0:
        raise ValueError("BC coefficient and update interval must be positive")
    selection = json.loads(manifest.read_bytes())
    if selection.get("schema") != "v4-general-expert-selection.v1" or selection.get("split") != "train":
        raise ValueError("experiment requires a general expert train manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA_V4 or payload.get("optimizer_state_dict") is None:
        raise ValueError("experiment requires a strict PPO checkpoint with optimizer state")
    tokens = continuation_options(Path(trainer_options).read_text(), payload)
    human_tokens = tokens + [
        "--expert-bc-manifest", str(manifest), "--expert-bc-coefficient", str(coefficient),
        "--expert-bc-every-updates", str(every_updates),
        "--expert-bc-records-per-rank", "2", "--expert-bc-window-steps", "128",
    ]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, output / "baseline.pt")
    for name, options in (("ppo-options.txt", tokens), ("human-options.txt", human_tokens)):
        (output / name).write_text(" ".join(options) + "\n", encoding="utf-8")
    plan = dict(
        schema="v4-human-bc-experiment.v1", source_checkpoint=str(Path(checkpoint).resolve()),
        source_checkpoint_id=payload["checkpoint_id"], source_update=payload["update_step"],
        baseline=str(output / "baseline.pt"), expert_manifest=str(manifest),
        warm_start_optimizer=True, starting_gate_temperature=payload["ppo_gate_temperature"],
        bc_coefficient=coefficient, bc_every_updates=every_updates,
    )
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer-options", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficient", type=float, default=0.1)
    parser.add_argument("--every-updates", type=int, default=4)
    args = parser.parse_args(argv)
    plan = prepare_experiment(args.checkpoint, args.trainer_options, args.expert_manifest, args.output,
                              coefficient=args.coefficient, every_updates=args.every_updates)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
