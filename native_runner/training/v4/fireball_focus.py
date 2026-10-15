"""Find trusted human Fireballs near a visible moving enemy; never rewrite targets.

The geometric filter identifies useful contexts, not successful hits or a
counterfactual best aim point. It uses only the stored FAIR policy input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path


FIREBALL_ID = 28000000
FOCUS_SCHEMA = "v4-fireball-moving-context.v1"


def validate_focus(config):
    if config.get("schema") != FOCUS_SCHEMA:
        raise ValueError("unsupported expert focus")
    for key in ("min_speed_tiles_per_second", "lookahead_seconds", "radius_tiles"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid fireball focus parameter: {key}")
    percent = config.get("percent")
    if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not 0 < percent <= 80:
        raise ValueError("focus percent must be in (0, 80], retaining general examples")


def focus_frames(shard, index, config):
    import torch

    validate_focus(config)
    descriptor = shard.descriptors[index]
    start, stop = descriptor.offset, descriptor.offset + descriptor.length
    actions, obs = shard.actions, shard.observations
    counts = actions.micro_action_count[start:stop]
    indices = actions.candidate_index[start:stop].clamp_min(0)
    ids = obs.candidates.native_visible_card_id[start:stop].gather(1, indices)
    micro = torch.arange(indices.shape[1])[None] < counts[:, None]
    trusted = shard.valid_mask[start:stop, 0] & shard.gate_loss_mask[start:stop, 0]
    selected = (ids == FIREBALL_ID) & micro & trusted[:, None]
    groups = obs.groups
    rows = []
    for local in torch.nonzero(selected.any(dim=1), as_tuple=False).flatten().tolist():
        t = start + local
        if not groups.mask.shape[1] or not groups.child_mask.shape[1]:
            continue
        parents = groups.child_group_index[t]
        safe_parents = parents.clamp(0, groups.mask.shape[1] - 1)
        # child kinds: troop=0, hero=2, character=6; no buildings/projectiles.
        kind = groups.child_type[t]
        usable = (groups.child_mask[t] & (parents >= 0) & (parents < groups.mask.shape[1])
                  & groups.mask[t, safe_parents] & (groups.owner_type[t, safe_parents] == 1)
                  & ((kind == 0) | (kind == 2) | (kind == 6)))
        # Tensorizer stores child velocity in tiles/s divided by 10.
        velocity = groups.child_features[t, :, 9:11].float() * 10.0
        speed2 = velocity.square().sum(dim=1)
        usable &= speed2 >= config["min_speed_tiles_per_second"] ** 2
        if not bool(usable.any()):
            continue
        position = groups.child_position[t].float()
        for j in torch.nonzero(selected[local], as_tuple=False).flatten().tolist():
            cell = int(actions.target_cell[t, j])
            if not 0 <= cell < 18 * 32:
                continue
            target = position.new_tensor([cell % 18 + 0.5, cell // 18 + 0.5])
            delta = target - position
            seconds = (delta * velocity).sum(dim=1) / speed2.clamp_min(1e-8)
            closest = position + seconds.clamp(0, config["lookahead_seconds"])[:, None] * velocity
            near = (closest - target).square().sum(dim=1) <= config["radius_tiles"] ** 2
            if bool((usable & near).any()):
                rows.append(local)
                break
    return rows


def _initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _annotate_shard(task):
    from .cache import load_il_cache_shard
    from .ppo_expert_bc import cached_general_sequence_quality

    path, rows, config = task
    shard = load_il_cache_shard(path, verify=True)
    result = []
    for row in rows:
        n = row["sequence_index"]
        descriptor = shard.descriptors[n]
        if (descriptor.replay_tag, descriptor.owner, descriptor.length) != (
            row["replay_tag"], row["owner"], row["sequence_length"]
        ) or row["cache_compressed_sha256"] != shard.summary["compressed_sha256"]:
            raise ValueError("expert cache identity changed after audit")
        quality = cached_general_sequence_quality(shard, n)
        if any(row.get(key) != value for key, value in quality.items()):
            raise ValueError("expert cached labels changed after audit")
        result.append(dict(row, fireball_focus_frames=focus_frames(shard, n, config)))
    return result


def build_focus_manifest(manifest, output, *, percent=50, workers=1):
    from collections import defaultdict

    source = Path(manifest).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"focus manifest already exists: {output}")
    payload = json.loads(source.read_bytes())
    if (payload.get("schema") != "v4-general-expert-selection.v1" or payload.get("split") not in {"train", "validation"}
            or payload.get("cache_labels_verified") is not True or payload.get("discard_tail_ticks") != 200):
        raise ValueError("focus requires an audited general expert train/validation manifest")
    if workers < 1:
        raise ValueError("workers must be positive")
    config = dict(schema=FOCUS_SCHEMA, percent=percent, min_speed_tiles_per_second=0.25,
                  lookahead_seconds=3.0, radius_tiles=3.0,
                  meaning="visible moving enemy near the aim point or its straight-line path; hit success not assumed")
    validate_focus(config)
    grouped = defaultdict(list)
    for row in payload["rows"]:
        if row.get("source_split") != payload["split"]:
            raise ValueError("expert split annotation mismatch")
        grouped[row["summary_path"]].append(row)
    tasks = [(path, rows, config) for path, rows in sorted(grouped.items())]
    rows = []
    if workers == 1:
        for n, task in enumerate(tasks, 1):
            rows.extend(_annotate_shard(task))
            print(json.dumps(dict(event="fireball_focus_progress", shards=n, total_shards=len(tasks))), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks))),
                                 mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_worker) as executor:
            for n, part in enumerate(executor.map(_annotate_shard, tasks), 1):
                rows.extend(part)
                print(json.dumps(dict(event="fireball_focus_progress", shards=n, total_shards=len(tasks))), flush=True)
    rows.sort(key=lambda row: (row["replay_tag"], row["owner"]))
    focused = sum(bool(row["fireball_focus_frames"]) for row in rows)
    if not focused:
        raise ValueError("no trusted moving-enemy Fireball contexts found; no manifest written")
    payload.update(rows=rows, focus=config, focus_source=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                   focus_summary=dict(all_sides=len(rows), focused_sides=focused,
                                      focus_frames=sum(len(row["fireball_focus_frames"]) for row in rows)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload["focus_summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-percent", type=float, default=50)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    summary = build_focus_manifest(args.manifest, args.output, percent=args.focus_percent, workers=args.workers)
    print(json.dumps(dict(event="fireball_focus_complete", **summary)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
