"""Audit selected human replay caches and split whole games for PPO imitation.

Input selection.jsonl is produced by select_expert_replays. Raw replay files
are not training tensors: run cache_builder first. Output paths refer to the
cache on this host; regenerate the manifests after moving a cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "v4-general-expert-selection.v1"


def selected_sides(path):
    rows = {}
    tags = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tag = item["replay_tag"]
        if not isinstance(tag, str) or not tag or tag in tags:
            raise ValueError("invalid or duplicate selected replay tag")
        tags.add(tag)
        if not item["learn"]:
            raise ValueError(f"selected replay has no learned sides: {tag}")
        for side in item["learn"]:
            owner = side["owner"]
            weight = side["weight"]
            if type(owner) is not int or owner not in (0, 1):
                raise ValueError(f"invalid native owner: {tag}")
            if side["outcome"] not in ("win", "close_loss") or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"invalid selected side quality: {tag}")
            key = (tag, owner)
            if key in rows:
                raise ValueError(f"duplicate selected owner: {key}")
            rows[key] = dict(
                replay_tag=tag, owner=owner, outcome=side["outcome"], weight=weight,
                source_side=side.get("side"), scoreline=item.get("scoreline"),
                winner_deck=item.get("deck"), elixir_leaked=side.get("elixir_leaked"),
            )
    if not rows:
        raise ValueError("empty source selection")
    return rows


def split_games(rows, *, validation_percent=10, seed=20261015):
    if not math.isfinite(validation_percent) or not 0 < validation_percent < 100:
        raise ValueError("validation percent must be between 0 and 100")
    tags = sorted({row["replay_tag"] for row in rows})
    if len(tags) < 2:
        raise ValueError("need at least two audited games for disjoint train/validation sets")
    ordered = sorted(tags, key=lambda tag: (hashlib.sha256(f"{seed}:{tag}".encode()).digest(), tag))
    count = min(len(tags) - 1, max(1, round(len(tags) * validation_percent / 100)))
    heldout = set(ordered[:count])
    return {
        name: [dict(row, source_split=name) for row in rows if (row["replay_tag"] in heldout) == is_validation]
        for name, is_validation in (("train", False), ("validation", True))
    }


def cache_status(result, *, allow_trusted_prefix=False):
    if result.get("completed") is True:
        return "completed"
    reason = str(result.get("failure_reason") or "")
    if (
        allow_trusted_prefix
        and int(result.get("decision_frame_count") or 0) > 0
        and int(result.get("simulated_end_tick") or 0) >= 800
        and type(result.get("first_untrusted_tick")) is int
        and (reason == "battle ended before all expert actions executed"
             or reason.startswith("terminal fidelity mismatch:"))
    ):
        return "trusted_prefix"
    return "excluded"


def cache_indexes(roots):
    paths = set()
    for root in roots:
        root = Path(root).resolve()
        if root.is_file():
            found = [root]
        elif (root / "index.json").is_file():
            found = [root / "index.json"]
        else:
            found = sorted(root.rglob("index.json"))
        if not found:
            raise ValueError(f"no cache index found under {root}")
        paths.update(found)
    return sorted(paths)


def audit_selection(selection, roots, *, allow_trusted_prefix=False):
    # Keep imports lazy so --help and source/split planning need no torch.
    from .cache import load_il_cache_index, load_il_cache_manifest, load_il_cache_shard
    from .ppo_expert_bc import cached_general_sequence_quality

    wanted = selected_sides(selection)
    summaries = set()
    for index in cache_indexes(roots):
        for row in load_il_cache_index(index)["shards"]:
            summaries.add((index.parent / row["summary_file"]).resolve())
    seen, accepted, exclusions = set(), [], []
    for i, path in enumerate(sorted(summaries)):
        manifest = load_il_cache_manifest(path)
        matches = [
            (n, seq) for n, seq in enumerate(manifest["sequences"])
            if (seq["replay_tag"], seq["owner"]) in wanted
        ]
        if not matches:
            continue
        results = {r["replay_tag"]: r for r in manifest["results"]}
        shard = None
        for n, seq in matches:
            key = (seq["replay_tag"], seq["owner"])
            if key in seen:
                raise ValueError(f"ambiguous duplicate cache owner: {key}; supply only the intended cache roots")
            seen.add(key)
            result = results[seq["replay_tag"]]
            status = cache_status(result, allow_trusted_prefix=allow_trusted_prefix)
            if status == "excluded":
                exclusions.append(dict(replay_tag=key[0], owner=key[1], reason=result.get("failure_reason")
                                       or "cache not completed"))
                continue
            if shard is None:
                shard = load_il_cache_shard(path, verify=True)
            quality = cached_general_sequence_quality(shard, n)
            if (
                quality["cached_micro_actions"] <= 0 or quality["cached_gate_frames"] <= 0
                or not quality["cached_episode_start"] or not quality["cached_natural_time"]
            ):
                exclusions.append(dict(replay_tag=key[0], owner=key[1], reason="no trusted actions/gates or episode start"))
                continue
            accepted.append(dict(
                wanted[key], **quality, summary_path=str(path), sequence_index=n,
                sequence_length=seq["length"], cache_status=status,
                cache_compressed_sha256=shard.summary["compressed_sha256"],
                first_untrusted_tick=result.get("first_untrusted_tick"),
                simulated_end_tick=result.get("simulated_end_tick"),
            ))
        # Release this shard before inflating the next one.
        shard = None
        print(json.dumps(dict(event="expert_cache_audit", shard=i + 1, total_shards=len(summaries),
                              accepted_sides=len(accepted), excluded_sides=len(exclusions))), flush=True)
    for tag, owner in sorted(set(wanted) - seen):
        exclusions.append(dict(replay_tag=tag, owner=owner, reason="selected owner missing from cache"))
    return sorted(accepted, key=lambda row: (row["replay_tag"], row["owner"])), exclusions


def build_manifests(selection, roots, output, *, validation_percent=10, seed=20261015, allow_trusted_prefix=False):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"manifest output must be empty: {output}")
    rows, exclusions = audit_selection(selection, roots, allow_trusted_prefix=allow_trusted_prefix)
    splits = split_games(rows, validation_percent=validation_percent, seed=seed)
    provenance = dict(
        selection_path=str(Path(selection).resolve()),
        selection_sha256=hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
        split_seed=seed, validation_percent=validation_percent,
        split_unit="whole replay tag; both owners always stay together",
        allow_trusted_prefix=allow_trusted_prefix,
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(accepted_sides=len(rows), accepted_games=len({r["replay_tag"] for r in rows}),
                   excluded_sides=len(exclusions), exclusions_by_reason=dict(Counter(r["reason"] for r in exclusions)))
    for name, selected in splits.items():
        stats = dict(games=len({r["replay_tag"] for r in selected}), sides=len(selected),
                     outcomes=dict(Counter(r["outcome"] for r in selected)),
                     cache_status=dict(Counter(r["cache_status"] for r in selected)))
        summary[name] = stats
        payload = dict(schema=SCHEMA, split=name, rows=selected, summary=stats, provenance=provenance,
                       discard_tail_ticks=200, cache_labels_verified=True)
        (output / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                                           encoding="utf-8")
    (output / "excluded.json").write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="selected dataset's selection.jsonl")
    parser.add_argument("--cache-root", type=Path, action="append", required=True,
                        help="cache index, cache directory or parent of partition directories; repeat if needed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-percent", type=float, default=10)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--allow-trusted-prefix", action="store_true",
                        help="also use masked safe prefixes of narrowly classified reconstruction failures")
    args = parser.parse_args(argv)
    summary = build_manifests(args.selection, args.cache_root, args.output, validation_percent=args.validation_percent,
                             seed=args.seed, allow_trusted_prefix=args.allow_trusted_prefix)
    print(json.dumps(dict(event="expert_manifest_complete", output=str(args.output), **summary), ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
