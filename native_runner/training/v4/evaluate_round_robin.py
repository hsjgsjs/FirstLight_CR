"""Run all assigned model pairs on one persistent external engine process."""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

from ...round_robin import base_match, digest, read_rows, write_json
from .evaluation_protocol import complete_natural_blocks, natural_schedule


def progress(shard, pair, stage):
    # The launcher can read this while the worker is running.
    temporary = shard / "progress.tmp"
    write_json(temporary, dict(pair=" vs ".join(pair), stage=stage))
    temporary.replace(shard / "progress.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from ...cr_native_env import NativeClashEnv
    from .deck_pool import DeckPoolManifestV4
    from .evaluate import evaluate

    options = json.loads((args.run / "settings.json").read_text(encoding="utf-8"))
    study = json.loads((args.run / "study.json").read_text(encoding="utf-8"))
    match = base_match(options)
    pool = DeckPoolManifestV4.load(options["deck_pool"])
    if digest(Path(options["deck_pool"])) != study["deck_pool_sha256"]:
        raise ValueError("deck pool changed after planning")
    for name, path in options["models"].items():
        if digest(Path(path)) != study["models"][name]:
            raise ValueError(f"checkpoint changed after planning: {name}")
    device = torch.device(options["device"])
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("round-robin device must be available cpu or cuda")
    if args.prepare_only:
        print(f"Validated {len(options['models'])} model files and {len(pool.decks)} decks.")
        return 0
    if args.shard is None:
        parser.error("--shard is required")
    torch.set_num_threads(options["cpu_threads"])
    work = json.loads((args.shard / "work.json").read_text(encoding="utf-8"))
    native = NativeClashEnv(args.host, args.port, timeout=30)
    try:
        attestation = native.attest().to_dict()
        if not attestation["production_ready"]:
            raise ValueError("engine is not production-ready")
        status = native._request("multi-status")
        if int(status.get("capacity", 0)) < options["lanes"]:
            raise ValueError(f"engine capacity is below {options['lanes']} slots")
        expected_path = args.run / "expected-engine.json"
        if expected_path.is_file() and json.loads(expected_path.read_text(encoding="utf-8")) != attestation:
            raise ValueError("engine identity differs from resumed history")
        write_json(args.shard / "engine.json", attestation)
    finally:
        native.close_transport()
    successful = True
    for index, item in enumerate(work):
        pair, blocks = item["models"], item["blocks"]
        destination = args.shard / f"pair-{index:02d}"
        destination.mkdir()
        write_json(destination / "pair.json", item)
        schedule = natural_schedule(match, blocks, deck_pool=pool)
        write_json(destination / "plan.json", [asdict(case) for case in schedule])
        progress(args.shard, pair, f"运行中：{len(blocks)} 组、{len(schedule)} 局")
        print(f"{pair[0]} vs {pair[1]}：{len(schedule)} 局，组号 {blocks}", flush=True)
        started = time.perf_counter()
        evaluate(options["models"][pair[0]], options["models"][pair[1]],
                 games=len(schedule), match=match, output=destination,
                 host=args.host, port=args.port, device=options["device"], sample=options["sample"],
                 delay_a_ticks=tuple(options["delay_ticks"]), delay_b_ticks=tuple(options["delay_ticks"]),
                 min_play_gap_ticks=tuple(options["min_play_gap_ticks"]),
                 gate_temperature_a=options["gate_temperature"], gate_temperature_b=options["gate_temperature"],
                 card_tile_temperature_a=options.get("card_tile_temperature"),
                 card_tile_temperature_b=options.get("card_tile_temperature"),
                 schedule=schedule, resident_lanes=options["lanes"],
                 trace_dir=destination / "traces" if options.get("trace") else None)
        elapsed = time.perf_counter() - started
        loaded = json.loads((destination / "identity.json").read_text(encoding="utf-8"))
        if [entry["sha256"] for entry in loaded["checkpoints"]] != [study["models"][name] for name in pair]:
            raise ValueError("loaded checkpoint hashes differ from study identity")
        completed = complete_natural_blocks(read_rows(destination / "matches.jsonl"))
        successful = successful and set(completed) == set(blocks)
        write_json(destination / "status.json", dict(complete_blocks=sorted(completed), elapsed_seconds=elapsed))
        progress(args.shard, pair, f"完成 {len(completed)}/{len(blocks)} 组，{elapsed:.0f} 秒")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
