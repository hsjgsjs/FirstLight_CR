"""Audit live recording latency without connecting to the game or changing policy weights.

python -m native_runner.analyze_live_latency RECORDING --checkpoint CHECKPOINT --compare OLD_RECORDING
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p / 100
    low, high = math.floor(index), math.ceil(index)
    return round(values[low] + (values[high] - values[low]) * (index - low), 3)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return dict(n=len(values), min=round(min(values), 3), median=percentile(values, 50),
                mean=round(statistics.mean(values), 3), p90=percentile(values, 90),
                p95=percentile(values, 95), max=round(max(values), 3))


def blob(root, row):
    with gzip.open(root / row["file"], "rt", encoding="utf-8") as stream:
        return json.load(stream)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    events = rows(root / "events.jsonl")
    index = rows(root / "index.jsonl")
    captures = {r["number"]: r for r in index if r["kind"] == "capture"}
    frames = {r.get("next_decision"): r for r in index if r["kind"] == "policy_frame" and "next_decision" in r}
    decisions = {r["decision_id"]: r for r in events if r["event"] == "decision"}
    sent = {}
    last_decision = None
    for event in events:
        if event["event"] == "decision":
            last_decision = event["decision_id"]
        elif event["event"] == "tap":
            sent.setdefault(last_decision, []).append(event)
    matches = []
    for event in events:
        if event["event"] != "deployment_spawn_latency":
            continue
        decision = decisions[event["decision"]]
        taps = [t for t in sent.get(event["decision"], []) if t["card"] == event["cardId"]]
        if len(taps) != 1:
            continue
        tap = taps[0]
        capture = captures[decision["capture_id"]]
        frame = frames.get(event["decision"])
        item = dict(decision_id=event["decision"], card_id=event["cardId"],
                    tick=event["decisionTick"], spawn_tick=event["spawnTick"],
                    spawn_kind=event["spawnKind"], deployment_id=event["deploymentId"],
                    total_ms=event["latencyMs"], observation_to_tap_ms=(tap["tick"]-event["decisionTick"])*50,
                    input_game_ms=(tap["after_tick"]-tap["tick"])*50,
                    input_wall_ms=tap["input_ms"],
                    tap_end_to_card_play_ms=(event["cardPlayTick"]-tap["after_tick"])*50,
                    card_play_to_spawn_ms=(event["spawnTick"]-event["cardPlayTick"])*50,
                    inference_ms=decision["inference_ms"],
                    native_capture_ms=(decision.get("capture_us") or 0)/1000,
                    capture_roundtrip_ms=(capture["response_received_ns"]-capture["request_started_ns"])/1e6,
                    capture_return_to_decision_log_ms=(decision["monotonic_ns"]-capture["response_received_ns"])/1e6,
                    decision_log_to_tap_log_ms=(tap["monotonic_ns"]-decision["monotonic_ns"])/1e6,
                    expected_execution_error_ms=(event["spawnTick"]-tap["execute_tick_estimate"])*50,
                    native_object_id=event["nativeObjectId"],
                    target_world=tap["world"], generation=manifest["identity"].get("generation"))
        if frame:
            observation = blob(root, frame)
            item["frame_keys"] = sorted(observation)
            item["decision_time_ms"] = observation.get("time", {})
            item["entity_count"] = len(observation.get("entities", []))
            item["own_elixir"] = next((p.get("elixir_exact") for p in observation.get("players", [])
                                       if p.get("owner") == manifest["identity"]["owner"]), None)
            item["pending_actions"] = observation.get("pending_actions", observation.get("pending_commands"))
        matches.append(item)
    all_decisions = list(decisions.values())
    capture_order = [r for r in index if r["kind"] == "capture"]
    gaps = [(b["tick"]-a["tick"])*50 for a,b in zip(capture_order,capture_order[1:])
            if a.get("tick") is not None and b.get("tick") is not None and b["tick"] >= a["tick"]]
    native = {e["nativeEventId"]: e for e in events if e["event"] == "native_deployment_timing"}
    requested = [e for e in events if e["event"] == "deployment_timing_requested"]
    unmatched = [e for e in events if e["event"] == "deployment_timing_unmatched"]
    unresolved = [e for e in events if e["event"] == "deployment_timing_unresolved"]
    per_card = {}
    for card in sorted({m["card_id"] for m in matches}):
        selected = [m for m in matches if m["card_id"] == card]
        per_card[str(card)] = {"total_ms": stats([m["total_ms"] for m in selected]),
                               "tap_end_to_card_play_ms": stats([m["tap_end_to_card_play_ms"] for m in selected])}
    capture_times = [(c["response_received_ns"]-c["request_started_ns"])/1e6 for c in capture_order]
    inference = [d["inference_ms"] for d in all_decisions]
    capture_to_log = [(d["monotonic_ns"]-captures[d["capture_id"]]["response_received_ns"])/1e6 for d in all_decisions]
    first_frame = blob(root, next(r for r in index if r["kind"] == "policy_frame"))
    sample_entity = next(iter(first_frame.get("entities", [])), {})
    frame_cache = {}
    def observation_at(row):
        if row["number"] not in frame_cache:
            frame_cache[row["number"]] = blob(root, row)
        return frame_cache[row["number"]]

    owner = manifest["identity"]["owner"]
    pending_after_spawn = []
    for match in matches:
        decision = decisions[match["decision_id"]]
        action_ids = {a["action_id"] for a in decision["actions"] if a.get("card_id") == match["card_id"]}
        relevant = [r for r in index if r["kind"] == "policy_frame"
                    and match["spawn_tick"] <= r["tick"] <= match["spawn_tick"] + 15]
        exposures = []
        for row in relevant:
            frame = observation_at(row)
            for pending in frame.get("pending_actions", []):
                if pending.get("action", {}).get("action_id") in action_ids:
                    exposures.append({"frame_tick": row["tick"],
                                      "after_actual_spawn_ms": (row["tick"]-match["spawn_tick"])*50,
                                      "expected_execution_tick": pending.get("expected_execution_tick"),
                                      "reserved_elixir": frame.get("action_mask", {}).get("reasons", {}).get("reserved_elixir")})
        if exposures:
            pending_after_spawn.append(dict(decision_id=match["decision_id"], card_id=match["card_id"],
                                            spawn_tick=match["spawn_tick"], frames=exposures))

    field_examples = []
    fireballs = []
    for match in matches:
        frame_row = frames.get(match["decision_id"])
        if not frame_row:
            continue
        frame = observation_at(frame_row)
        enemy_entities = [e for e in frame.get("entities", []) if e.get("owner") == 1-owner]
        if len(field_examples) < 2 and frame.get("entities"):
            field_examples.append(frame["entities"][0])
        if match["card_id"] == 28000000:
            fireballs.append({"decision_id": match["decision_id"], "tick": match["tick"],
                             "projectile_spawn_tick": match["spawn_tick"], "delay_ms": match["total_ms"],
                             "target_world": match["target_world"], "enemy_entities": enemy_entities})
    activation_evidence = []
    for match in matches:
        if match["spawn_kind"] != "spawn":
            continue
        native_id = match["native_object_id"]
        native_rows = [r for r in capture_order if match["spawn_tick"] <= r["tick"] <= match["spawn_tick"]+45]
        first_entity = None
        first_active = None
        for row in native_rows:
            capture = blob(root, row)
            entity = next((e for e in capture.get("rich", {}).get("objects", []) if e.get("nativeObjectId") == native_id), None)
            if not entity:
                continue
            phase = entity.get("phaseRuntime") or {}
            if first_entity is None:
                first_entity = {"tick": row["tick"], "phaseRuntime": phase}
            remaining = phase.get("deployRemainingMs")
            if remaining == 0:
                first_active = row["tick"]
                break
        activation_evidence.append({"decision_id": match["decision_id"], "card_id": match["card_id"],
                                    "spawn_tick": match["spawn_tick"], "first_entity": first_entity,
                                    "first_observed_zero_deploy_tick": first_active,
                                    "decision_to_observed_zero_deploy_ms": (first_active-match["tick"])*50 if first_active is not None else None})

    native_card_plays = [e for e in native.values() if e["kind"] == "card_play"]
    card_play_latencies = []
    for native_event in native_card_plays:
        deployment = next((m for m in matches if m["deployment_id"] == native_event["deploymentId"]), None)
        if deployment:
            card_play_latencies.append((native_event["tick"]-deployment["tick"])*50)
            continue
        unresolved_match = next((e for e in unresolved if e.get("deploymentId") == native_event["deploymentId"]), None)
        if unresolved_match:
            card_play_latencies.append((native_event["tick"]-unresolved_match["decisionTick"])*50)
    report = dict(directory=str(root), identity=manifest["identity"], summary=manifest["summary"],
                  start_utc=manifest["started_utc"], end_utc=manifest.get("ended_utc"),
                  counts=dict(events=Counter(e["event"] for e in events), requested=len(requested),
                              native_card_play=sum(e["kind"] == "card_play" for e in native.values()),
                              matched=len(matches), unmatched_card_play=len(unmatched), unresolved=len(unresolved)),
                  stage_stats={key: stats([m[key] for m in matches]) for key in (
                      "total_ms", "observation_to_tap_ms", "input_game_ms", "input_wall_ms",
                      "tap_end_to_card_play_ms", "card_play_to_spawn_ms", "inference_ms",
                      "capture_roundtrip_ms", "capture_return_to_decision_log_ms", "decision_log_to_tap_log_ms",
                      "expected_execution_error_ms")},
                  all_decision_stats=dict(inference_ms=stats(inference),
                                          inference_after_first_ms=stats(inference[1:]),
                                          capture_roundtrip_ms=stats(capture_times), capture_gap_ms=stats(gaps),
                                          capture_return_to_decision_log_ms=stats(capture_to_log),
                                          native_capture_ms=stats([(d.get("capture_us") or 0)/1000 for d in all_decisions if d.get("capture_us") is not None])),
                  per_card=per_card, matches=matches, unresolved=unresolved, unmatched=unmatched,
                  native_card_play_latency_ms=stats(card_play_latencies),
                  tap_input_wall_ms=stats([e["input_ms"] for e in events if e["event"] == "tap"]),
                  pending_after_actual_spawn=pending_after_spawn, entity_examples=field_examples,
                  fireball_contexts=fireballs,
                  gap_waits=[e for e in events if e["event"] == "play_gap_wait"],
                  lost=[e for e in events if e["event"] == "tap_lost"],
                  first_frame_schema={"keys": sorted(first_frame), "entity_keys": sorted(sample_entity),
                                      "players": first_frame.get("players"), "time": first_frame.get("time")})
    return report


def checkpoint_info(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {k: payload.get(k) for k in ("schema", "checkpoint_id", "update_step", "training_stage",
                                       "ppo_gate_temperature", "ppo_action_temperature", "ppo_continue_temperature")} | {
        "extra_keys": sorted(payload.get("extra", {})),
        "resume_contract": payload.get("extra", {}).get("resume_contract"),
        "model_config": payload.get("contract", {}).get("model_config")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"current": audit(args.recording)}
    if args.compare:
        report["previous"] = audit(args.compare)
    if args.checkpoint:
        report["checkpoint"] = checkpoint_info(args.checkpoint)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), "utf-8")
    for name, audit_report in report.items():
        if name == "checkpoint":
            print(name, json.dumps(audit_report, ensure_ascii=False, default=str))
        else:
            summary = {k: audit_report[k] for k in ("counts", "stage_stats", "all_decision_stats", "native_card_play_latency_ms")}
            summary["pending_after_actual_spawn_count"] = len(audit_report["pending_after_actual_spawn"])
            summary["pending_after_actual_spawn_example"] = audit_report["pending_after_actual_spawn"][:1]
            summary["entity_example"] = audit_report["entity_examples"][:1]
            print(name, json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
