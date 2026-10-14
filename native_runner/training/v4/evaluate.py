"""Evaluate two checkpoints in complete, sequential offline native matches."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ActionKind, ActionV1
from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT, NativeClashEnv
from ...match_factory import MatchConfig
from ...rich_telemetry_adapter import RuntimeEffectCatalog
from .async_cluster_self_play import parse_play_gap_ticks, space_plays_v4
from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from .factory import production_semantic_bundle
from .policy_session import build_policy_session_v4, load_policy_v4

_DELAYABLE = frozenset({ActionKind.PLAY_CARD, ActionKind.ACTIVATE_ABILITY})


def evaluate(
    checkpoint_a,
    checkpoint_b,
    *,
    games,
    match,
    output,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    device="cpu",
    sample=False,
    delay_a_ticks=0,
    delay_b_ticks=0,
    min_play_gap_ticks=None,
    gate_temperature_a=None,
    gate_temperature_b=None,
):
    if games < 1:
        raise ValueError("games must be positive")
    if not (0 <= delay_a_ticks <= 0xFFFF and 0 <= delay_b_ticks <= 0xFFFF):
        raise ValueError("delay ticks must be in [0, 65535]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if torch.device(device).type == "cuda" and torch.device(device).index is not None:
        torch.cuda.set_device(torch.device(device))
    policies = [load_policy_v4(path, device=device) for path in (checkpoint_a, checkpoint_b)]
    # Checkpoints carry the gate temperature of the update that saved them; --sample uses it unless overridden.
    saved_gate_temperatures = [policy.model.ppo_gate_temperature for policy in policies]
    for policy, temperature in zip(policies, (gate_temperature_a, gate_temperature_b), strict=True):
        if temperature is not None:
            policy.model.set_ppo_gate_temperature(float(temperature))
    bundle = production_semantic_bundle()
    native = NativeClashEnv(host, port, timeout=30.0)
    native.public_card_play_events_from_combat_ring = True
    environment = BattleEnvV1(
        native=native,
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
    )
    identity = {
        "games": games,
        "sample": sample,
        "delay_a_ticks": delay_a_ticks,
        "delay_b_ticks": delay_b_ticks,
        "min_play_gap_ticks": min_play_gap_ticks,
        "saved_gate_temperatures": saved_gate_temperatures,
        "gate_temperatures": [policy.model.ppo_gate_temperature for policy in policies],
        "match": asdict(match),
        "checkpoints": [
            {"path": str(p.checkpoint_path), "sha256": p.checkpoint_sha256, "checkpoint_id": p.checkpoint_id}
            for p in policies
        ],
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    rows = []
    sessions = []
    try:
        with (output / "matches.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for index in range(games):
                a_owner = index % 2
                current = replace(match, seed=match.seed + index)
                if a_owner:
                    current = replace(
                        current,
                        deck0=match.deck1,
                        deck1=match.deck0,
                        deck0_form_availability=match.deck1_form_availability,
                        deck1_form_availability=match.deck0_form_availability,
                        tower_troop0_id=match.tower_troop1_id,
                        tower_troop1_id=match.tower_troop0_id,
                    )
                torch.manual_seed(current.seed)
                observations, _ = environment.reset(
                    match_config=current,
                    options={
                        "render_mode": "headless",
                        "decision_ticks": POLICY_DECISION_TICKS,
                        "event_driven_decisions": False,
                        "allow_initial_remaining_runtime_rejections": True,
                    },
                )
                initial_elixir = {
                    owner: next(p.elixir_exact for p in obs.players if p.owner == owner)
                    for owner, obs in observations.items()
                }
                sessions = [
                    build_policy_session_v4(
                        environment,
                        policies[0 if owner == a_owner else 1].model,
                        actor_owner=owner,
                        device=device,
                        sample=sample,
                    )
                    for owner in (0, 1)
                ]
                for owner, session in enumerate(sessions):
                    session.start_episode(observations[owner], initial_elixir=initial_elixir)
                actions = [0, 0]
                last_landing = [None, None]
                first_play_tick = [None, None]
                rejected = set()
                terminated = truncated = False
                while not terminated and not truncated:
                    commands = {}
                    for owner, session in enumerate(sessions):
                        observation = observations[owner]
                        chosen = ()
                        if observation.tick >= FIRST_POLICY_DECISION_TICK:
                            chosen = session.decide(observation).decoded.actions
                            delay = delay_a_ticks if owner == a_owner else delay_b_ticks
                            if delay:
                                # The model's plays land this many ticks after the decision, as taps do on a
                                # live client; BattleEnvV1 keeps the slot pending and its elixir reserved.
                                chosen = tuple(
                                    replace(a, execute_offset_ticks=delay) if a.kind in _DELAYABLE else a
                                    for a in chosen
                                )
                            chosen, last_landing[owner] = space_plays_v4(
                                tuple(chosen),
                                min_play_gap_ticks,
                                f"{index}-{owner}-{observation.tick}",
                                decision_tick=int(observation.tick),
                                last_landing_tick=last_landing[owner],
                            )
                            actions[owner] += sum(a.kind != ActionKind.WAIT for a in chosen)
                            if first_play_tick[owner] is None and any(a.kind != ActionKind.WAIT for a in chosen):
                                first_play_tick[owner] = int(observation.tick)
                        else:
                            session.tensorizer.tensorize(observation, validate=False)
                        commands[owner] = tuple(chosen) or (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS),)
                    observations, _, terms, truncs, _ = environment.step(commands, record_trace=False)
                    terminated, truncated = bool(terms["__all__"]), bool(truncs["__all__"])
                    for observation in observations.values():
                        for event in observation.events:
                            if event.event_type == "action_rejected":
                                rejected.add(json.dumps(event.to_dict(), sort_keys=True))
                terminal = observations[0].terminal
                winner = None if terminal.winner is None else ("a" if terminal.winner == a_owner else "b")
                row = {
                    "game": index + 1,
                    "seed": current.seed,
                    "a_owner": a_owner,
                    "winner": winner,
                    "terminated": terminated,
                    "truncated": truncated,
                    "tick": observations[0].tick,
                    "terminal": terminal.to_dict(),
                    "actions_by_owner": actions,
                    # Decision tick of each side's first card or ability (it lands after any delay).
                    "first_play_seconds": {
                        side: None if first_play_tick[owner] is None else first_play_tick[owner] / 20
                        for side, owner in (("a", a_owner), ("b", 1 - a_owner))
                    },
                    "rejected_actions": len(rejected),
                }
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                for session in sessions:
                    session.end_episode()
                sessions = []
        result = {
            "requested": games,
            "completed": len(rows),
            "a_wins": sum(r["winner"] == "a" for r in rows),
            "b_wins": sum(r["winner"] == "b" for r in rows),
            "draws": sum(r["terminated"] and r["winner"] is None for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
        }
        for side in ("a", "b"):
            firsts = sorted(r["first_play_seconds"][side] for r in rows if r["first_play_seconds"][side] is not None)
            if firsts:
                result[f"{side}_first_play_seconds_10_50_90"] = [
                    firsts[min(len(firsts) - 1, int(q * len(firsts)))] for q in (0.1, 0.5, 0.9)
                ]
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for session in sessions:
            session.end_episode()
        native.close_transport()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--match-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--delay-a-ticks",
        type=int,
        default=0,
        help="execute model A's card plays and abilities this many ticks after its decision (a live tap lands ~21)",
    )
    parser.add_argument(
        "--delay-b-ticks",
        type=int,
        default=0,
        help="execute model B's card plays and abilities this many ticks after its decision (a live tap lands ~21)",
    )
    parser.add_argument(
        "--min-play-gap-ticks",
        type=parse_play_gap_ticks,
        help="land each side's consecutive plays and abilities at least MIN-MAX ticks apart (4-8 = 200-400 ms)",
    )
    parser.add_argument(
        "--gate-temperature-a",
        type=float,
        help="with --sample, model A's act/wait temperature (default: the one saved in its checkpoint)",
    )
    parser.add_argument("--gate-temperature-b", type=float, help="the same for model B")
    args = parser.parse_args(argv)
    config = json.loads(args.match_config.read_text(encoding="utf-8")) if args.match_config else {}
    for key in ("owner0_account_id", "owner1_account_id"):
        config.pop(key, None)
    config["seed"] = args.seed
    torch.set_num_threads(1)
    evaluate(
        args.checkpoint_a,
        args.checkpoint_b,
        games=args.games,
        match=MatchConfig(**config),
        output=args.output,
        host=args.host,
        port=args.port,
        device=args.device,
        sample=args.sample,
        delay_a_ticks=args.delay_a_ticks,
        delay_b_ticks=args.delay_b_ticks,
        min_play_gap_ticks=args.min_play_gap_ticks,
        gate_temperature_a=args.gate_temperature_a,
        gate_temperature_b=args.gate_temperature_b,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
