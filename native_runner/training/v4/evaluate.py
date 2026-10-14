"""Evaluate two checkpoints in complete, sequential offline native matches."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ActionKind, ActionV1
from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT, NativeClashEnv
from ...match_factory import MatchConfig
from ...rich_telemetry_adapter import RuntimeEffectCatalog
from .async_cluster_self_play import (
    _first_successful_card_tick_v4,
    _frozen_opponent_opening_force_owner_v4,
    parse_play_gap_ticks,
    space_plays_v4,
)
from .cluster_self_play import build_resident_engine_v4
from .deck_pool import DeckPoolManifestV4
from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from .factory import production_semantic_bundle
from .policy_session import build_policy_session_v4, load_policy_v4
from .ppo_runtime import ppo_environment_config_v4
from .tensors import concatenate_padded_tensor_records, concatenate_tensor_records

# One training engine exposes eight resident slots, and one opening block is eight games.
RESIDENT_EVAL_LANES = 8

_DELAYABLE = frozenset({ActionKind.PLAY_CARD, ActionKind.ACTIVATE_ABILITY})


@contextmanager
def paired_cpu_rng(seed, side, tick):
    """Independent model/decision draws, shared across the paired conditions.

    Resetting per decision prevents an extra forced sample on B from shifting A's
    stream or either model's later draws. The CPU generator is always reset. When
    the model samples on CUDA, that generator is reset from the same key.
    """
    key = f"evaluation-{seed}-{side}-{tick}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(value).get_state())
        if cuda_devices:
            torch.cuda.manual_seed_all(value)
        yield


def sample_deck_pair(match, deck_pool, rng, *, battle_level=None):
    """Two decks drawn by sqrt(uses) from the pool, card order shuffled as in PPO, on ``match``'s settings."""

    decks = rng.choices(deck_pool.decks, weights=[math.sqrt(deck.uses) for deck in deck_pool.decks], k=2)
    layouts = []
    for deck in decks:
        order = list(range(8))
        rng.shuffle(order)
        layouts.append(
            (
                tuple(deck.card_ids[i] for i in order),
                tuple(deck.form_availability[i] for i in order),
                deck.tower_troop_id,
            )
        )
    levels = {}
    if battle_level is not None:
        levels = {"level_cap": battle_level, "minimum_card_level": battle_level, "king_tower_level": battle_level}
    paired = replace(
        match,
        deck0=layouts[0][0],
        deck1=layouts[1][0],
        deck0_form_availability=layouts[0][1],
        deck1_form_availability=layouts[1][1],
        tower_troop0_id=layouts[0][2],
        tower_troop1_id=layouts[1][2],
        **levels,
    )
    return paired, (decks[0].deck_id, decks[1].deck_id)


def _choose_commands(
    sessions,
    observations,
    *,
    a_owner,
    case,
    seed,
    game_index,
    actions,
    last_landing,
    first_play_tick,
    opening_requests,
    opening_overrides,
    delay_a_ticks,
    delay_b_ticks,
    min_play_gap_ticks,
):
    """One decision for both owners. Mutates the per-match counters in place."""

    commands = {}
    for owner, session in enumerate(sessions):
        observation = observations[owner]
        chosen = ()
        if observation.tick >= FIRST_POLICY_DECISION_TICK:
            side = "a" if owner == a_owner else "b"
            if case is None:
                decision = session.decide(observation)
            else:
                force_owner = _frozen_opponent_opening_force_owner_v4(
                    current_owners=(a_owner,),
                    active_actions=tuple(actions),
                    tick=int(observation.tick),
                    after_seconds=case.force_b_seconds,
                )
                force = owner == force_owner
                opening_requests[owner] += int(force)
                with paired_cpu_rng(seed, side, observation.tick):
                    decision = session.decide(observation, force_opening=force)
                opening_overrides[owner] += int(decision.forced_opening)
            chosen = decision.decoded.actions
            delay = delay_a_ticks if owner == a_owner else delay_b_ticks
            if delay:
                # The model's plays land this many ticks after the decision, as taps do on a
                # live client; BattleEnvV1 keeps the slot pending and its elixir reserved.
                chosen = tuple(
                    replace(action, execute_offset_ticks=delay) if action.kind in _DELAYABLE else action
                    for action in chosen
                )
            chosen, last_landing[owner] = space_plays_v4(
                tuple(chosen),
                min_play_gap_ticks,
                (
                    f"{game_index}-{owner}-{observation.tick}"
                    if case is None
                    else f"{seed}-{side}-{observation.tick}"
                ),
                decision_tick=int(observation.tick),
                last_landing_tick=last_landing[owner],
            )
            actions[owner] += sum(action.kind != ActionKind.WAIT for action in chosen)
            if first_play_tick[owner] is None and any(action.kind != ActionKind.WAIT for action in chosen):
                first_play_tick[owner] = int(observation.tick)
        else:
            session.tensorizer.tensorize(observation, validate=False)
        commands[owner] = tuple(chosen) or (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS),)
    return commands


def _module_device(model):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _move_module(model, device):
    current = _module_device(model)
    if current is None or current == torch.device(device):
        return
    model.to(device)


def _forward_index_group(entries, indexes, device):
    sessions = [entries[index][0] for index in indexes]
    model = sessions[0].model
    _move_module(model, device)
    combined = concatenate_padded_tensor_records(tuple(entries[index][1] for index in indexes))
    combined_input = combined.to_model_input(device)
    state = concatenate_tensor_records(tuple(session.state for session in sessions)).to(device)
    episode_start = torch.tensor(
        [bool(entries[index][2]) for index in indexes], dtype=torch.bool, device=device
    )
    context = model.forward(combined_input, state, episode_start=episode_start, validate=False)
    return [
        (combined_input.narrow_batch(local, 1), context.narrow_batch(local, 1))
        for local in range(len(indexes))
    ]


@torch.inference_mode()
def forward_resident_contexts(entries):
    """One forward per model for every ready decision.

    ``entries`` are ``(session, host batch, first decision)``. Only one model
    occupies the GPU at a time. A chunk that does not fit falls back to CPU,
    and that model stays on CPU so sampling uses the same device.

    Disable autograd for the entire batch, including recurrent-state assembly.
    eval() alone would retain a growing graph through each session's LSTM state.
    """

    results = [None] * len(entries)
    grouped: dict[int, list[int]] = {}
    for index, (session, _host, _first) in enumerate(entries):
        grouped.setdefault(id(session.model), []).append(index)
    parked = []
    for indexes in grouped.values():
        sessions = [entries[index][0] for index in indexes]
        model = sessions[0].model
        for other in parked:
            if other is not model:
                _move_module(other, "cpu")
        parked[:] = [model]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        preferred = torch.device(sessions[0].device)
        chunk = min(4, len(indexes))
        while True:
            try:
                produced = []
                for start in range(0, len(indexes), chunk):
                    produced.extend(_forward_index_group(entries, indexes[start : start + chunk], preferred))
                break
            except torch.cuda.OutOfMemoryError as oom:
                produced = []
                del oom
                torch.cuda.empty_cache()
                if preferred.type == "cuda":
                    preferred = torch.device("cpu")
                    for session in sessions:
                        session.device = preferred
                    _move_module(sessions[0].model, preferred)
                    torch.cuda.empty_cache()
                    chunk = min(4, len(indexes))
                    continue
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
        for index, row in zip(indexes, produced, strict=True):
            results[index] = row
    return results


def _commands_from_contexts(
    lane,
    contexts,
    *,
    delay_a_ticks,
    delay_b_ticks,
    min_play_gap_ticks,
):
    """Turn one lane's precomputed contexts into engine commands.

    ``contexts`` maps owner to ``(host batch, context, force)`` for owners
    whose tick is ready. Earlier ticks only refresh the tensorizer.
    """

    commands = {}
    for owner, session in enumerate(lane.sessions):
        observation = lane.observations[owner]
        chosen = ()
        if observation.tick >= FIRST_POLICY_DECISION_TICK:
            host_batch, context, force = contexts[owner]
            side = "a" if owner == lane.a_owner else "b"
            with paired_cpu_rng(lane.seed, side, observation.tick):
                decision = session.finish_decision(
                    observation, host_batch, context, force_opening=force
                )
            lane.opening_overrides[owner] += int(decision.forced_opening)
            chosen = decision.decoded.actions
            delay = delay_a_ticks if owner == lane.a_owner else delay_b_ticks
            if delay:
                chosen = tuple(
                    replace(action, execute_offset_ticks=delay) if action.kind in _DELAYABLE else action
                    for action in chosen
                )
            chosen, lane.last_landing[owner] = space_plays_v4(
                tuple(chosen),
                min_play_gap_ticks,
                f"{lane.seed}-{side}-{observation.tick}",
                decision_tick=int(observation.tick),
                last_landing_tick=lane.last_landing[owner],
            )
            lane.actions[owner] += sum(action.kind != ActionKind.WAIT for action in chosen)
            if lane.first_play_tick[owner] is None and any(action.kind != ActionKind.WAIT for action in chosen):
                lane.first_play_tick[owner] = int(observation.tick)
        else:
            session.tensorizer.tensorize(observation, validate=False)
        commands[owner] = tuple(chosen) or (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS),)
    return commands


def _absorb_step(observations, first_card_tick, rejected):
    for observation in observations.values():
        for owner in (0, 1):
            first_card_tick[owner] = _first_successful_card_tick_v4(
                observation.events, owner=owner, current=first_card_tick[owner]
            )
        for event in observation.events:
            if event.event_type == "action_rejected":
                rejected.add(json.dumps(event.to_dict(), sort_keys=True))


def _match_row(
    *,
    game_index,
    seed,
    a_owner,
    pair_ids,
    observations,
    terminated,
    truncated,
    actions,
    first_play_tick,
    first_card_tick,
    rejected,
    case,
    opening_requests,
    opening_overrides,
):
    terminal = observations[0].terminal
    winner = None if terminal.winner is None else ("a" if terminal.winner == a_owner else "b")
    row = {
        "game": game_index + 1,
        "seed": seed,
        "a_owner": a_owner,
        "winner": winner,
        "terminated": terminated,
        "truncated": truncated,
        "tick": observations[0].tick,
        "terminal": terminal.to_dict(),
        "actions_by_owner": list(actions),
        "deck_ids": None if pair_ids is None else {"a": pair_ids[a_owner], "b": pair_ids[1 - a_owner]},
        # Decision tick of each side's first card or ability (it lands after any delay).
        "first_play_seconds": {
            side: None if first_play_tick[owner] is None else first_play_tick[owner] / 20
            for side, owner in (("a", a_owner), ("b", 1 - a_owner))
        },
        "rejected_actions": len(rejected),
        "first_card_executed_seconds": {
            side: None if first_card_tick[owner] is None else first_card_tick[owner] / 20
            for side, owner in (("a", a_owner), ("b", 1 - a_owner))
        },
    }
    if case is not None:
        row.update(case.metadata())
        row["match"] = asdict(case.match)
        for key, values in (("opening_requests", opening_requests), ("opening_overrides", opening_overrides)):
            row[key] = {"a": values[a_owner], "b": values[1 - a_owner]}
    return row


def _summary_payload(games, rows):
    result = {
        "requested": games,
        "completed": len(rows),
        "a_wins": sum(row["winner"] == "a" for row in rows),
        "b_wins": sum(row["winner"] == "b" for row in rows),
        "draws": sum(row["terminated"] and row["winner"] is None for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
    }
    for side in ("a", "b"):
        firsts = sorted(
            row["first_play_seconds"][side] for row in rows if row["first_play_seconds"][side] is not None
        )
        if firsts:
            result[f"{side}_first_play_seconds_10_50_90"] = [
                firsts[min(len(firsts) - 1, int(quantile * len(firsts)))] for quantile in (0.1, 0.5, 0.9)
            ]
    return result


_RESET_OPTIONS = {
    "render_mode": "headless",
    "decision_ticks": POLICY_DECISION_TICKS,
    "event_driven_decisions": False,
    "allow_initial_remaining_runtime_rejections": True,
}


@dataclass
class _ResidentLane:
    index: int
    env_id: int
    environment: object
    case: object
    seed: int
    a_owner: int
    deck_ids: object
    sessions: list
    observations: dict
    actions: list
    last_landing: list
    first_play_tick: list
    first_card_tick: list
    opening_requests: list
    opening_overrides: list
    rejected: set
    active: bool = True
    terminated: bool = False
    truncated: bool = False


def _share_identical_policies(policies):
    """Use one copy when both sides are the same checkpoint, so one forward covers every row."""

    first, second = policies
    try:
        same = (
            first.checkpoint_sha256 == second.checkpoint_sha256
            and first.model.ppo_gate_temperature == second.model.ppo_gate_temperature
            and first.model.ppo_action_temperature == second.model.ppo_action_temperature
            and first.model.ppo_continue_temperature == second.model.ppo_continue_temperature
        )
    except AttributeError:
        return policies
    if not same:
        return policies
    return (first, first)


def _evaluate_resident(
    policies,
    *,
    schedule,
    games,
    output,
    host,
    port,
    device,
    sample,
    delay_a_ticks,
    delay_b_ticks,
    min_play_gap_ticks,
):
    """Play a paired schedule on one engine's resident slots.

    A block is eight games and a training engine has eight slots, so each wave
    locksteps one block. Capture is the training compact format. Decisions,
    forced openings, delays and the paired RNG stay on this side of the socket.
    """

    policies = _share_identical_policies(policies)
    engine = build_resident_engine_v4(
        0,
        environment_config=ppo_environment_config_v4(),
        host=host,
        base_port=port,
        lanes=RESIDENT_EVAL_LANES,
        capture_mode="compact",
    )
    rows = []
    open_sessions = []
    try:
        for native in engine.natives:
            native.public_card_play_events_from_combat_ring = True
        engine.prepare()
        with (output / "matches.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            with ThreadPoolExecutor(max_workers=RESIDENT_EVAL_LANES, thread_name_prefix="eval-lane") as executor:
                for wave_start in range(0, len(schedule), RESIDENT_EVAL_LANES):
                    wave = tuple(schedule[wave_start : wave_start + RESIDENT_EVAL_LANES])
                    engine.coordinator.close_session()
                    lanes = []
                    for offset, case in enumerate(wave):
                        environment = engine.environments[offset]
                        if torch.device(device).type == "cpu":
                            torch.set_rng_state(
                                torch.Generator(device="cpu").manual_seed(case.match.seed).get_state()
                            )
                        observations, _info = environment.reset(match_config=case.match, options=_RESET_OPTIONS)
                        elixir = {
                            owner: next(player.elixir_exact for player in obs.players if player.owner == owner)
                            for owner, obs in observations.items()
                        }
                        sessions = [
                            build_policy_session_v4(
                                environment,
                                policies[0 if owner == case.a_owner else 1].model,
                                actor_owner=owner,
                                device=device,
                                sample=sample,
                            )
                            for owner in (0, 1)
                        ]
                        for owner, session in enumerate(sessions):
                            session.start_episode(observations[owner], initial_elixir=elixir)
                        open_sessions.extend(sessions)
                        lanes.append(
                            _ResidentLane(
                                index=wave_start + offset,
                                env_id=int(getattr(environment.native, "env_id", offset)),
                                environment=environment,
                                case=case,
                                seed=case.match.seed,
                                a_owner=case.a_owner,
                                deck_ids=case.deck_ids,
                                sessions=sessions,
                                observations=observations,
                                actions=[0, 0],
                                last_landing=[None, None],
                                first_play_tick=[None, None],
                                first_card_tick=[None, None],
                                opening_requests=[0, 0],
                                opening_overrides=[0, 0],
                                rejected=set(),
                            )
                        )
                    engine.coordinator.start_session()
                    while any(lane.active for lane in lanes):
                        entries = []
                        owners = []
                        for lane in lanes:
                            if not lane.active:
                                continue
                            for owner, session in enumerate(lane.sessions):
                                observation = lane.observations[owner]
                                if observation.tick < FIRST_POLICY_DECISION_TICK:
                                    continue
                                force_owner = _frozen_opponent_opening_force_owner_v4(
                                    current_owners=(lane.a_owner,),
                                    active_actions=tuple(lane.actions),
                                    tick=int(observation.tick),
                                    after_seconds=lane.case.force_b_seconds,
                                )
                                force = owner == force_owner
                                lane.opening_requests[owner] += int(force)
                                host_batch, first = session.prepare_host_batch(observation)
                                owners.append((lane, owner, force))
                                entries.append((session, host_batch, first))
                        forwarded = forward_resident_contexts(entries) if entries else []
                        by_lane = {}
                        for (lane, owner, force), forwarded_row in zip(owners, forwarded, strict=True):
                            host_batch, context = forwarded_row
                            by_lane.setdefault(lane.env_id, {})[owner] = (host_batch, context, force)
                        pending = []
                        for lane in lanes:
                            if not lane.active:
                                continue
                            pending.append((
                                lane,
                                _commands_from_contexts(
                                    lane,
                                    by_lane.get(lane.env_id, {}),
                                    delay_a_ticks=delay_a_ticks,
                                    delay_b_ticks=delay_b_ticks,
                                    min_play_gap_ticks=min_play_gap_ticks,
                                ),
                            ))
                        engine.coordinator.begin_round(tuple(lane.env_id for lane, _commands in pending))
                        futures = {
                            executor.submit(
                                lane.environment.step,
                                commands,
                                record_trace=False,
                                policy_features_only=True,
                            ): lane
                            for lane, commands in pending
                        }
                        stepped = []
                        try:
                            for future in as_completed(futures):
                                stepped.append((futures[future], future.result()))
                        except BaseException as error:
                            engine.coordinator.abort_round(error)
                            for future in futures:
                                future.cancel()
                            raise
                        else:
                            engine.coordinator.finish_round()
                        for lane, (observations, _rewards, terms, truncs, _info) in stepped:
                            lane.observations = observations
                            _absorb_step(observations, lane.first_card_tick, lane.rejected)
                            lane.terminated = bool(terms["__all__"])
                            lane.truncated = bool(truncs["__all__"])
                            if not (lane.terminated or lane.truncated):
                                continue
                            lane.active = False
                            row = _match_row(
                                game_index=lane.index,
                                seed=lane.seed,
                                a_owner=lane.a_owner,
                                pair_ids=lane.deck_ids,
                                observations=observations,
                                terminated=lane.terminated,
                                truncated=lane.truncated,
                                actions=lane.actions,
                                first_play_tick=lane.first_play_tick,
                                first_card_tick=lane.first_card_tick,
                                rejected=lane.rejected,
                                case=lane.case,
                                opening_requests=lane.opening_requests,
                                opening_overrides=lane.opening_overrides,
                            )
                            rows.append(row)
                            log.write(json.dumps(row) + "\n")
                            print(json.dumps(row), flush=True)
                            for session in lane.sessions:
                                session.end_episode()
                            for session in lane.sessions:
                                open_sessions.remove(session)
                    engine.coordinator.close_session()
        result = _summary_payload(games, rows)
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for session in open_sessions:
            session.end_episode()
        engine.close()


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
    deck_pool=None,
    battle_level=None,
    schedule=None,
):
    if games < 1:
        raise ValueError("games must be positive")
    if not (0 <= delay_a_ticks <= 0xFFFF and 0 <= delay_b_ticks <= 0xFFFF):
        raise ValueError("delay ticks must be in [0, 65535]")
    if schedule is not None and len(schedule) != games:
        raise ValueError("a paired schedule requires exactly games cases")
    if schedule is not None and torch.device(device).type not in ("cpu", "cuda"):
        raise ValueError("a paired schedule runs the model on cpu or cuda; sampling stays on the CPU generator")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if torch.device(device).type == "cuda" and torch.device(device).index is not None:
        torch.cuda.set_device(torch.device(device))
    # Scheduled duels load on CPU and move one model onto the GPU at a time.
    # Two fp32 copies already fill a 22 GiB card before any activation is allocated.
    load_device = "cpu" if schedule is not None and torch.device(device).type == "cuda" else device
    policies = [load_policy_v4(path, device=load_device) for path in (checkpoint_a, checkpoint_b)]
    # Checkpoints carry the gate temperature of the update that saved them; --sample uses it unless overridden.
    saved_gate_temperatures = [policy.model.ppo_gate_temperature for policy in policies]
    for policy, temperature in zip(policies, (gate_temperature_a, gate_temperature_b), strict=True):
        if temperature is not None:
            policy.model.set_ppo_gate_temperature(float(temperature))
    identity = {
        "games": games,
        "sample": sample,
        "delay_a_ticks": delay_a_ticks,
        "delay_b_ticks": delay_b_ticks,
        "min_play_gap_ticks": min_play_gap_ticks,
        "saved_gate_temperatures": saved_gate_temperatures,
        "gate_temperatures": [policy.model.ppo_gate_temperature for policy in policies],
        "match": asdict(match),
        "deck_pool": None if deck_pool is None else {"decks": len(deck_pool.decks), "schema": deck_pool.schema},
        "battle_level": battle_level,
        "schedule": None if schedule is None else [asdict(case) for case in schedule],
        "transport": "resident-compact" if schedule is not None else "json",
        "checkpoints": [
            {"path": str(p.checkpoint_path), "sha256": p.checkpoint_sha256, "checkpoint_id": p.checkpoint_id}
            for p in policies
        ],
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    if schedule is not None:
        policies = _share_identical_policies(policies)
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        return _evaluate_resident(
            policies,
            schedule=schedule,
            games=games,
            output=output,
            host=host,
            port=port,
            device=device,
            sample=sample,
            delay_a_ticks=delay_a_ticks,
            delay_b_ticks=delay_b_ticks,
            min_play_gap_ticks=min_play_gap_ticks,
        )
    bundle = production_semantic_bundle()
    native = NativeClashEnv(host, port, timeout=30.0)
    native.public_card_play_events_from_combat_ring = True
    environment = BattleEnvV1(
        native=native,
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
    )
    rows = []
    sessions = []
    try:
        with (output / "matches.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            # Without a deck pool, A keeps deck0 and only the side swaps. With one, every pair of games draws two
            # decks and the second game hands each model the other deck (owner 1 plays deck1), so both models
            # play both decks.
            deck_rng = random.Random(match.seed)
            pair, pair_ids = match, None
            for index in range(games):
                case = None if schedule is None else schedule[index]
                if case is not None:
                    current, a_owner, pair_ids = case.match, case.a_owner, case.deck_ids
                else:
                    a_owner = index % 2
                    if deck_pool is not None and a_owner == 0:
                        pair, pair_ids = sample_deck_pair(match, deck_pool, deck_rng, battle_level=battle_level)
                    current = replace(pair, seed=match.seed + index)
                    if a_owner and deck_pool is None:
                        current = replace(
                            current,
                            deck0=pair.deck1,
                            deck1=pair.deck0,
                            deck0_form_availability=pair.deck1_form_availability,
                            deck1_form_availability=pair.deck0_form_availability,
                            tower_troop0_id=pair.tower_troop1_id,
                            tower_troop1_id=pair.tower_troop0_id,
                        )
                if torch.device(device).type == "cpu":
                    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(current.seed).get_state())
                else:
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
                first_card_tick = [None, None]
                opening_requests = [0, 0]
                opening_overrides = [0, 0]
                rejected = set()
                terminated = truncated = False
                while not terminated and not truncated:
                    commands = _choose_commands(
                        sessions,
                        observations,
                        a_owner=a_owner,
                        case=case,
                        seed=current.seed,
                        game_index=index,
                        actions=actions,
                        last_landing=last_landing,
                        first_play_tick=first_play_tick,
                        opening_requests=opening_requests,
                        opening_overrides=opening_overrides,
                        delay_a_ticks=delay_a_ticks,
                        delay_b_ticks=delay_b_ticks,
                        min_play_gap_ticks=min_play_gap_ticks,
                    )
                    observations, _, terms, truncs, _ = environment.step(commands, record_trace=False)
                    terminated, truncated = bool(terms["__all__"]), bool(truncs["__all__"])
                    _absorb_step(observations, first_card_tick, rejected)
                row = _match_row(
                    game_index=index,
                    seed=current.seed,
                    a_owner=a_owner,
                    pair_ids=pair_ids,
                    observations=observations,
                    terminated=terminated,
                    truncated=truncated,
                    actions=actions,
                    first_play_tick=first_play_tick,
                    first_card_tick=first_card_tick,
                    rejected=rejected,
                    case=case,
                    opening_requests=opening_requests,
                    opening_overrides=opening_overrides,
                )
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                for session in sessions:
                    session.end_episode()
                sessions = []
        result = _summary_payload(games, rows)
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
    parser.add_argument(
        "--deck-pool",
        type=Path,
        help="deck-pool manifest (as for PPO): each pair of games draws two decks by sqrt(uses), one per side, "
        "then swaps them; without it every game uses the match config's decks",
    )
    parser.add_argument("--battle-level", type=int, choices=(11, 16), help="card and King Tower level")
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
        deck_pool=None if args.deck_pool is None else DeckPoolManifestV4.load(args.deck_pool),
        battle_level=args.battle_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
