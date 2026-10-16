"""Where did training move the policy? Compare checkpoints with a baseline on human game states.

Every checkpoint and the baseline read the same recorded human games step by step, each keeping its own
recurrent state. At every decision the script measures, at one common sampling temperature:

- the exact divergence of the play/wait gate, and both models' play probability;
- for the later heads (card, placement, delay, second action) an unbiased sampled divergence: draw an
  action from the checkpoint, score it under both models, average the per-head log-ratio;
- what the sampled plays look like (cards, placement row, delay bin) for both models;
- how well each model still predicts the human's own action, head by head.

Large movement in a head that does not reach the game (the delay offset under --execution-delay-ticks)
or diffuse movement everywhere points at noise; movement concentrated in one card or placement points at
a learned habit. No engine is needed.
"""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from ...card_names import CARD_NAMES
from .cache import load_il_cache_shard
from .catalog import FIRST_REAL_CARD_VOCAB_ID
from .checkpoint import load_actor_critic_checkpoint
from .decoding import ShadowCandidateLegality
from .factory import build_production_model_v4
from .imitation import slice_il_sequence
from .ppo_expert_bc import load_expert_sampler_v4
from .tensors import GATE_ACT, GATE_WAIT, ActionSequenceV4

HEADS = ("candidate", "target", "delay", "continue")


def load_policy(path, device, temperatures):
    model = build_production_model_v4(device=device)
    load_actor_critic_checkpoint(path, model, map_location=device)
    model.eval()
    gate, action, continuation = temperatures
    model.set_ppo_gate_temperature(gate)
    model.set_ppo_action_temperature(action)
    model.set_ppo_continue_temperature(continuation)
    return model


def wait_actions(batch_size, device, width):
    return ActionSequenceV4(
        gate=torch.full((batch_size,), GATE_WAIT, dtype=torch.long, device=device),
        micro_action_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        candidate_index=torch.full((batch_size, width), -1, dtype=torch.long, device=device),
        candidate_uid=torch.full((batch_size, width), -1, dtype=torch.long, device=device),
        target_cell=torch.full((batch_size, width), -1, dtype=torch.long, device=device),
        delay_offset_bin=torch.full((batch_size, width), -1, dtype=torch.long, device=device),
    )


def with_slot_uids(model, batch, human):
    """Number candidates by slot. A UID only names a candidate and is not a model input, but recorded games can
    repeat one, and a repeated UID cannot be matched back to a single candidate when a play is scored.
    The human's own play keeps its candidate if its recorded UID was unique, otherwise it is dropped (None).
    """
    candidates = batch.candidates
    slots = torch.arange(1, candidates.uid.shape[1] + 1, device=candidates.uid.device).expand_as(candidates.uid)
    renumbered = replace(batch, candidates=replace(candidates, uid=slots.to(candidates.uid.dtype)))
    try:
        index = model._resolve_candidate_indices(batch, human)
    except ValueError:
        return renumbered, None
    uid = torch.where(index >= 0, index + 1, torch.full_like(index, -1)).to(human.candidate_uid.dtype)
    return renumbered, replace(human, candidate_uid=uid)


def head_log_probs(components):
    return {
        "gate": components.gate_log_prob,
        "candidate": components.candidate_log_prob.sum(-1),
        "target": components.target_log_prob.sum(-1),
        "delay": components.delay_offset_log_prob.sum(-1),
        "continue": components.continue_log_prob,
    }


def card_name(catalog, vocab_id):
    index = int(vocab_id) - FIRST_REAL_CARD_VOCAB_ID
    if not 0 <= index < len(catalog.raw_card_ids):
        return f"vocab {int(vocab_id)}"
    raw = catalog.raw_card_ids[index]
    names = CARD_NAMES.get(raw)
    return names[0] if names else str(raw)


class Tally:
    """Running sums for one model pair over every evaluated decision."""

    def __init__(self):
        self.steps = 0
        self.gate_kl = 0.0
        self.act_new = 0.0
        self.act_base = 0.0
        self.sampled_acts = 0
        self.head_kl = dict.fromkeys(HEADS, 0.0)
        self.cards = {"new": Counter(), "base": Counter()}
        self.rows = {"new": [], "base": []}
        self.delays = {"new": Counter(), "base": Counter()}
        self.human = {"new": Counter(), "base": Counter()}
        self.human_counts = Counter()
        self.human_dropped = 0
        self.skipped = 0


def describe_plays(output, batch, catalog, width, side, tally):
    actions = output.actions
    for row in torch.nonzero(actions.gate == GATE_ACT, as_tuple=False).flatten().tolist():
        for step in range(int(actions.micro_action_count[row])):
            index = int(actions.candidate_index[row, step])
            if int(batch.candidates.ability_vocab_id[row, index]) > 0:
                tally.cards[side]["(技能)"] += 1
            else:
                tally.cards[side][card_name(catalog, batch.candidates.visible_card_vocab_id[row, index])] += 1
            cell = int(actions.target_cell[row, step])
            if cell >= 0:
                tally.rows[side].append(cell // width)
            tally.delays[side][int(actions.delay_offset_bin[row, step])] += 1


def compare_sequence(base, models, sequence, tallies, *, temperatures, width, generator):
    gate_t, action_t, continue_t = temperatures
    device = next(base.parameters()).device
    sequence = sequence.to(device)
    states = [base.initial_state(1)] + [model.initial_state(1) for model in models]
    decode = {
        "gate_temperature": gate_t,
        "action_temperature": action_t,
        "continue_temperature": continue_t,
        "validate": False,
    }
    gate_mask = sequence.valid_mask if sequence.gate_loss_mask is None else sequence.gate_loss_mask
    for t, (batch, human) in enumerate(zip(sequence.observations, sequence.actions, strict=True)):
        start = sequence.episode_start[t]
        contexts = []
        for index, model in enumerate((base, *models)):
            context = model.forward(batch, states[index], episode_start=start, validate=False)
            states[index] = context.next_state
            contexts.append(context)
        if not bool(sequence.valid_mask[t, 0]):
            continue
        try:
            measure_step(
                base,
                models,
                contexts,
                tallies,
                batch,
                human,
                trusted_gate=bool(gate_mask[t, 0]),
                decode=decode,
                width=width,
                generator=generator,
            )
        except ValueError as error:
            # One odd recorded state must not cost the whole report; it is counted and named once.
            if not any(tally.skipped for tally in tallies):
                print(f"skipping a decision the models cannot score: {error}", file=sys.stderr, flush=True)
            for tally in tallies:
                tally.skipped += 1


def measure_step(base, models, contexts, tallies, batch, human, *, trusted_gate, decode, width, generator):
    """Compare every checkpoint with the baseline at one recorded decision."""
    device = next(base.parameters()).device
    batch, human = with_slot_uids(base, batch, human)
    waits = wait_actions(1, device, base.config.max_micro_actions)
    base_wait = base._decode(batch, contexts[0], sample=False, forced_actions=waits, **decode).log_prob
    base_act = 1.0 - float(base_wait.exp())
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
    base_sample = base._decode(batch, contexts[0], sample=True, forced_actions=None, **decode)
    human_base = (
        None
        if human is None
        else head_log_probs(
            base._decode(batch, contexts[0], sample=False, forced_actions=human, **decode).action_components
        )
    )
    human_act = human is not None and bool(human.gate[0] == GATE_ACT)
    # Short of elixir or with nothing to play, only the gate is a choice; the later heads are not asked.
    can_play = bool(ShadowCandidateLegality(batch.candidates, base.config).candidate_mask().any())
    for model, context, tally in zip(models, contexts[1:], tallies, strict=True):
        tally.steps += 1
        new_wait = model._decode(batch, context, sample=False, forced_actions=waits, **decode).log_prob
        new_act = 1.0 - float(new_wait.exp())
        p, q = min(max(new_act, 1e-9), 1 - 1e-9), min(max(base_act, 1e-9), 1 - 1e-9)
        tally.gate_kl += p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        tally.act_new += new_act
        tally.act_base += base_act
        # A play drawn from the checkpoint, its gate forced to ACT so every playable step informs the later heads.
        drawn = None
        if can_play:
            drawn = model._decode(
                batch,
                context,
                sample=True,
                forced_actions=None,
                preselected_gate=torch.full((1,), GATE_ACT, dtype=torch.long, device=device),
                **decode,
            )
        if drawn is not None and int(drawn.actions.micro_action_count[0]) > 0:
            scored_base = base._decode(batch, contexts[0], sample=False, forced_actions=drawn.actions, **decode)
            new_heads = head_log_probs(drawn.action_components)
            base_heads = head_log_probs(scored_base.action_components)
            tally.sampled_acts += 1
            for head in HEADS:
                tally.head_kl[head] += float(new_heads[head][0] - base_heads[head][0])
        sample = model._decode(batch, context, sample=True, forced_actions=None, **decode)
        describe_plays(sample, batch, base.catalog, width, "new", tally)
        describe_plays(base_sample, batch, base.catalog, width, "base", tally)
        if human is None:
            tally.human_dropped += 1
            continue
        human_new = head_log_probs(
            model._decode(batch, context, sample=False, forced_actions=human, **decode).action_components
        )
        for head in ("gate", *HEADS):
            if head == "gate" and not trusted_gate:
                continue
            if head != "gate" and not human_act:
                continue
            tally.human["new"][head] += float(human_new[head][0])
            tally.human["base"][head] += float(human_base[head][0])
            tally.human_counts[head] += 1


def report(label, tally):
    steps = max(tally.steps, 1)
    acts = max(tally.sampled_acts, 1)
    lines = [
        f"== {label}  ({tally.steps} decisions"
        + (f"; {tally.human_dropped} human plays with a repeated card UID left out" if tally.human_dropped else "")
        + (f"; {tally.skipped} decisions could not be scored" if tally.skipped else "")
        + ")"
    ]
    lines.append(
        f"   play probability per decision: baseline {tally.act_base / steps:.4f} -> {tally.act_new / steps:.4f}"
        f"   gate KL {tally.gate_kl / steps:.5f}"
    )
    lines.append(
        "   per-play KL (checkpoint || baseline): "
        + "  ".join(f"{head} {tally.head_kl[head] / acts:.4f}" for head in HEADS)
    )
    lines.append(
        "   human-action log-likelihood change (higher = closer to humans): "
        + "  ".join(
            f"{head} {(tally.human['new'][head] - tally.human['base'][head]) / max(tally.human_counts[head], 1):+.4f}"
            for head in ("gate", *HEADS)
        )
    )
    total = {side: max(sum(tally.cards[side].values()), 1) for side in ("new", "base")}
    names = sorted(
        set(tally.cards["new"]) | set(tally.cards["base"]),
        key=lambda name: -abs(tally.cards["new"][name] / total["new"] - tally.cards["base"][name] / total["base"]),
    )
    lines.append(
        "   card share, biggest changes: "
        + "  ".join(
            f"{name} {100 * tally.cards['base'][name] / total['base']:.1f}%->{100 * tally.cards['new'][name] / total['new']:.1f}%"
            for name in names[:8]
        )
    )
    rows = {side: (sum(values) / len(values) if values else float("nan")) for side, values in tally.rows.items()}
    lines.append(f"   mean placement row: baseline {rows['base']:.2f} -> {rows['new']:.2f}")
    delays = {side: max(sum(tally.delays[side].values()), 1) for side in ("new", "base")}
    bins = sorted(set(tally.delays["new"]) | set(tally.delays["base"]))
    lines.append(
        "   delay bins: "
        + "  ".join(
            f"{b}: {100 * tally.delays['base'][b] / delays['base']:.0f}%->{100 * tally.delays['new'][b] / delays['new']:.0f}%"
            for b in bins
        )
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="the run's --expert-bc-manifest (see runs/NAME/trainer-options.txt)",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--games", type=int, default=8, help="human games to replay")
    parser.add_argument("--max-steps", type=int, default=800, help="decisions per game from its start (800 = 200 s)")
    parser.add_argument("--gate-temperature", type=float, default=0.33)
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--continue-temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    temperatures = (args.gate_temperature, args.action_temperature, args.continue_temperature)
    torch.set_grad_enabled(False)
    base = load_policy(args.baseline, args.device, temperatures)
    models = [load_policy(path, args.device, temperatures) for path in args.checkpoints]
    sampler = load_expert_sampler_v4(args.manifest, seed=args.seed, records_per_rank=args.games)
    rows = sampler.rows_for_update(1, 0, 1)
    tallies = [Tally() for _ in models]
    generator = torch.Generator().manual_seed(args.seed)
    width = base.config.board_width
    shards = {}
    for number, row in enumerate(rows, 1):
        path = row["summary_path"]
        if path not in shards:
            shards.clear()
            shards[path] = load_il_cache_shard(path, verify=True)
        sequence = shards[path].sequence(int(row["sequence_index"]))
        sequence = slice_il_sequence(sequence, 0, min(sequence.time_steps, args.max_steps))
        compare_sequence(base, models, sequence, tallies, temperatures=temperatures, width=width, generator=generator)
        print(f"replayed game {number}/{len(rows)}", file=sys.stderr, flush=True)
    print(
        f"baseline {args.baseline}; temperatures gate {temperatures[0]}, action {temperatures[1]}, "
        f"continue {temperatures[2]}; {len(rows)} human games, first {args.max_steps} decisions each"
    )
    for path, tally in zip(args.checkpoints, tallies, strict=True):
        print(report(str(path), tally))
    print(
        json.dumps(
            {
                str(path): {
                    "gate_kl": t.gate_kl / max(t.steps, 1),
                    **{f"{h}_kl": t.head_kl[h] / max(t.sampled_acts, 1) for h in HEADS},
                }
                for path, t in zip(args.checkpoints, tallies, strict=True)
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
