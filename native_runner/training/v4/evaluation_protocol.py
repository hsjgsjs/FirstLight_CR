"""CPU evaluation schedule and paired statistics; no inference or engine dependencies."""

from dataclasses import dataclass, replace
import itertools
import math
import random
import statistics

from ...match_factory import MatchConfig


@dataclass(frozen=True)
class EvaluationCase:
    block: int
    variant: int
    condition: str
    match: MatchConfig
    a_owner: int
    a_deck: int
    deadline_seconds: float
    deck_ids: tuple[str, str] | None = None

    @property
    def force_b_seconds(self):
        return self.deadline_seconds if self.condition == "forced" else None

    def metadata(self):
        return {
            "block": self.block,
            "variant": self.variant,
            "condition": self.condition,
            "a_deck": self.a_deck,
            "opening_deadline_seconds": self.deadline_seconds,
        }


def swap_decks(match):
    """Move the complete deck, including form slots and Tower Troop, across owners."""
    return replace(
        match,
        deck0=match.deck1, deck1=match.deck0,
        deck0_form_availability=match.deck1_form_availability,
        deck1_form_availability=match.deck0_form_availability,
        tower_troop0_id=match.tower_troop1_id, tower_troop1_id=match.tower_troop0_id,
    )


def balanced_cases(base, *, block, deadline_seconds, deck_ids=None):
    """Eight games: both model sides x both deck assignments x both conditions.

    All eight share the engine seed and original card order. Conditions alternate
    which runs first, and deadlines remain attached to model B after a side swap.
    """
    cases = []
    for variant, (a_owner, a_deck) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        swapped = a_owner != a_deck
        match = swap_decks(base) if swapped else base
        ids = None if deck_ids is None else tuple(deck_ids[::-1] if swapped else deck_ids)
        conditions = ("natural", "forced") if (block + variant) % 2 == 0 else ("forced", "natural")
        for condition in conditions:
            cases.append(EvaluationCase(block, variant, condition, match, a_owner, a_deck, deadline_seconds, ids))
    return cases


def natural_cases(base, *, block, deck_ids=None):
    """Four natural games: both model sides x both complete deck assignments."""
    return [case for case in balanced_cases(base, block=block, deadline_seconds=0, deck_ids=deck_ids)
            if case.condition == "natural"]


def natural_schedule(match, blocks, *, deck_pool=None):
    """Random access by global block ID; independent of pair, shard and execution order."""
    cases = []
    for block in blocks:
        if isinstance(block, bool) or not isinstance(block, int) or block < 0:
            raise ValueError("block IDs must be nonnegative integers")
        base = replace(match, seed=match.seed + block)
        ids = None
        if deck_pool is not None:
            rng = random.Random(f"round-robin-decks-v1-{match.seed}-{block}")
            decks = rng.choices(deck_pool.decks, weights=[math.sqrt(deck.uses) for deck in deck_pool.decks], k=2)
            layouts = []
            for deck in decks:
                order = list(range(8))
                rng.shuffle(order)
                layouts.append((tuple(deck.card_ids[i] for i in order),
                                tuple(deck.form_availability[i] for i in order), deck.tower_troop_id))
            base = replace(base, deck0=layouts[0][0], deck1=layouts[1][0],
                           deck0_form_availability=layouts[0][1], deck1_form_availability=layouts[1][1],
                           tower_troop0_id=layouts[0][2], tower_troop1_id=layouts[1][2])
            ids = tuple(deck.deck_id for deck in decks)
        cases.extend(natural_cases(base, block=block, deck_ids=ids))
    return cases


def complete_natural_blocks(rows):
    grouped = {}
    for row in rows:
        variant = row["variant"]
        if row["condition"] != "natural" or variant not in range(4):
            raise ValueError("round-robin results must be natural variants 0..3")
        if row["winner"] not in (None, "a", "b"):
            raise ValueError("invalid winner")
        group = grouped.setdefault(row["block"], {})
        if variant in group:
            raise ValueError("duplicate round-robin game")
        group[variant] = row
    return {block: [group[v] for v in range(4)] for block, group in grouped.items()
            if len(group) == 4 and all(_finished(row) for row in group.values())}


def bradley_terry(names, comparisons, *, anchor):
    """MM fit, half-point draws, 0.5 pseudo-win per direction to keep separated fits finite.

    comparisons contains (first model, second model, first score points, games).
    Ratings are in Elo units: expected score = 1 / (1 + 10 ** ((Rb - Ra) / 400)).
    """
    index = {name: i for i, name in enumerate(names)}
    if anchor not in index or len(index) != len(names):
        raise ValueError("invalid model names or rating anchor")
    edges = [(index[a], index[b], float(points), float(games)) for a, b, points, games in comparisons]
    if any(a == b or games <= 0 or not 0 <= points <= games for a, b, points, games in edges):
        raise ValueError("invalid Bradley-Terry comparison")
    reached = {index[anchor]}
    while True:
        expanded = reached | {b for a, b, _, _ in edges if a in reached} | {
            a for a, b, _, _ in edges if b in reached}
        if expanded == reached:
            break
        reached = expanded
    if len(reached) != len(names):
        raise ValueError("rating comparison graph is disconnected")
    wins = [0.0] * len(names)
    for a, b, points, games in edges:
        wins[a] += points + 0.5
        wins[b] += games - points + 0.5
    strength = [1.0] * len(names)
    for _ in range(2000):
        denominator = [0.0] * len(names)
        for a, b, _, games in edges:
            weight = (games + 1) / (strength[a] + strength[b])
            denominator[a] += weight
            denominator[b] += weight
        updated = [wins[i] / denominator[i] for i in range(len(names))]
        scale = updated[index[anchor]]
        updated = [value / scale for value in updated]
        change = max(abs(math.log(new / old)) for new, old in zip(updated, strength))
        strength = updated
        if change < 1e-8:
            break
    else:
        raise ValueError("Bradley-Terry fit did not converge")
    return {name: 400 * math.log10(strength[i]) for name, i in index.items()}


def round_robin_summary(pair_blocks, names, *, anchor, bootstrap_samples=1000, seed=1):
    """Bootstrap global blocks together across all model pairs, including rating refits."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    pairs = list(itertools.combinations(names, 2))
    rng = random.Random(seed)

    def interval(values):
        values = sorted(values)
        return [values[int(.025 * len(values))], values[min(len(values) - 1, int(.975 * len(values)))]]

    summaries = []
    for a, b in pairs:
        groups = pair_blocks.get((a, b), {})
        scores = [statistics.mean(1 - _score(row) for row in rows) for rows in groups.values()]
        rows = [row for group in groups.values() for row in group]
        ci = None
        if len(scores) >= 2:
            ci = interval([statistics.mean(rng.choices(scores, k=len(scores)))
                           for _ in range(bootstrap_samples)])
        summaries.append(dict(a=a, b=b, blocks=len(groups), games=len(rows),
                              a_wins=sum(row["winner"] == "a" for row in rows),
                              b_wins=sum(row["winner"] == "b" for row in rows),
                              draws=sum(row["winner"] is None for row in rows),
                              a_score=statistics.mean(scores) if scores else None, a_score_95=ci))
    shared = sorted(set.intersection(*(set(pair_blocks.get(pair, {})) for pair in pairs)))

    def comparisons(block_ids):
        return [(a, b, sum(1 - _score(row) for block in block_ids
                           for row in pair_blocks[(a, b)][block]), 4 * len(block_ids)) for a, b in pairs]

    ratings = None
    if shared:
        fitted = bradley_terry(names, comparisons(shared), anchor=anchor)
        samples = {name: [] for name in names}
        if len(shared) >= 2:
            for _ in range(bootstrap_samples):
                draw = bradley_terry(names, comparisons(rng.choices(shared, k=len(shared))), anchor=anchor)
                for name in names:
                    samples[name].append(draw[name])
        ratings = sorted([dict(model=name, elo=fitted[name], elo_95=interval(samples[name])
                               if samples[name] else None) for name in names], key=lambda item: -item["elo"])
    return dict(pairs=summaries, ratings=ratings, anchor=anchor, shared_rating_blocks=len(shared),
                bootstrap_samples=bootstrap_samples, bootstrap_seed=seed,
                rating_method="Bradley-Terry, Elo scale, 0.5 pseudo-win per direction per pair",
                interval_method="global block percentile bootstrap; ratings use blocks complete for every pair")


def _finished(row):
    return row["terminated"] and not row["truncated"]


def _score(row):
    return 0.5 if row["winner"] is None else float(row["winner"] == "b")


def _quantiles(values):
    values = sorted(values)
    return [values[min(len(values) - 1, int(q * len(values)))] for q in (0.1, 0.5, 0.9)] if values else None


def paired_summary(rows, *, max_score_drop=0.05, bootstrap_samples=5000, seed=1):
    """Resample complete eight-game blocks, never treat correlated games as independent."""
    if not 0 <= max_score_drop <= 1 or bootstrap_samples < 1:
        raise ValueError("invalid score-drop margin or bootstrap sample count")
    grouped = {}
    for row in rows:
        if row["condition"] not in ("natural", "forced") or row["variant"] not in range(4):
            raise ValueError("unknown evaluation condition or variant")
        block = grouped.setdefault(row["block"], {})
        key = (row["condition"], row["variant"])
        if key in block:
            raise ValueError("duplicate evaluation case")
        block[key] = row
    complete = []
    for block in grouped.values():
        if len(block) != 8 or not all(_finished(row) for row in block.values()):
            continue
        for variant in range(4):
            natural, forced = (block[(condition, variant)] for condition in ("natural", "forced"))
            for key in ("seed", "a_owner", "a_deck", "opening_deadline_seconds", "match"):
                if natural.get(key) != forced.get(key):
                    raise ValueError(f"paired conditions have different {key}")
        complete.append(block)
    deltas = [
        statistics.mean(_score(block[("forced", v)]) - _score(block[("natural", v)]) for v in range(4))
        for block in complete
    ]
    interval = None
    if len(deltas) >= 2:
        rng = random.Random(seed)
        boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(bootstrap_samples))
        interval = [boot[int(0.025 * bootstrap_samples)], boot[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]]
    conclusion = "insufficient_blocks" if len(deltas) < 20 else "inconclusive"
    if len(deltas) >= 20 and interval is not None:
        if interval[1] < -max_score_drop:
            conclusion = "drop_exceeds_margin"
        elif interval[0] > -max_score_drop:
            conclusion = "drop_within_margin"
    conditions = {}
    for condition in ("natural", "forced"):
        observed = [row for row in rows if row["condition"] == condition]
        finished = [row for row in observed if _finished(row)]
        wins = sum(row["winner"] == "b" for row in finished)
        draws = sum(row["winner"] is None for row in finished)
        paired = [block[(condition, v)] for block in complete for v in range(4)]
        conditions[condition] = {
            "games": len(observed), "finished": len(finished),
            "unfinished": len(observed) - len(finished),
            "b_wins": wins, "b_losses": len(finished) - wins - draws, "draws": draws,
            "b_win_rate": wins / len(finished) if finished else None,
            "b_score": statistics.mean(map(_score, finished)) if finished else None,
            "paired_b_score": statistics.mean(map(_score, paired)) if paired else None,
            "forced_games": sum(row.get("opening_overrides", {}).get("b", 0) > 0 for row in observed),
            "rejected_actions": sum(row.get("rejected_actions", 0) for row in observed),
            "first_card_executed_seconds_10_50_90": {
                side: _quantiles([
                    row["first_card_executed_seconds"][side] for row in observed
                    if row.get("first_card_executed_seconds", {}).get(side) is not None
                ]) for side in ("a", "b")
            },
        }
    return {
        "conditions": conditions,
        "complete_blocks": len(complete), "excluded_blocks": len(grouped) - len(complete),
        "paired_games_per_condition": 4 * len(complete),
        "b_score_delta_forced_minus_natural": statistics.mean(deltas) if deltas else None,
        "b_score_delta_95": interval,
        "interval_method": "paired block percentile bootstrap",
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "max_score_drop": max_score_drop, "conclusion": conclusion,
    }
