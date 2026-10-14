"""CPU evaluation schedule and paired statistics; no inference or engine dependencies."""

from dataclasses import dataclass, replace
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
