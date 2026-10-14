from __future__ import annotations

import json
import random
from pathlib import Path

from native_runner.match_factory import MatchConfig
from native_runner.training.v4.deck_pool import DeckIdentityV4, DeckPoolManifestV4
from native_runner.training.v4.evaluate import sample_deck_pair
from native_runner.training.v4.evaluate_summary import summarize, wilson_interval


def _deck(index: int) -> DeckIdentityV4:
    return DeckIdentityV4(
        deck_id=f"deck-{index:03d}",
        card_keys=tuple(sorted(f"card-{index:03d}-{slot}" for slot in range(8))),
        card_ids=tuple(index * 100 + slot + 1 for slot in range(8)),
        form_availability=(0,) * 8,
        tower_key="tower-princess",
        tower_troop_id=159_000_000,
        uses=index + 5,
    )


def test_deck_pairs_come_from_the_pool_with_levels() -> None:
    pool = DeckPoolManifestV4(decks=tuple(_deck(index) for index in range(30)), minimum_uses=5, metadata={})
    paired, ids = sample_deck_pair(MatchConfig(), pool, random.Random(3), battle_level=11)
    by_id = {deck.deck_id: deck for deck in pool.decks}
    assert sorted(paired.deck0) == sorted(by_id[ids[0]].card_ids)
    assert sorted(paired.deck1) == sorted(by_id[ids[1]].card_ids)
    assert paired.tower_troop0_id == paired.tower_troop1_id == 159_000_000
    assert paired.level_cap == paired.minimum_card_level == paired.king_tower_level == 11
    # The same seed draws the same pairs.
    assert sample_deck_pair(MatchConfig(), pool, random.Random(3), battle_level=11) == (paired, ids)


def test_summary_counts_draws_as_half(tmp_path: Path) -> None:
    rows = [
        {"winner": "a", "first_play_seconds": {"a": 10, "b": 20}},
        {"winner": "b", "first_play_seconds": {"a": 12, "b": None}},
        {"winner": None, "first_play_seconds": {"a": 14, "b": 30}},
        {"winner": "a"},
    ]
    (tmp_path / "matches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = summarize([tmp_path])
    assert (result["games"], result["a_wins"], result["b_wins"], result["draws"]) == (4, 2, 1, 1)
    assert result["a_score"] == 0.625
    assert result["first_play_10_50_90"]["a"] == "10/12/14"
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and abs((low + high) / 2 - 0.5) < 1e-9
