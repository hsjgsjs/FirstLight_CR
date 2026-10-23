from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from native_runner.training.v4 import build_ppo_deck_pool
from native_runner.training.v4.deck_pool import DeckPoolManifestV4, deck_identity_from_keys, mark_focus_decks

FOCUS_DIR = Path(build_ppo_deck_pool.__file__).resolve().parent
FOCUS_V1 = FOCUS_DIR / "focus_decks_20.json"
FOCUS_V2 = FOCUS_DIR / "focus_decks_20_v2.json"


def _rows(path: Path) -> list[tuple[tuple[str, ...], str]]:
    return list(build_ppo_deck_pool._focus_identities(path))


def _run(*args: str) -> None:
    with mock.patch.object(sys, "argv", ["build_ppo_deck_pool", *map(str, args)]):
        assert build_ppo_deck_pool.main() == 0


@pytest.mark.parametrize("path", [FOCUS_V1, FOCUS_V2])
def test_shipped_focus_files_hold_twenty_distinct_decks(path: Path) -> None:
    identities = {deck_identity_from_keys(cards, tower, uses=1).deck_id for cards, tower in _rows(path)}
    assert len(identities) == 20


def test_v2_swaps_ranks_ten_and_twenty_for_goblin_barrel_and_mega_knight() -> None:
    v1, v2 = _rows(FOCUS_V1), _rows(FOCUS_V2)
    changed = [rank for rank, (old, new) in enumerate(zip(v1, v2, strict=True), start=1) if old != new]
    assert changed == [10, 20]
    assert "goblin-barrel-ev1" in v2[9][0]
    assert "mega-knight-ev1" in v2[19][0]


def test_from_pool_keeps_decks_and_uses_and_only_moves_focus_ranks(tmp_path: Path) -> None:
    v1, v2 = _rows(FOCUS_V1), _rows(FOCUS_V2)
    rows = {deck_identity_from_keys(cards, tower, uses=1).deck_id: (cards, tower) for cards, tower in v1 + v2}
    decks = tuple(
        deck_identity_from_keys(cards, tower, uses=10 + index) for index, (cards, tower) in enumerate(rows.values())
    )
    source = mark_focus_decks(DeckPoolManifestV4(decks=decks, minimum_uses=5, metadata={"pool_size": 22}), v1)
    source_path, output_path = tmp_path / "pool.json", tmp_path / "pool-v2.json"
    source.save(source_path)

    _run("--from-pool", source_path, "--focus-decks", FOCUS_V2, "--output", output_path, "--expected-pool-size", 22)

    result = DeckPoolManifestV4.load(output_path)
    assert [(item.deck_id, item.uses) for item in result.decks] == [(item.deck_id, item.uses) for item in source.decks]
    expected_v2 = [deck_identity_from_keys(cards, tower, uses=1).deck_id for cards, tower in v2]
    assert [item.deck_id for item in result.focus_decks] == expected_v2
    # The two decks swapped out are now ordinary tail decks.
    dropped = {deck_identity_from_keys(*v1[9], uses=1).deck_id, deck_identity_from_keys(*v1[19], uses=1).deck_id}
    assert {item.deck_id for item in result.tail_decks} == dropped
    assert result.metadata["source_pool"] == str(source_path.resolve())
    assert result.metadata["pool_size"] == 22
    # The source pool is left as it was.
    assert DeckPoolManifestV4.load(source_path) == source


def test_from_pool_refuses_a_focus_deck_missing_from_the_pool(tmp_path: Path) -> None:
    v1 = _rows(FOCUS_V1)
    decks = tuple(deck_identity_from_keys(cards, tower, uses=10) for cards, tower in v1)
    source_path = tmp_path / "pool.json"
    mark_focus_decks(DeckPoolManifestV4(decks=decks, minimum_uses=5, metadata={}), v1).save(source_path)
    with pytest.raises(ValueError, match="absent from the usage-filtered pool"):
        _run("--from-pool", source_path, "--focus-decks", FOCUS_V2, "--output", tmp_path / "out.json")


def test_from_pool_needs_focus_decks_and_a_separate_output(tmp_path: Path) -> None:
    source_path = tmp_path / "pool.json"
    source_path.write_text(json.dumps({}))
    with pytest.raises(SystemExit):
        _run("--from-pool", source_path, "--output", tmp_path / "out.json")
    with pytest.raises(SystemExit):
        _run("--from-pool", source_path, "--focus-decks", FOCUS_V2, "--output", source_path)
