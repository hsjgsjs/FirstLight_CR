from __future__ import annotations

import pytest

from native_runner.cr_native_env import RunnerError
from native_runner.live_match import match_config_from_capture, tower_troop_of


def capture_with_tower(kind=None, owner=1):
    players = [{"owner": o, "deck": [
        {"deckSlot": slot, "cardId": 26000000 + slot} for slot in range(8)
    ], "hand": []} for o in (0, 1)]
    return {"ordinary": {"players": players}, "rich": {"objects": [
        {"owner": owner, "cardId": -1, "towerTroopRuntime": {"kind": kind} if kind else None}
    ]}}


@pytest.mark.parametrize("kind,expected", [(None, 159000000), ("royal_chef", 159000004),
                                         ("dagger_duchess", 159000002)])
@pytest.mark.parametrize("owner", [0, 1])
def test_live_match_carries_exact_special_tower_runtime(kind, expected, owner):
    capture = capture_with_tower(kind, owner)
    match = match_config_from_capture(capture)
    assert getattr(match, f"tower_troop{owner}_id") == expected
    assert getattr(match, f"tower_troop{1-owner}_id") == 159000000


def test_conflicting_tower_runtime_is_rejected_instead_of_guessed():
    capture = capture_with_tower("royal_chef")
    capture["rich"]["objects"].append({"owner": 1, "cardId": -1,
                                       "towerTroopRuntime": {"kind": "dagger_duchess"}})
    with pytest.raises(RunnerError, match="conflicting"):
        tower_troop_of(capture, 1)


def test_non_tower_runtime_cannot_change_tower_config():
    capture = capture_with_tower("royal_chef")
    capture["rich"]["objects"][0]["cardId"] = 26000000
    assert tower_troop_of(capture, 1) == 159000000
