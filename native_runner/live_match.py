"""Bridge a live battle (probe live mode) to FirstLight's BattleEnvV1 and policy.

``LiveCaptureNative`` stands in for ``NativeClashEnv`` so the unchanged
BattleEnvV1 / tensorizer / policy stack reads a battle the game runs itself:
creating, pausing or speeding a match are no-ops (the game owns the battle) and
every observation is one game-thread capture from ``live observe-atomic``.
``RecordedLiveNative`` serves saved captures the same way, for offline checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from .cr_native_env import NativeClashEnv, RunnerError
from .match_factory import MatchConfig

# A ladder-style deck evolves only in its first two slots (the wild slot is
# not readable from a live capture yet, so it is treated as a base card).
LIVE_EVOLUTION_SLOTS = 2


def _player(capture: Mapping[str, Any], owner: int) -> Mapping[str, Any]:
    ordinary = capture.get("ordinary") or {}
    for player in ordinary.get("players", ()):
        if int(player.get("owner", -1)) == owner:
            return player
    raise RunnerError(f"live capture has no owner {owner}")


def deck_of(capture: Mapping[str, Any], owner: int) -> tuple[int, ...]:
    cards = sorted(_player(capture, owner).get("deck", ()), key=lambda card: int(card["deckSlot"]))
    deck = tuple(int(card["cardId"]) for card in cards)
    if len(deck) != 8:
        raise RunnerError(f"owner {owner} deck is not readable from the live capture ({len(deck)} cards)")
    return deck


def form_availability_of(capture: Mapping[str, Any], owner: int) -> tuple[int, ...]:
    masks = [0] * 8
    for player in (capture.get("rich") or {}).get("players", ()):
        if int(player.get("owner", -1)) != owner:
            continue
        for slot in player.get("evolutionRuntime") or ():
            deck_slot = int(slot.get("deckSlot", -1))
            if slot.get("evolvable") and 0 <= deck_slot < LIVE_EVOLUTION_SLOTS:
                masks[deck_slot] |= 1
    return tuple(masks)


def match_config_from_capture(capture: Mapping[str, Any]) -> MatchConfig:
    """The MatchConfig BattleEnvV1 needs, read from one live capture."""

    return MatchConfig(
        deck0=deck_of(capture, 0),
        deck1=deck_of(capture, 1),
        deck0_form_availability=form_availability_of(capture, 0),
        deck1_form_availability=form_availability_of(capture, 1),
    )


class _LiveNativeBase(NativeClashEnv):
    """Match lifecycle calls are no-ops: the stock game owns a live battle."""

    def create_native_match(self, config: MatchConfig, **_: Any) -> dict[str, Any]:
        return dict(self.observe_atomic()["ordinary"])

    def pause(self) -> dict[str, Any]:
        return {"ok": True}

    def resume(self) -> dict[str, Any]:
        return {"ok": True}

    def set_speed(self, multiplier: float) -> dict[str, Any]:
        return {"ok": True}

    def set_rendering(self, enabled: bool) -> dict[str, Any]:
        return {"ok": True}

    def observe(self) -> dict[str, Any]:
        return dict(self.observe_atomic()["ordinary"])


class LiveCaptureNative(_LiveNativeBase):
    """Reads the live battle through the probe's game-thread capture."""

    def observe_atomic(self) -> dict[str, Any]:
        return self._validate_atomic_result(self._request("live observe-atomic"))


class RecordedLiveNative(_LiveNativeBase):
    """Serves saved live captures in order; ``advance()`` moves to the next one."""

    def __init__(self, captures: Sequence[Mapping[str, Any]]) -> None:
        super().__init__()
        if not captures:
            raise ValueError("no recorded live captures")
        self.captures = [json.loads(json.dumps(capture)) for capture in captures]
        self.index = 0
        self.public_card_play_events_from_combat_ring = True

    def _request(self, command: str, **_: Any) -> dict[str, Any]:
        raise RunnerError(f"recorded live captures cannot answer {command!r}")

    def advance(self) -> bool:
        if self.index + 1 >= len(self.captures):
            return False
        self.index += 1
        return True

    def observe_atomic(self) -> dict[str, Any]:
        return self._validate_atomic_result(json.loads(json.dumps(self.captures[self.index])))
