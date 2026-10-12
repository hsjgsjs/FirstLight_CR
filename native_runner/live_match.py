"""Bridge a live battle (probe live mode) to FirstLight's BattleEnvV1 and policy.

``LiveCaptureNative`` stands in for ``NativeClashEnv`` so the unchanged
BattleEnvV1 / tensorizer / policy stack reads a battle the game runs itself:
creating, pausing or speeding a match are no-ops (the game owns the battle) and
every observation is one game-thread capture from ``live observe-atomic``.
``live_play.RecordedTapNative`` plays saved captures back the same way, for offline checks.

Event rings travel as deltas: each capture asks the probe only for the events
after the cursors BattleEnvV1 has already consumed (``event_floors``), so a
capture stays small however long the battle runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


# The rings sent as deltas, in the order of the probe's "live observe-atomic" cursors.
EVENT_RINGS = ("combatEvents", "phaseRuntime", "visibilityRuntime", "remainingRuntime")

EventFloors = Callable[[], Sequence[int]]


def event_floors(env: Any) -> tuple[int, int, int, int]:
    """The next unseen sequence of each ring BattleEnvV1 has consumed (0: send everything)."""

    combat = [value for value in (env._combat_next_sequence, env._public_card_play_next_sequence) if value is not None]
    return (
        min(combat) if combat else 0,
        env._phase_next_sequence or 0,
        env._visibility_next_sequence or 0,
        env._remaining_next_sequence or 0,
    )


def trim_event_rings(capture: Mapping[str, Any], floors: Sequence[int]) -> dict[str, Any]:
    """What the probe sends for ``floors``: each ring from max(oldest retained, min(floor, next))."""

    trimmed = json.loads(json.dumps(capture))
    rich = trimmed.get("rich") or {}
    for name, floor in zip(EVENT_RINGS, floors):
        ring = rich.get(name)
        if not isinstance(ring, dict) or not isinstance(ring.get("events"), list):
            continue
        sent_from = int(ring.get("transmitFromSequence", ring["oldestRetainedSequence"]))
        start = max(int(ring["oldestRetainedSequence"]), min(int(floor), int(ring["nextSequence"])))
        if start > sent_from:
            ring["events"] = ring["events"][start - sent_from :]
            ring["transmitFromSequence"] = start
    return trimmed


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

    floors: EventFloors | None = None

    def observe_atomic(self) -> dict[str, Any]:
        command = "live observe-atomic"
        if self.floors is not None:
            command += " " + " ".join(str(int(value)) for value in self.floors())
        return self._validate_atomic_result(self._request(command))


class LiveEpisode:
    """BattleEnvV1 + a V4 policy session for one seat of a battle read through a live native."""

    def __init__(self, native: _LiveNativeBase, model: Any, owner: int, *, device: str = "cpu") -> None:
        from .battle_env import BattleEnvV1
        from .rich_telemetry_adapter import RuntimeEffectCatalog
        from .training.v4.expert import POLICY_DECISION_TICKS
        from .training.v4.factory import production_semantic_bundle
        from .training.v4.policy_session import build_policy_session_v4

        bundle = production_semantic_bundle()
        self.native = native
        self.owner = owner
        self.env = BattleEnvV1(
            native=native,
            card_catalog=bundle.native_card_catalog,
            semantic_subset_contract=bundle.semantic_subset_contract,
            effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
        )
        # The stock game owns a live battle: there is no offline ruleset endpoint to attest.
        self.env.ruleset_manifest = None
        match = match_config_from_capture(native.observe_atomic())
        observations, _info = self.env.reset(
            match_config=match,
            options={
                "render_mode": "native-render",
                "allow_initial_remaining_runtime_rejections": True,
                "decision_ticks": POLICY_DECISION_TICKS,
                "event_driven_decisions": False,
            },
        )
        native.floors = lambda: event_floors(self.env)
        self.session = build_policy_session_v4(self.env, model, actor_owner=owner, device=device, sample=False)
        elixir = {o: next(p.elixir_exact for p in obs.players if p.owner == o) for o, obs in observations.items()}
        self.session.start_episode(observations[owner], initial_elixir=elixir)
        self.observation = observations[owner]

    @property
    def raw(self) -> Mapping[str, Any]:
        return self.env._raw

    def close(self) -> None:
        self.session.end_episode()
