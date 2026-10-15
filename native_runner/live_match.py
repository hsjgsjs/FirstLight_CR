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
import re
import time
from typing import Any

from .cr_native_env import NativeClashEnv, RunnerError, decode_native_card_parameter
from .match_factory import MatchConfig
from .normal_form_evidence import NORMAL_MODE_HERO_FORM_TO_BASE_CARD

# Without a readable slot flag, a card that can evolve is taken as an evolution only in
# the first two deck slots.  Forms the battle itself shows always win over this guess.
LIVE_EVOLUTION_SLOTS = 2
EVOLUTION_FORM, HERO_FORM = 1, 2

# BattleEnvV1's complaint about a form the episode deck contract left out.
_CONTRACT_MISS = re.compile(r"deck contract: owner=(\d+), card=(\d+), (?:form=(\d+), )?enabled=")

FormsSeen = dict[int, dict[int, int]]  # owner -> deck slot -> form (1 evolution, 2 hero)


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


def observe_forms(capture: Mapping[str, Any], seen: FormsSeen) -> None:
    """Record the evolution / hero forms the capture's packed card selections carry."""

    for player in (capture.get("ordinary") or {}).get("players", ()):
        owner = int(player.get("owner", -1))
        for card in (*(player.get("deck") or ()), *(player.get("hand") or ())):
            try:
                form = decode_native_card_parameter(int(card["cardParameter"])).form_code
            except (KeyError, TypeError, ValueError):
                continue
            if int(card.get("commandCardId", card.get("cardId", 0))) in NORMAL_MODE_HERO_FORM_TO_BASE_CARD:
                form = HERO_FORM
            if form in (EVOLUTION_FORM, HERO_FORM):
                seen.setdefault(owner, {})[int(card["deckSlot"])] = form


def is_contract_miss(message: str) -> bool:
    return _CONTRACT_MISS.search(message) is not None


def learn_contract_miss(message: str, capture: Mapping[str, Any], seen: FormsSeen) -> bool:
    """Add the form a BattleEnvV1 contract error names to ``seen``; False if it names none."""

    match = _CONTRACT_MISS.search(message)
    if match is None:
        return False
    owner, card_id = int(match.group(1)), int(match.group(2))
    form = int(match.group(3) or HERO_FORM)
    slots = [slot for slot, deck_card in enumerate(deck_of(capture, owner)) if deck_card == card_id]
    if len(slots) != 1 or form not in (EVOLUTION_FORM, HERO_FORM) or seen.get(owner, {}).get(slots[0]) == form:
        return False
    seen.setdefault(owner, {})[slots[0]] = form
    return True


def form_availability_of(capture: Mapping[str, Any], owner: int, seen: FormsSeen | None = None) -> tuple[int, ...]:
    """Per deck slot: 2 hero, 1 evolution, 0 base, within the V4 contract (<=2 each, <=3 in all)."""

    known = dict((seen or {}).get(owner, {}))
    masks = [known.get(slot, 0) for slot in range(8)]
    guessed = []
    for player in (capture.get("rich") or {}).get("players", ()):
        if int(player.get("owner", -1)) != owner:
            continue
        for slot in player.get("evolutionRuntime") or ():
            deck_slot = int(slot.get("deckSlot", -1))
            if slot.get("evolvable") and 0 <= deck_slot < LIVE_EVOLUTION_SLOTS and deck_slot not in known:
                masks[deck_slot] = EVOLUTION_FORM
                guessed.append(deck_slot)

    def over() -> bool:
        evolutions, heroes = masks.count(EVOLUTION_FORM), masks.count(HERO_FORM)
        return evolutions > 2 or heroes > 2 or evolutions + heroes > 3

    while over() and guessed:  # a seen form outranks a guess
        masks[guessed.pop()] = 0
    if over():
        raise RunnerError(f"owner {owner} shows more evolution/hero forms than the V4 contract allows: {masks}")
    return tuple(masks)


def tower_troop_of(capture: Mapping[str, Any], owner: int) -> int:
    """Use exact public tower runtime, rather than assuming every live match is Princess.

    Chef belongs to the King Tower and Duchess to side towers. Their native
    runtime snapshots explicitly identify the type; no image/HP heuristic is used.
    The existing Princess default is retained when neither special runtime is present.
    """
    special = set()
    for item in (capture.get("rich") or {}).get("objects", ()):
        if int(item.get("owner", -1)) != owner or int(item.get("cardId", 0)) != -1:
            continue
        runtime = item.get("towerTroopRuntime") or {}
        kind = runtime.get("kind")
        if kind == "royal_chef":
            special.add(159_000_004)
        elif kind == "dagger_duchess":
            special.add(159_000_002)
    if len(special) > 1:
        raise RunnerError(f"owner {owner} has conflicting live Tower Troop runtimes: {sorted(special)}")
    return next(iter(special), 159_000_000)


def match_config_from_capture(capture: Mapping[str, Any], seen: FormsSeen | None = None) -> MatchConfig:
    """The MatchConfig BattleEnvV1 needs, read from one live capture (plus forms seen earlier)."""

    seen = {owner: dict(slots) for owner, slots in (seen or {}).items()}
    observe_forms(capture, seen)
    return MatchConfig(
        deck0=deck_of(capture, 0),
        deck1=deck_of(capture, 1),
        deck0_form_availability=form_availability_of(capture, 0, seen),
        deck1_form_availability=form_availability_of(capture, 1, seen),
        tower_troop0_id=tower_troop_of(capture, 0),
        tower_troop1_id=tower_troop_of(capture, 1),
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
    public_card_play_events_from_combat_ring = True

    def observe_atomic(self) -> dict[str, Any]:
        command = "live observe-atomic"
        if self.floors is not None:
            command += " " + " ".join(str(int(value)) for value in self.floors())
        started = time.monotonic_ns()
        result = self._request(command)
        sink = getattr(self, "capture_sink", None)
        if sink is not None:
            # Preserve the raw response even if validation subsequently rejects
            # it; failure evidence must not disappear from a valuable live game.
            self.last_capture_id = sink(result, request_started_ns=started)
        return self._validate_atomic_result(result)


class LiveEpisode:
    """BattleEnvV1 + a V4 policy session for one seat of a battle read through a live native."""

    def __init__(
        self,
        native: _LiveNativeBase,
        model: Any,
        owner: int,
        *,
        device: str = "cpu",
        forms_seen: FormsSeen | None = None,
        sample: bool = True,
    ) -> None:
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
        native.floors = None  # a fresh env has consumed nothing yet
        self.match = match = match_config_from_capture(native.observe_atomic(), forms_seen)
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
        # Sampled like offline_agent: a deterministic argmax of a per-0.25s decision is almost
        # always "wait", so a greedy policy hardly ever plays a card.
        self.session = build_policy_session_v4(self.env, model, actor_owner=owner, device=device, sample=sample)
        elixir = {o: next(p.elixir_exact for p in obs.players if p.owner == o) for o, obs in observations.items()}
        self.session.start_episode(observations[owner], initial_elixir=elixir)
        self.observation = observations[owner]

    @property
    def raw(self) -> Mapping[str, Any]:
        return self.env._raw

    def close(self) -> None:
        self.session.end_episode()
