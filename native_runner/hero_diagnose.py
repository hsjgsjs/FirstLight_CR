#!/usr/bin/env python3
"""Read-only snapshot of Hero/Evo form and ability state in a running match.

Run it while a native-render match is on screen (MuMu or phone) and attach
the JSON to a bug report.  It shows, per owner, the form the engine actually
dealt for every hand card, the raw ability/evolution controllers, and whether
the strict rich-telemetry validation accepts them.

    python -m native_runner.hero_diagnose > hero-state.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from .card_names import CARD_NAMES
from .cr_native_env import NativeClashEnv, RunnerError, decode_native_card_parameter
from .local_config import setting


def _hand_card(card: dict[str, Any]) -> dict[str, Any]:
    card_id = int(card.get("cardId", 0))
    entry: dict[str, Any] = {
        "handIndex": card.get("handIndex"),
        "cardId": card_id,
        "name": " / ".join(CARD_NAMES.get(card_id, (str(card_id),))),
        "commandCardId": card.get("commandCardId"),
        "cost": card.get("cost"),
    }
    parameter = card.get("cardParameter")
    if parameter is None:
        entry["form"] = "not-encoded"
        return entry
    try:
        descriptor = decode_native_card_parameter(int(parameter))
    except ValueError as error:
        entry["form"] = f"invalid: {error}"
    else:
        entry["form"] = descriptor.form_name
        entry["deckSlot"] = descriptor.deck_slot
    return entry


def snapshot(env: NativeClashEnv) -> dict[str, Any]:
    observation = env.observe()
    report: dict[str, Any] = {"tick": observation.get("tick"), "players": []}
    for player in observation.get("players", ()):
        report["players"].append(
            {
                "owner": player.get("owner"),
                "elixirRaw": player.get("elixirRaw"),
                "hand": [_hand_card(card) for card in player.get("hand", ())],
            }
        )
    # The raw envelope bypasses host validation so a rejected field is still
    # visible; the validated call below reports why it was rejected.
    raw = env._request("observe-rich")
    report["rawPlayerRuntime"] = raw.get("players")
    try:
        env.observe_rich()
        report["richValidation"] = "ok"
    except RunnerError as error:
        report["richValidation"] = str(error)
    try:
        report["touch"] = env.touch_status()
    except RunnerError as error:
        report["touch"] = str(error)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=setting("CR_CONTROL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(setting("CR_CONTROL_PORT", "26789")))
    args = parser.parse_args(None if argv is None else list(argv))
    env = NativeClashEnv(args.host, args.port, timeout=10.0)
    print(json.dumps(snapshot(env), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
