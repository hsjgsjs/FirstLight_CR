"""Offline checks for a saved live capture; never connects to ADB or the game."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path

from .live_match import LiveCaptureNative, LiveEpisode


class FrozenCaptureNative(LiveCaptureNative):
    def __init__(self, capture):
        super().__init__("127.0.0.1", 0)
        self.capture = capture

    def _request(self, command, **kwargs):
        if command.startswith("live observe-atomic"):
            return copy.deepcopy(self.capture)
        raise RuntimeError(f"offline preflight forbids commands: {command}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--account-id", type=int, help="Account ID from your local capture; required with --checkpoint")
    args = parser.parse_args()
    if args.checkpoint and args.account_id is None:
        parser.error("--account-id is required with --checkpoint")
    with gzip.open(args.capture, "rt", encoding="utf-8") as stream:
        capture = json.load(stream)
    print("tick", capture.get("tick"), "generation", capture.get("generation"))
    for player in capture["ordinary"]["players"]:
        print("player", {"owner": player.get("owner")})
    for item in capture.get("rich", {}).get("objects", []):
        if item.get("cardId") == -1 or item.get("towerTroopRuntime"):
            print("tower", {k: v for k, v in item.items() if k in (
                "slot", "nativeObjectId", "entityKey", "owner", "cardId", "dataGlobalId", "x", "y",
                "hp", "maxHp", "towerTroopRuntime")})
    if args.checkpoint:
        from .training.v4.policy_session import load_policy_v4
        loaded = load_policy_v4(args.checkpoint, device="cpu")
        native = FrozenCaptureNative(capture)
        owner = next(p["owner"] for p in capture["ordinary"]["players"] if p["accountId"] == args.account_id)
        episode = LiveEpisode(native, loaded.model, owner, device="cpu")
        try:
            decision = episode.session.decide(episode.observation)
            print("PREFLIGHT_OK", "owner", owner, "tick", episode.observation.tick,
                  "towers", [episode.match.tower_troop0_id, episode.match.tower_troop1_id],
                  "actions", [a.to_dict() for a in decision.decoded.actions])
        finally:
            episode.close()
            native.close_transport()


if __name__ == "__main__":
    main()
