"""Select human replays for expert BC: scoreline quotas, low elixir leak, deck caps.

Close expert games (1-0) dominate the selection; one-sided wins are sampled
less. Only sides whose elixir leak is at or below the winners' quantile are
learned. The losing side of a 1-0 game is also learned at a lower weight; the
cache stores both owners anyway, so it costs no extra reconstruction.

Outputs ``replays/part-000000.parquet`` (selected rows in random order, so a
cache_builder ``--source-count`` prefix is representative), ``selection.jsonl``
(sides to learn, with native owners) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...royaleapi_replay import RoyaleAPIReplayError, resolve_native_card

SCORELINE_SHARES = {"1-0": 0.50, "2-0": 0.20, "2-1": 0.08, "3-0": 0.10, "3-1": 0.07, "3-2": 0.05}
FILL_ORDER = ("1-0", "2-0", "3-0", "2-1", "3-1", "3-2")
DEFAULT_MODES = ("ranked", "ladder", "grand challenge")
CARD_SUPPORT = Path(__file__).resolve().parents[2] / "data" / "competitive" / "card_support.json"
COLUMNS = ("replay_tag", "game_mode", "result", "team_crowns", "opponent_crowns", "payload_json")


def supported_card_ids(path: Path = CARD_SUPPORT) -> frozenset[int]:
    return frozenset(int(card["card_id"]) for card in json.loads(path.read_text(encoding="utf-8"))["supported_cards"])


class DeckSupport:
    def __init__(self, supported: frozenset[int]) -> None:
        self.supported = supported
        self.cache: dict[tuple[str, str], bool] = {}

    def card(self, item: Mapping[str, Any]) -> bool:
        key = (str(item.get("card_key") or ""), str(item.get("name") or ""))
        if key not in self.cache:
            try:
                self.cache[key] = bool(key[0]) and resolve_native_card(*key).card_id in self.supported
            except (RoyaleAPIReplayError, KeyError, ValueError):
                self.cache[key] = False
        return self.cache[key]

    def deck(self, deck: object) -> bool:
        return isinstance(deck, list) and len(deck) == 8 and all(isinstance(c, Mapping) and self.card(c) for c in deck)


def owner_for_side(payload: Mapping[str, Any]) -> dict[str, int] | None:
    # Cache owners are native True Red=0 / True Blue=1, as in royaleapi_replay.
    markers = {
        event.get("source_fields", {}).get("data_i")
        for event in payload.get("events", [])
        if "data_i" in event.get("source_fields", {})
    }
    if len(markers) != 1 or next(iter(markers)) not in (0, 1):
        return None
    data_i = next(iter(markers))
    return {"team": 1 - data_i, "opponent": data_i}


def candidate(row: Mapping[str, Any], modes: Sequence[str], support: DeckSupport) -> dict[str, Any] | None:
    if str(row.get("game_mode") or "").casefold() not in modes or row.get("result") not in ("victory", "defeat"):
        return None
    winner, loser = ("team", "opponent") if row["result"] == "victory" else ("opponent", "team")
    crowns = {"team": int(row.get("team_crowns") or 0), "opponent": int(row.get("opponent_crowns") or 0)}
    scoreline = f"{crowns[winner]}-{crowns[loser]}"
    if scoreline not in SCORELINE_SHARES:
        return None
    payload = orjson.loads(row["payload_json"])
    battle = payload.get("battle", {})
    players = {side: battle.get(side, {}).get("players", []) for side in (winner, loser)}
    if any(len(value) != 1 for value in players.values()):
        return None
    if not all(support.deck(players[side][0].get("deck")) for side in (winner, loser)):
        return None
    owners = owner_for_side(payload)
    leaks = {side: players[side][0].get("elixir_leaked") for side in (winner, loser)}
    if owners is None or not all(isinstance(value, (int, float)) for value in leaks.values()):
        return None
    deck = tuple(sorted(str(card["card_key"]) for card in players[winner][0]["deck"]))
    return {
        "replay_tag": str(row["replay_tag"]),
        "game_mode": str(row["game_mode"]),
        "scoreline": scoreline,
        "deck": deck,
        "sides": {
            "winner": {"side": winner, "owner": owners[winner], "elixir_leaked": float(leaks[winner])},
            "loser": {"side": loser, "owner": owners[loser], "elixir_leaked": float(leaks[loser])},
        },
    }


def iter_rows(files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=list(COLUMNS)):
            yield from batch.to_pylist()


def leak_cutoff(candidates: Sequence[Mapping[str, Any]], quantile: float) -> float:
    leaks = sorted(item["sides"]["winner"]["elixir_leaked"] for item in candidates)
    if not leaks:
        raise ValueError("no eligible replays")
    return leaks[min(len(leaks) - 1, max(0, int(quantile * (len(leaks) - 1))))]


def select(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target: int,
    deck_cap: int,
    cutoff: float,
    shares: Mapping[str, float] = SCORELINE_SHARES,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[Mapping[str, Any]]] = {scoreline: [] for scoreline in shares}
    for item in candidates:
        if item["sides"]["winner"]["elixir_leaked"] <= cutoff:
            groups[item["scoreline"]].append(item)
    for group in groups.values():
        rng.shuffle(group)
    cursors = Counter[str]()
    deck_counts = Counter[tuple[str, ...]]()
    chosen: list[dict[str, Any]] = []

    def take(scoreline: str, limit: int) -> None:
        group = groups[scoreline]
        taken = 0
        while taken < limit and cursors[scoreline] < len(group):
            item = group[cursors[scoreline]]
            cursors[scoreline] += 1
            if deck_counts[item["deck"]] >= deck_cap:
                continue
            deck_counts[item["deck"]] += 1
            chosen.append(learn_sides(item, cutoff))
            taken += 1

    for scoreline, share in shares.items():
        take(scoreline, round(target * share))
    for scoreline in FILL_ORDER:
        if len(chosen) >= target:
            break
        take(scoreline, target - len(chosen))
    rng.shuffle(chosen)
    return chosen[:target]


def learn_sides(item: Mapping[str, Any], cutoff: float) -> dict[str, Any]:
    winner, loser = item["sides"]["winner"], item["sides"]["loser"]
    learn = [{**winner, "outcome": "win", "weight": 1.0}]
    if item["scoreline"] == "1-0" and loser["elixir_leaked"] <= cutoff:
        learn.append({**loser, "outcome": "close_loss", "weight": 0.5})
    return {key: item[key] for key in ("replay_tag", "game_mode", "scoreline")} | {"learn": learn}


def write_outputs(files: Sequence[Path], chosen: Sequence[Mapping[str, Any]], output: Path) -> int:
    order = {item["replay_tag"]: index for index, item in enumerate(chosen)}
    wanted = pa.array(list(order))
    tables = []
    for path in files:
        table = pq.read_table(path)
        tables.append(table.filter(pc.is_in(table["replay_tag"], value_set=wanted)))
    selected = pa.concat_tables(tables)
    positions = [order[tag] for tag in selected["replay_tag"].to_pylist()]
    selected = selected.take(sorted(range(len(positions)), key=positions.__getitem__))
    (output / "replays").mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, output / "replays" / "part-000000.parquet", compression="zstd")
    with (output / "selection.jsonl").open("w", encoding="utf-8") as handle:
        for item in chosen:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return selected.num_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="directory containing replays/part-*.parquet")
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--deck-cap", type=int, default=25, help="maximum replays per winner deck")
    parser.add_argument("--leak-quantile", type=float, default=0.5, help="winner elixir-leak quantile to keep")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="comma-separated game_mode values")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")
    files = sorted((args.dataset_root / "replays").glob("part-*.parquet"))
    if not files:
        raise SystemExit(f"no replays/part-*.parquet under {args.dataset_root}")
    modes = tuple(mode.strip().casefold() for mode in args.modes.split(",") if mode.strip())
    support = DeckSupport(supported_card_ids())
    seen: set[str] = set()
    candidates = []
    scanned = 0
    for row in iter_rows(files):
        scanned += 1
        item = candidate(row, modes, support)
        if item is not None and item["replay_tag"] not in seen:
            seen.add(item["replay_tag"])
            candidates.append(item)
    cutoff = leak_cutoff(candidates, args.leak_quantile)
    chosen = select(candidates, target=args.target, deck_cap=args.deck_cap, cutoff=cutoff, seed=args.seed)
    written = write_outputs(files, chosen, args.output)
    summary = {
        "scanned_replays": scanned,
        "eligible_replays": len(candidates),
        "eligible_by_scoreline": dict(Counter(item["scoreline"] for item in candidates)),
        "winner_elixir_leak_cutoff": cutoff,
        "selected_replays": written,
        "selected_by_scoreline": dict(Counter(item["scoreline"] for item in chosen)),
        "learned_sides": sum(len(item["learn"]) for item in chosen),
        "distinct_winner_decks": len({tuple(sorted(c["deck"])) for c in candidates}),
        "unsupported_card_keys": sorted(f"{k[0]}" for k, ok in support.cache.items() if not ok),
        "settings": {key: str(value) for key, value in vars(args).items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in list(summary)[:8]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
