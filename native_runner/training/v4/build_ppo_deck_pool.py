"""Build one exact-form-plus-Tower-Troop PPO deck-pool manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import orjson
import pyarrow.parquet as pq

from .deck_pool import (
    DeckPoolManifestV4,
    build_deck_pool_manifest_from_counts,
    deck_counts_from_payloads,
    exclude_decks_with_cards,
    mark_focus_decks,
)


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"


def _payload_rows(parquet_parts: tuple[Path, ...]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for path in parquet_parts:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if not {"replay_tag", "payload_json"}.issubset(names):
            continue
        for batch in parquet.iter_batches(columns=("replay_tag", "payload_json"), batch_size=4096):
            replay_tags = batch.column(0).to_pylist()
            payloads = batch.column(1).to_pylist()
            for replay_tag, raw_payload in zip(replay_tags, payloads, strict=True):
                payload = orjson.loads(raw_payload)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"payload for replay {replay_tag!r} is not a mapping")
                yield str(replay_tag), payload


def _focus_identities(path: Path) -> tuple[tuple[tuple[str, ...], str], ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("decks") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise ValueError("focus deck file must be a list or contain a deck list")
    result: list[tuple[tuple[str, ...], str]] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("focus deck row must be a mapping")
        if row.get("rank", rank) != rank:
            raise ValueError("focus deck ranks must be ordered and contiguous")
        cards = row.get("card_keys")
        if not isinstance(cards, list):
            raise ValueError("focus deck row has no card_keys list")
        result.append((tuple(str(item) for item in cards), str(row.get("tower_key", "tower-princess"))))
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V4 PPO exact deck plus Tower Troop pool")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-uses", type=int, default=5)
    parser.add_argument("--focus-decks", type=Path)
    parser.add_argument("--expected-pool-size", type=int)
    parser.add_argument(
        "--from-pool",
        type=Path,
        help=(
            "start from an existing pool instead of scanning replays, to re-mark its focus decks "
            "or drop decks (needs --focus-decks or --exclude-card-id)"
        ),
    )
    parser.add_argument(
        "--exclude-card-id",
        type=int,
        action="append",
        default=[],
        help="repeatable native card ID (e.g. 28000006 Mirror): drop every deck that holds it",
    )
    args = parser.parse_args()

    if args.from_pool is not None:
        if args.focus_decks is None and not args.exclude_card_id:
            parser.error("--from-pool needs --focus-decks or --exclude-card-id")
        if args.from_pool.resolve() == args.output.resolve():
            parser.error("--from-pool must not overwrite its own input")
        # Same decks and usage counts; only the focus ranks change, or excluded decks go.
        manifest = DeckPoolManifestV4.load(args.from_pool)
        scan_metadata = {"source_pool": str(args.from_pool.resolve())}
        manifest = DeckPoolManifestV4(
            decks=manifest.decks,
            minimum_uses=manifest.minimum_uses,
            metadata={**dict(manifest.metadata), **scan_metadata},
        )
    else:
        parts = tuple(sorted(args.dataset_root.glob("*/replays/*.parquet")))
        if not parts:
            raise FileNotFoundError(f"no replay Parquet parts found below {args.dataset_root}")
        counts, scan_metadata = deck_counts_from_payloads(_payload_rows(parts))
        manifest = build_deck_pool_manifest_from_counts(
            counts,
            minimum_uses=args.minimum_uses,
            metadata={
                **scan_metadata,
                "dataset_root": str(args.dataset_root.resolve()),
                "parquet_part_count": len(parts),
            },
        )
    if args.focus_decks is not None:
        manifest = mark_focus_decks(manifest, _focus_identities(args.focus_decks))
    if args.exclude_card_id:
        manifest = exclude_decks_with_cards(manifest, args.exclude_card_id)
    if args.expected_pool_size is not None and len(manifest.decks) != args.expected_pool_size:
        raise RuntimeError(f"expected {args.expected_pool_size} decks, found {len(manifest.decks)}")
    manifest.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pool_size": len(manifest.decks),
                "focus_decks": len(manifest.focus_decks),
                **scan_metadata,
                **{
                    key: manifest.metadata[key]
                    for key in ("excluded_card_ids", "excluded_deck_count")
                    if key in manifest.metadata
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
