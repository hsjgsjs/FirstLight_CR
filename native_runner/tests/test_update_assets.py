from __future__ import annotations

import hashlib
import json

import pytest

from native_runner.update_assets import fetch_verified, hash_convention, missing_files, parse_fingerprint


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


FINGERPRINT = json.dumps(
    {
        "sha": "c0ffee",
        "files": [
            {"file": "sc3d/characters/hero/hero_knight/hero_prefab_knight_battle.json", "sha": _sha(b"prefab")},
            {"file": "sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json", "sha": _sha(b"asm")},
            {"file": "sc3d/characters/hero/hero_valkyrie/a.json", "sha": _sha(b"valk")},
            {"file": "sc/ui.sc", "sha": _sha(b"ui")},
        ],
    }
)


def test_missing_files_lists_only_absent_or_stale_entries_under_prefix() -> None:
    content_sha, fingerprint = parse_fingerprint(FINGERPRINT)
    present = {
        "sc3d/characters/hero/hero_valkyrie/a.json": _sha(b"valk"),
        "sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json": _sha(b"old"),
    }

    wanted = missing_files(fingerprint, present, "sc3d/")

    assert content_sha == "c0ffee"
    assert list(wanted) == [
        "sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json",
        "sc3d/characters/hero/hero_knight/hero_prefab_knight_battle.json",
    ]
    assert hash_convention(fingerprint, present) == (1, 1)


def test_fetch_keeps_only_exact_sha1_matches(tmp_path) -> None:
    _, fingerprint = parse_fingerprint(FINGERPRINT)
    wanted = missing_files(fingerprint, {}, "sc3d/characters/hero/hero_knight/")
    source = {
        "sc3d/characters/hero/hero_knight/hero_prefab_knight_battle.json": b"prefab",
        "sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json": b"corrupted",
    }

    ok, failed = fetch_verified(wanted, source.__getitem__, tmp_path, log=lambda _message: None)

    assert ok == ["sc3d/characters/hero/hero_knight/hero_prefab_knight_battle.json"]
    assert failed == ["sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json"]
    assert (tmp_path / ok[0]).read_bytes() == b"prefab"
    assert not (tmp_path / failed[0]).exists()


@pytest.mark.parametrize("name", ["../evil", "/abs/path", "a\\b"])
def test_fingerprint_rejects_unsafe_paths(name: str) -> None:
    with pytest.raises(ValueError):
        parse_fingerprint(json.dumps({"files": [{"file": name, "sha": "0" * 40}]}))
