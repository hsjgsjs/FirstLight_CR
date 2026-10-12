from __future__ import annotations

import hashlib
import json

import pytest

import zipfile

from native_runner.update_assets import (
    apk_hashes,
    fetch_verified,
    hash_convention,
    missing_files,
    parse_fingerprint,
    summarize,
)


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


def test_bundled_apk_assets_are_not_missing(tmp_path) -> None:
    _, fingerprint = parse_fingerprint(FINGERPRINT)
    apk = tmp_path / "game.apk"
    with zipfile.ZipFile(apk, "w") as archive:
        archive.writestr("assets/sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json", b"asm")
        archive.writestr("assets/sc3d/characters/hero/hero_valkyrie/a.json", b"stale")
        archive.writestr("assets/sc/ui.sc", b"ui")
        archive.writestr("lib/arm64-v8a/libg.so", b"x")

    bundled = apk_hashes(apk, "sc3d/")
    wanted = missing_files(fingerprint, {}, "sc3d/", bundled)

    assert set(bundled) == {
        "sc3d/characters/hero/hero_knight/asm_hero_knight_battle.json",
        "sc3d/characters/hero/hero_valkyrie/a.json",
    }
    assert list(wanted) == [
        "sc3d/characters/hero/hero_knight/hero_prefab_knight_battle.json",
        "sc3d/characters/hero/hero_valkyrie/a.json",
    ]


def test_summarize_groups_by_directory() -> None:
    names = ["sc3d/characters/hero/a", "sc3d/characters/hero/b", "sc3d/effects/x/y", "sc3d/top.png"]
    assert summarize(names) == [("sc3d/characters/hero", 2), ("sc3d/effects/x", 1), ("sc3d/top.png", 1)]
