"""Fill in on-demand game content that the offline engine can never download.

Hero forms render with 3D prefabs (``sc3d/characters/hero/...``).  The stock
client only downloads those prefab/ASM ``.json`` files on demand, so after the
initial lobby update most of them are listed in ``update/fingerprint.json``
but absent on disk.  The offline engine is network-isolated, so the Hero body
silently fails to instantiate (logic, ability and projectiles still work).

This tool compares the device's update directory with its own fingerprint,
fetches the missing files from a source you choose, verifies every file
against the fingerprint SHA-1, and installs them with the app's ownership.

    python -m native_runner.update_assets status
    python -m native_runner.update_assets sync --from-serial 127.0.0.1:16416
    python -m native_runner.update_assets sync --from-dir D:\\cr-update
    python -m native_runner.update_assets sync --content-url https://<cdn-root>

Sources:
  --from-serial  another (rooted) MuMu instance or phone running the ORIGINAL
                 online game after it has cached the assets (open the Heroes
                 in the card collection / play a battle with them online).
  --from-dir     a local copy of an update directory (same relative paths).
  --content-url  the game's content CDN root; files are requested as
                 <root>/<fingerprint sha>/<path>, the usual Supercell layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Callable, Iterable, Mapping
import urllib.request
import uuid
import zipfile

from .local_config import setting
from .offline_install import APP, DATA, Device
from .paths import REPOSITORY_ROOT, WORKSPACE_ROOT

UPDATE = f"{DATA}/update"
DEFAULT_PREFIX = "sc3d/"
STAGING_ROOT = WORKSPACE_ROOT / "runs" / "asset-sync"
DEFAULT_APK = REPOSITORY_ROOT / "user-provided" / "nulls-royale.apk"


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not name:
        raise ValueError(f"unsafe fingerprint path: {name!r}")
    return str(path)


def parse_fingerprint(text: str) -> tuple[str, dict[str, str]]:
    """Return (content sha, {relative file: sha1}) from a fingerprint.json."""

    payload = json.loads(text)
    files: dict[str, str] = {}
    for entry in payload.get("files", ()):
        sha = str(entry["sha"]).lower()
        if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError(f"fingerprint entry has an invalid sha: {entry!r}")
        files[_relative(str(entry["file"]))] = sha
    return str(payload.get("sha", "")), files


def _read_bytes(device: Device, path: str) -> bytes:
    result = subprocess.run(device.command + ["exec-out", "cat", path], capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or f"cannot read {path}")
    return result.stdout


def device_fingerprint(device: Device) -> tuple[str, dict[str, str]]:
    return parse_fingerprint(_read_bytes(device, f"{UPDATE}/fingerprint.json").decode("utf-8"))


def device_hashes(device: Device, prefix: str) -> dict[str, str]:
    """SHA-1 of every file the device has under ``prefix`` (relative to update/)."""

    top = prefix.strip("/").split("/", 1)[0] or "."
    script = (
        f"cd {shlex.quote(UPDATE)} && "
        f"if [ -e {shlex.quote(top)} ]; then find {shlex.quote(top)} -type f -exec sha1sum {{}} +; fi"
    )
    hashes: dict[str, str] = {}
    for line in device.run("shell", script).splitlines():
        digest, _, name = line.strip().partition(" ")
        name = name.strip().removeprefix("./")
        if len(digest) == 40 and name.startswith(prefix.lstrip("/")):
            hashes[name] = digest.lower()
    return hashes


def apk_hashes(apk: Path, prefix: str) -> dict[str, str]:
    """SHA-1 of the APK's bundled ``assets/<prefix>`` files, keyed like update/.

    The client falls back to these when update/ has no copy, so an identical
    bundled file is not missing.
    """

    prefix = prefix.lstrip("/")
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(apk) as archive:
        for item in archive.infolist():
            name = item.filename.removeprefix("assets/")
            if item.is_dir() or not item.filename.startswith("assets/") or not name.startswith(prefix):
                continue
            digest = hashlib.sha1()
            with archive.open(item) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
    return hashes


def missing_files(
    fingerprint: Mapping[str, str],
    present: Mapping[str, str],
    prefix: str,
    bundled: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Fingerprint entries under ``prefix`` neither on the device nor bundled identically."""

    prefix = prefix.lstrip("/")
    bundled = bundled or {}
    return {
        name: sha
        for name, sha in sorted(fingerprint.items())
        if name.startswith(prefix) and present.get(name) != sha and bundled.get(name) != sha
    }


def summarize(names: Iterable[str], depth: int = 3) -> list[tuple[str, int]]:
    """Count names per leading ``depth`` path components, largest first."""

    counts: dict[str, int] = {}
    for name in names:
        key = "/".join(name.split("/")[:depth])
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def hash_convention(fingerprint: Mapping[str, str], present: Mapping[str, str]) -> tuple[int, int]:
    """(matching, differing) counts for files present on both sides.

    A high matching count proves the fingerprint sha is the SHA-1 of the file
    exactly as stored in update/, which is what this tool verifies against.
    """

    shared = [name for name in present if name in fingerprint]
    matching = sum(present[name] == fingerprint[name] for name in shared)
    return matching, len(shared) - matching


Fetcher = Callable[[str], bytes]


def fetch_verified(wanted: Mapping[str, str], fetch: Fetcher, staging: Path,
                   log: Callable[[str], None] = print) -> tuple[list[str], list[str]]:
    """Fetch each wanted file, keep only exact SHA-1 matches in ``staging``."""

    ok: list[str] = []
    failed: list[str] = []
    for name, sha in wanted.items():
        try:
            data = fetch(name)
        except Exception as error:  # noqa: BLE001 - every source error is reported per file
            failed.append(name)
            log(f"  缺少  {name}：{error}")
            continue
        if hashlib.sha1(data).hexdigest() != sha:
            failed.append(name)
            log(f"  校验失败 {name}：SHA-1 与 fingerprint 不一致，已丢弃")
            continue
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        ok.append(name)
    return ok, failed


def install_staged(device: Device, staging: Path) -> None:
    """Copy a staged tree into update/ and restore the app's ownership/labels."""

    uid = device.shell(f"stat -c %u {shlex.quote(DATA)}")
    if not uid.isdecimal() or int(uid) < 10000:
        raise ValueError("cannot determine the installed app UID")
    remote = f"/data/local/tmp/firstlight-assets-{uuid.uuid4().hex}"
    device.shell(f"am force-stop {APP}")
    device.run("push", str(staging), remote)
    try:
        device.shell(
            f"cp -r {remote}/. {shlex.quote(UPDATE)}/ && chown -R {uid}:{uid} {shlex.quote(UPDATE)} && "
            f"restorecon -RF {shlex.quote(UPDATE)}"
        )
    finally:
        device.run("shell", f"rm -rf {remote}", allow_failure=True)


def _source(args: argparse.Namespace, content_sha: str) -> Fetcher:
    if args.from_serial:
        source = Device(args.adb, args.from_serial)
        return lambda name: _read_bytes(source, f"{UPDATE}/{name}")
    if args.from_dir:
        root = Path(args.from_dir)
        return lambda name: (root / name).read_bytes()
    base = args.content_url.rstrip("/")
    if not content_sha:
        raise ValueError("fingerprint.json has no top-level sha for the CDN path")

    def download(name: str) -> bytes:
        with urllib.request.urlopen(f"{base}/{content_sha}/{name}", timeout=60) as response:
            return response.read()

    return download


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adb", default=setting("CR_ADB", "adb") or "adb")
    parser.add_argument("--serial", default=setting("CR_ADB_SERIAL"), help="offline instance (default: CR_ADB_SERIAL)")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="update/ subtree to check (default: sc3d/)")
    parser.add_argument(
        "--apk",
        default=setting("CR_INPUT_APK") or str(DEFAULT_APK),
        help="original APK; identical bundled assets are not counted as missing",
    )
    parser.add_argument("--all", action="store_true", help="status: list every missing file, not only a summary")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list fingerprint files missing on the offline instance")
    sync = sub.add_parser("sync", help="fetch, verify and install the missing files")
    source = sync.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-serial", help="adb serial of a device whose original game cached the files")
    source.add_argument("--from-dir", help="local directory laid out like update/")
    source.add_argument("--content-url", help="content CDN root (files at <root>/<sha>/<path>)")
    sync.add_argument("--dry-run", action="store_true", help="fetch and verify only, do not install")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.serial:
        raise SystemExit("set CR_ADB_SERIAL or pass --serial for the offline instance")
    device = Device(args.adb, args.serial)
    content_sha, fingerprint = device_fingerprint(device)
    present = device_hashes(device, args.prefix)
    matching, differing = hash_convention(fingerprint, present)
    apk = Path(args.apk)
    if apk.is_file():
        bundled = apk_hashes(apk, args.prefix)
        bundled_matching = sum(bundled.get(name) == sha for name, sha in fingerprint.items() if name in bundled)
        apk_note = f"APK 内置且一致 {bundled_matching} 个"
    else:
        bundled = {}
        apk_note = f"未找到原版 APK（{apk}），APK 内置文件也会被算作缺失；用 --apk 指定"
    wanted = missing_files(fingerprint, present, args.prefix, bundled)
    print(f"fingerprint 中 {args.prefix} 下共 {sum(n.startswith(args.prefix) for n in fingerprint)} 个文件；"
          f"设备已有且一致 {matching} 个，不一致 {differing} 个；{apk_note}；真正缺失 {len(wanted)} 个。")
    if args.command == "status":
        if args.all:
            for name in wanted:
                print(f"  {name}")
        else:
            print("按目录统计（--all 列出全部）：")
            for directory, count in summarize(wanted)[:40]:
                print(f"  {count:5d}  {directory}/")
        return 0 if not wanted else 1
    if not wanted:
        print("没有需要补齐的文件。")
        return 0
    staging = STAGING_ROOT / (content_sha or "unknown")
    ok, failed = fetch_verified(wanted, _source(args, content_sha), staging)
    print(f"已取得并校验 {len(ok)} 个文件，失败 {len(failed)} 个。暂存目录：{staging}")
    if args.dry_run or not ok:
        return 0 if ok and not failed else 1
    install_staged(device, staging)
    after = device_hashes(device, args.prefix)
    still_missing = missing_files(fingerprint, after, args.prefix)
    print(f"安装完成：仍缺失或不一致 {len(still_missing)} 个。请重启离线游戏后再开局。")
    return 0 if not still_missing else 1


if __name__ == "__main__":
    sys.exit(main())
