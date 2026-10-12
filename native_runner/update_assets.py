"""Fill in on-demand game content that the offline engine can never download.

Note: invisible Hero bodies on MuMu are a renderer problem, not missing
files; switch the MuMu instance's graphics renderer to Vulkan.  This tool
only compares update/ with its fingerprint and fills real gaps.

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
    python -m native_runner.update_assets sync --auto     # find the CDN itself

Sources:
  --from-serial  another (rooted) MuMu instance or phone running the ORIGINAL
                 online game after it has cached the assets (open the Heroes
                 in the card collection / play a battle with them online).
  --from-dir     a local copy of an update directory (same relative paths).
  --content-url  the game's content CDN root; files are requested as
                 <root>/<fingerprint sha>/<path>, the usual Supercell layout.
  --auto         discover that CDN root: URLs in the original APK's code, in
                 the original game's saved data on the device and in logcat
                 are tried until one serves this exact fingerprint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
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
CONTENT_URL_CACHE = STAGING_ROOT / "content_url.txt"
USER_AGENT = "Mozilla/5.0 (Linux; Android 12) FirstLight-asset-sync"
# Tried after every discovered candidate; harmless if it does not serve this
# fingerprint because each root must return the exact fingerprint sha.
KNOWN_CONTENT_ROOTS = ("https://game-assets.clashroyaleapp.com",)
_URL = re.compile(rb"https?://[A-Za-z0-9.-]+(?::[0-9]{1,5})?(?:/[A-Za-z0-9._~%!$&'()*+,;=:@/-]*)?")
_SHA_SEGMENT = re.compile(r"/[0-9a-fA-F]{40}(?=/|$)")
_IGNORED_HOSTS = ("schemas.android.com", "www.w3.org", "ns.adobe.com", "xml.org", "apache.org", "google.com",
                  "googleapis.com", "android.com", "apple.com", "facebook.com", "crashlytics.com")


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


def urls_in(data: bytes) -> list[str]:
    """Distinct http(s) URLs embedded in binary or text data, in order."""

    seen: dict[str, None] = {}
    for match in _URL.finditer(data):
        seen.setdefault(match.group().decode("ascii", "ignore").rstrip(".,;:'\")"), None)
    return list(seen)


def candidate_roots(urls: Iterable[str]) -> list[str]:
    """Possible content roots: the part before a 40-hex version segment, or
    every path prefix of a URL that has none."""

    roots: dict[str, None] = {}
    for url in urls:
        scheme, _, rest = url.partition("://")
        host, _, path = rest.partition("/")
        if not host or any(host == item or host.endswith("." + item) for item in _IGNORED_HOSTS):
            continue
        path = path.split("?", 1)[0].split("#", 1)[0]
        versioned = _SHA_SEGMENT.search("/" + path)
        if versioned:
            roots.setdefault(f"{scheme}://{host}" + ("/" + path)[: versioned.start()].rstrip("/"), None)
            continue
        parts = [part for part in path.split("/") if part]
        if parts and "." in parts[-1]:
            parts = parts[:-1]  # drop a file name
        for length in range(len(parts), -1, -1):
            roots.setdefault(f"{scheme}://{host}" + "".join("/" + part for part in parts[:length]), None)
    return list(roots)


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# Candidate probes must fail fast: an unreachable host otherwise blocks for
# the full download timeout and the search looks frozen.
PROBE_TIMEOUT_SECONDS = 5.0
PROBE_WORKERS = 8


def _probe_get(url: str) -> bytes:
    return _http_get(url, timeout=PROBE_TIMEOUT_SECONDS)


def root_serves(root: str, content_sha: str, get: Callable[[str], bytes] = _probe_get) -> bool:
    """True when ``root`` serves this exact content version's fingerprint."""

    try:
        served_sha, _ = parse_fingerprint(get(f"{root}/{content_sha}/fingerprint.json").decode("utf-8"))
    except Exception:  # noqa: BLE001 - any network/format failure just rejects the candidate
        return False
    return served_sha == content_sha


def apk_urls(apk: Path) -> list[str]:
    """URLs compiled into the original APK's native library and dex code."""

    found: list[str] = []
    with zipfile.ZipFile(apk) as archive:
        for item in archive.infolist():
            name = item.filename
            if name.endswith(".so") and name.startswith("lib/arm64-v8a/") or (
                name.startswith("classes") and name.endswith(".dex")
            ):
                found.extend(urls_in(archive.read(item)))
    return found


def device_urls(device: Device) -> list[str]:
    """URLs the original game left in its saved data (update/ excluded) and logcat."""

    found: list[str] = []
    for script in (
        f"cd {shlex.quote(DATA)} && tar -cf - $(ls -A | grep -v '^update$') 2>/dev/null",
        f"cd /data/user_de/0/{APP} 2>/dev/null && tar -cf - . 2>/dev/null",
        "logcat -d 2>/dev/null",
    ):
        result = subprocess.run(device.command + ["exec-out", script], capture_output=True, timeout=300)
        found.extend(urls_in(result.stdout))
    return found


def discover_content_root(
    content_sha: str,
    sources: Iterable[tuple[str, Callable[[], list[str]]]],
    get: Callable[[str], bytes] = _probe_get,
    log: Callable[[str], None] = print,
    limit: int = 200,
) -> str | None:
    """Try candidate roots from each source; return one that serves this exact
    fingerprint.  Candidates of a source are probed concurrently."""

    tried: set[str] = set()
    for label, collect in sources:
        try:
            urls = collect()
        except Exception as error:  # noqa: BLE001 - a missing source just contributes nothing
            log(f"  {label}：读取失败（{error}）")
            continue
        roots = [root for root in candidate_roots(urls) if root not in tried][: max(0, limit - len(tried))]
        log(f"  {label}：{len(urls)} 个网址，{len(roots)} 个候选根地址")
        for url in urls:
            log(f"      网址 {url}")
        tried.update(roots)
        if not roots:
            continue
        with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
            futures = {pool.submit(root_serves, root, content_sha, get): root for root in roots}
            for future in as_completed(futures):
                root = futures[future]
                if future.result():
                    log(f"    ✓ {root}")
                    for other in futures:
                        other.cancel()
                    return root
                log(f"    ✗ {root}")
    return None


def _discover(args: argparse.Namespace, device: Device, content_sha: str) -> str:
    if not content_sha:
        raise ValueError("fingerprint.json has no top-level sha for the CDN path")
    if CONTENT_URL_CACHE.is_file():
        cached = CONTENT_URL_CACHE.read_text(encoding="utf-8").strip()
        if cached and root_serves(cached, content_sha):
            print(f"使用已找到的资源地址：{cached}")
            return cached
    apk = Path(args.apk)
    print("正在寻找游戏资源下载地址（CDN）…")
    root = discover_content_root(
        content_sha,
        (
            ("设备上原版游戏的数据和日志", lambda: device_urls(device)),
            ("原版 APK 的程序文件", lambda: apk_urls(apk) if apk.is_file() else []),
            ("已知地址", lambda: list(KNOWN_CONTENT_ROOTS)),
        ),
    )
    if root is None:
        raise SystemExit(
            "没有找到能提供这个资源版本的地址。可以让原版游戏联网更新一次后再试，"
            "或改用 --from-serial / --from-dir。"
        )
    CONTENT_URL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CONTENT_URL_CACHE.write_text(root + "\n", encoding="utf-8")
    print(f"找到资源地址：{root}")
    return root


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
        return _http_get(f"{base}/{content_sha}/{name}")

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
    sub.add_parser("discover", help="find the content CDN root that serves this fingerprint")
    sync = sub.add_parser("sync", help="fetch, verify and install the missing files")
    source = sync.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-serial", help="adb serial of a device whose original game cached the files")
    source.add_argument("--from-dir", help="local directory laid out like update/")
    source.add_argument("--content-url", help="content CDN root (files at <root>/<sha>/<path>)")
    source.add_argument("--auto", action="store_true", help="discover the content CDN root, then download")
    sync.add_argument("--dry-run", action="store_true", help="fetch and verify only, do not install")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.serial:
        raise SystemExit("set CR_ADB_SERIAL or pass --serial for the offline instance")
    device = Device(args.adb, args.serial)
    content_sha, fingerprint = device_fingerprint(device)
    if args.command == "discover":
        _discover(args, device, content_sha)
        return 0
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
    if getattr(args, "auto", False):
        args.content_url = _discover(args, device, content_sha)
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
