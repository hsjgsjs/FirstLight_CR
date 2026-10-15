"""Lossless, bounded live-battle archives and an offline recording inspection command."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import orjson


class LiveRecordingError(RuntimeError):
    pass


class LiveBattleRecording:
    """Freeze data on submission; compress/write on one bounded background worker.

    No records are dropped or filtered. Backpressure is measured, and storage
    errors stop recording explicitly instead of producing a silently partial set.
    Each blob is an independent gzip member/file so a crash preserves earlier data.
    """

    def __init__(self, root, *, identity=None, queue_size=4):
        self.path = Path(root) / f"match-{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex[:6]}"
        self.path.mkdir(parents=True)
        self.manifest = dict(
            schema="firstlight-live-recording.v1", started_utc=datetime.now(timezone.utc).isoformat(),
            monotonic_origin_ns=time.monotonic_ns(), status="open", identity=identity or {},
            training_ready=False, training_note="raw archive; requires alignment and quality audit before training",
            retention="all observed games/frames/actions, including losses and errors",
        )
        self._write_manifest()
        self._events = (self.path / "events.jsonl").open("wb", buffering=0)
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self._counts = Counter()
        self._written = Counter()
        self._event_count = 0
        self.submission_ms = 0.0
        self._thread = threading.Thread(target=self._write_blobs, name="live-recording", daemon=True)
        self._thread.start()

    def _write_manifest(self):
        temporary = self.path / "manifest.json.tmp"
        temporary.write_bytes(orjson.dumps(self.manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
        temporary.replace(self.path / "manifest.json")

    def _check(self):
        if self._error is not None:
            raise LiveRecordingError(f"live recording storage failed: {self._error}") from self._error
        if self._closed:
            raise LiveRecordingError("live recording is closed")

    def event(self, record):
        self._check()
        try:
            self._events.write(orjson.dumps(dict(record, record_sequence=self._event_count + 1,
                                                monotonic_ns=time.monotonic_ns()),
                                           option=orjson.OPT_NON_STR_KEYS) + b"\n")
            self._event_count += 1
        except OSError as error:
            self._error = error
            raise LiveRecordingError(f"live recording event write failed: {error}") from error

    def update_identity(self, **identity):
        self.manifest["identity"].update(identity)
        self._write_manifest()

    def submit(self, kind, payload, **metadata):
        self._check()
        if kind not in ("capture", "policy_frame"):
            raise ValueError("unknown live recording blob kind")
        # orjson freezes the object before BattleEnv/tensorizer can reuse it.
        started = time.monotonic_ns()
        encoded = orjson.dumps(payload, option=orjson.OPT_NON_STR_KEYS)
        number = self._counts[kind]
        row = dict(kind=kind, number=number, file=f"{kind}s/{number:08d}.json.gz",
                   monotonic_ns=started, raw_bytes=len(encoded), **metadata)
        while True:
            self._check()
            try:
                self._queue.put((row, encoded), timeout=0.1)
                break
            except queue.Full:
                continue
        self._counts[kind] += 1
        elapsed = (time.monotonic_ns() - started) / 1e6
        self.submission_ms += elapsed
        self.event(dict(event="recording_submitted", **row, submit_ms=elapsed))
        return number

    def capture(self, capture, *, request_started_ns):
        return self.submit(
            "capture", capture, tick=capture.get("tick"), generation=capture.get("generation"),
            state_epoch=capture.get("stateEpoch"), request_started_ns=request_started_ns,
            response_received_ns=time.monotonic_ns(),
        )

    def _write_blobs(self):
        try:
            with (self.path / "index.jsonl").open("ab", buffering=0) as index:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    row, encoded = item
                    target = self.path / row["file"]
                    target.parent.mkdir(exist_ok=True)
                    compressed = gzip.compress(encoded, compresslevel=1, mtime=0)
                    temporary = target.with_suffix(".tmp")
                    temporary.write_bytes(compressed)
                    temporary.replace(target)
                    row = dict(row, compressed_bytes=len(compressed),
                               compressed_sha256=hashlib.sha256(compressed).hexdigest())
                    index.write(orjson.dumps(row) + b"\n")
                    self._written[row["kind"]] += 1
        except BaseException as error:
            self._error = error

    def close(self, summary=None):
        if self._closed:
            return
        try:
            while self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
            self._check()
        finally:
            self._closed = True
            self._events.close()
            self.manifest.update(
                status="error" if self._error else "closed", ended_utc=datetime.now(timezone.utc).isoformat(),
                submitted=dict(self._counts), written=dict(self._written), events=self._event_count,
                recording_submit_ms=self.submission_ms, summary=summary or {},
                error=str(self._error) if self._error else None,
            )
            self._write_manifest()

    def screenshot(self, name, png):
        self._check()
        if name not in {"start", "end"}:
            raise ValueError("unknown recording screenshot")
        (self.path / f"{name}.png").write_bytes(png)
        self.event(dict(event="screenshot", file=f"{name}.png", bytes=len(png),
                        note="debug image; separate from atomic capture, not a synchronized training frame"))


def inspect_recording(directory, *, verify=False):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest.get("schema") != "firstlight-live-recording.v1":
        raise ValueError("unsupported live recording")
    problems = []

    def read_rows(name):
        path = root / name
        if not path.is_file():
            problems.append(f"missing index/log: {name}")
            return []
        rows = []
        for n, line in enumerate(path.read_bytes().splitlines(), 1):
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                problems.append(f"invalid/incomplete JSON line: {name}:{n}")
        return rows

    index = read_rows("index.jsonl")
    events = read_rows("events.jsonl")
    counts = Counter(row["kind"] for row in index)
    if manifest.get("status") != "closed":
        problems.append(f"recording status: {manifest.get('status')}")
    if dict(counts) != manifest.get("submitted") or dict(counts) != manifest.get("written"):
        problems.append("submitted/written/index counts differ")
    if len(events) != manifest.get("events"):
        problems.append("event count differs")
    last = {}
    tick_gaps = []
    for row in index:
        path = root / row["file"]
        if not path.is_file():
            problems.append(f"missing blob: {row['file']}")
            continue
        if verify:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["compressed_sha256"]:
                problems.append(f"blob hash mismatch: {row['file']}")
            else:
                try:
                    payload = json.loads(gzip.decompress(raw))
                except (OSError, EOFError, ValueError):
                    problems.append(f"blob cannot be decoded: {row['file']}")
                    continue
                if row["kind"] == "capture" and (payload.get("tick"), payload.get("generation")) != (
                    row["tick"], row["generation"]
                ):
                    problems.append(f"capture identity mismatch: {row['file']}")
        if row["kind"] == "capture":
            generation, tick = row["generation"], row["tick"]
            if not isinstance(generation, int) or not isinstance(tick, int):
                problems.append(f"invalid raw capture identity: {row['file']}")
                continue
            if generation in last:
                gap = tick - last[generation]
                tick_gaps.append(gap)
                if gap < 0:
                    problems.append("capture tick moved backwards")
            last[generation] = tick
    flags = Counter(row["event"] for row in events if row.get("event") in {
        "tap_skipped", "tap_lost", "step_rejected", "contract_rebuild", "deployment_timing_unresolved",
    })
    latency = [row["latencyMs"] for row in events if row.get("event") == "deployment_spawn_latency"]
    submission = [row["submit_ms"] for row in events if row.get("event") == "recording_submitted"]
    return dict(
        directory=str(root), status=manifest["status"], blobs=dict(counts), events=len(events),
        quality_flags=dict(flags), problems=problems, identity=manifest["identity"],
        max_capture_gap_ticks=max(tick_gaps, default=0), deployment_latency_ms=latency,
        recording_submit_ms=manifest.get("recording_submit_ms"),
        max_recording_submit_ms=max(submission, default=0),
        summary=manifest.get("summary"), training_ready=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify", action="store_true", help="also verify/decompress all saved blobs")
    args = parser.parse_args(argv)
    report = inspect_recording(args.directory, verify=args.verify)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
