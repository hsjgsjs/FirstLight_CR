import gzip
import json
from types import SimpleNamespace

import pytest

from native_runner.live_recording import LiveBattleRecording, LiveRecordingError, inspect_recording


def test_recording_freezes_all_frames_and_preserves_losing_error_game(tmp_path):
    recorder = LiveBattleRecording(tmp_path, identity={"resolution": [1440, 2560]}, queue_size=1)
    capture = {"tick": 90, "generation": 7, "ordinary": {"objects": [{"hp": 123}]}}
    for i in range(10):
        capture["tick"] = 90 + i * 5
        assert recorder.capture(capture, request_started_ns=1) == i
    capture["ordinary"]["objects"][0]["hp"] = 0
    recorder.submit("policy_frame", {"tick": 130, "owner": 1}, tick=130, capture_id=8)
    recorder.event({"event": "tap_skipped", "reason": "slot is empty"})
    recorder.event({"event": "contract_rebuild"})
    recorder.event({"event": "deployment_spawn_latency", "latencyMs": 1500})
    recorder.screenshot("start", b"png")
    recorder.close({"result": "terminated", "winner": 0, "owner": 1})
    report = inspect_recording(recorder.path, verify=True)
    assert report["blobs"] == {"capture": 10, "policy_frame": 1}
    assert not report["problems"] and not report["training_ready"]
    assert report["quality_flags"] == {"tap_skipped": 1, "contract_rebuild": 1}
    assert report["max_capture_gap_ticks"] == 5
    assert report["deployment_latency_ms"] == [1500]
    assert report["summary"]["winner"] == 0
    saved = json.loads(gzip.decompress((recorder.path / "captures/00000000.json.gz").read_bytes()))
    assert saved["tick"] == 90 and saved["ordinary"]["objects"][0]["hp"] == 123
    # Nothing is overwritten when two matches start at the same time.
    other = LiveBattleRecording(tmp_path)
    assert other.path != recorder.path
    other.close({"result": "interrupted"})
    recorder.close()
    with pytest.raises(LiveRecordingError, match="closed"):
        recorder.event({"event": "late"})


def test_inspection_detects_missing_and_modified_data(tmp_path):
    recorder = LiveBattleRecording(tmp_path)
    recorder.capture({"tick": 100, "generation": 7}, request_started_ns=1)
    recorder.close()
    path = recorder.path / "captures/00000000.json.gz"
    path.write_bytes(b"corrupt")
    assert "hash mismatch" in inspect_recording(recorder.path, verify=True)["problems"][0]
    path.unlink()
    assert "missing blob" in inspect_recording(recorder.path)["problems"][0]


def test_interrupted_partial_log_is_reported_and_complete_records_are_read(tmp_path):
    recorder = LiveBattleRecording(tmp_path)
    recorder.capture({"tick": 100, "generation": 7}, request_started_ns=1)
    recorder.close({"result": "interrupted"})
    with (recorder.path / "index.jsonl").open("ab") as stream:
        stream.write(b'{"kind":')
    report = inspect_recording(recorder.path, verify=True)
    assert report["blobs"] == {"capture": 1}
    assert any("incomplete JSON" in p for p in report["problems"])


def test_disk_failure_closes_without_deadlocking(tmp_path, monkeypatch):
    import threading

    recorder = LiveBattleRecording(tmp_path, queue_size=1)
    attempted = threading.Event()

    def fail(*_a, **_kw):
        attempted.set()
        raise OSError("disk full")

    monkeypatch.setattr("native_runner.live_recording.gzip.compress", fail)
    recorder.capture({"tick": 100, "generation": 7}, request_started_ns=1)
    assert attempted.wait(3)
    recorder._thread.join(timeout=3)
    with pytest.raises(LiveRecordingError, match="disk full"):
        recorder.close()
    manifest = json.loads((recorder.path / "manifest.json").read_bytes())
    assert manifest["status"] == "error" and manifest["submitted"] == {"capture": 1}


def test_capture_hook_uses_existing_atomic_response_without_extra_requests(tmp_path, monkeypatch):
    from native_runner.tests.test_live_play import _Native

    native = _Native(100)
    recorder = LiveBattleRecording(tmp_path)
    native.capture_sink = recorder.capture
    calls = []
    capture = {"tick": 100, "generation": 7, "ordinary": {"players": [{"owner": 1, "hand": []}]}}

    def request(_command):
        calls.append(True)
        return capture

    monkeypatch.setattr(native, "_request", request)
    monkeypatch.setattr(native, "_validate_atomic_result", lambda result: result)
    assert native.observe_atomic() is capture
    assert native.last_capture_id == 0 and len(calls) == 1
    recorder.close()
    assert inspect_recording(recorder.path, verify=True)["blobs"] == {"capture": 1}


def test_invalid_atomic_response_is_saved_before_validation_rejects(tmp_path, monkeypatch):
    from native_runner.live_match import LiveCaptureNative
    from native_runner.cr_native_env import RunnerError

    native = LiveCaptureNative()
    recorder = LiveBattleRecording(tmp_path)
    native.capture_sink = recorder.capture
    monkeypatch.setattr(native, "_request", lambda _command: {"error": "bad capture", "tick": 90})
    with pytest.raises(RunnerError):
        native.observe_atomic()
    recorder.close({"result": "invalid_capture"})
    saved = json.loads(gzip.decompress((recorder.path / "captures/00000000.json.gz").read_bytes()))
    assert saved["error"] == "bad capture"


def test_play_battle_recording_preserves_terminal_and_closes_on_setup_failure(tmp_path, monkeypatch):
    from native_runner import live_play, live_match

    class Observation:
        tick = 95
        terminal = SimpleNamespace(to_dict=lambda: {"winner": 0})
        events = ()

        def to_dict(self):
            return {"tick": self.tick, "owner": 1, "terminal": self.terminal.to_dict()}

    class Native:
        capture_sink = None
        ordinary = {}
        last_capture_id = None
        capture_micros = 10

        def observe_atomic(self):
            capture = {"tick": 90, "generation": 7, "ordinary": {"players": [
                {"owner": 0, "accountId": 1, "deck": []}, {"owner": 1, "accountId": 2, "deck": []},
            ]}}
            self.last_capture_id = self.capture_sink(capture, request_started_ns=1)
            return capture

        def close_transport(self):
            pass

    class Episode:
        def __init__(self, native, *_a, **_kw):
            self.observation = Observation()
            self.match = SimpleNamespace(deck0_form_availability=(), deck1_form_availability=())
            self.session = SimpleNamespace(decide=lambda _obs: SimpleNamespace(
                decoded=SimpleNamespace(actions=()), inference_ms=1), end_episode=lambda: None)
            self.env = self

        def step(self, *_a, **_kw):
            return {1: self.observation}, {1: -1.0}, {"__all__": True}, {}, {}

        def deployment_timing_events(self, _owner):
            return []

        def close(self):
            pass

    monkeypatch.setattr(live_match, "LiveEpisode", Episode)
    monkeypatch.setattr(live_match, "observe_forms", lambda *_a: None)
    args = SimpleNamespace(checkpoint="model.pt", owner=1, account_id=None, device="cpu", record=True,
                           record_dir=tmp_path / "recordings")
    layout = live_play.ScreenLayout.default(1440, 2560)
    summary = live_play.play_battle(None, args, layout, None, native_factory=lambda **_kw: Native(), log_dir=tmp_path)
    report = inspect_recording(summary["recording"], verify=True)
    assert not report["problems"]
    assert report["blobs"] == {"capture": 1, "policy_frame": 2}
    assert report["identity"]["owner"] == 1 and report["summary"]["terminal"] == {"winner": 0}

    monkeypatch.setattr(live_play, "choose_seat", lambda *_a: (_ for _ in ()).throw(ValueError("seat missing")))
    with pytest.raises(ValueError, match="seat missing"):
        live_play.play_battle(None, args, layout, None, native_factory=lambda **_kw: Native(), log_dir=tmp_path)
    manifests = [json.loads(p.read_bytes()) for p in args.record_dir.glob("*/manifest.json")]
    assert len(manifests) == 2 and all(m["status"] == "closed" for m in manifests)
    assert any(m["summary"]["result"] == "setup_failed" for m in manifests)
