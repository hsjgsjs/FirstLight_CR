from __future__ import annotations

from pathlib import Path

from native_runner.live_diagnose import check_capture, summarize_capture
from native_runner.tests.cpp_source import cpp_function


PROBE = Path(__file__).resolve().parents[1] / "probe" / "cr_replay_probe.cpp"


def test_probe_live_mode_is_read_only_and_captures_on_the_game_thread() -> None:
    source = PROBE.read_text(encoding="utf-8")
    hook = cpp_function(source, "replay_battle_controller_full_update_hook")
    updated = cpp_function(source, "live_controller_updated")
    command = cpp_function(source, "handle_live_command")
    identity = cpp_function(source, "observation_capture_identity_locked")
    dispatch = cpp_function(source, "handle_control_command")

    # Stock (not ours) battles: rate-app flag cleared, original update first, then the live attach.
    stock_branch = hook.split("is_current_native_controller", 1)[1].split("suppress_offline_connection", 1)[0]
    assert stock_branch.index("kRateAppPendingOffset") < stock_branch.index("original(controller, delta_seconds)")
    assert stock_branch.index("original(controller, delta_seconds)") < stock_branch.index("live_controller_updated")
    # The capture is built on the game thread, after the update, under the control mutex.
    assert "build_atomic_observation_json_locked(g_live_capture_buffer" in updated
    assert "pthread_mutex_lock(&g_control_mutex)" in updated
    # Read-only: nothing in live mode queues or injects commands.
    for forbidden in ("inject_logic_command", "g_pending_native_command", "queue_native_command_and_wait"):
        assert forbidden not in updated and forbidden not in command
    assert "g_live_enabled.load" in identity and "g_live_manager.load" in identity
    assert dispatch.index("handle_live_command") < dispatch.index("handle_engine_instance_command")
    assert "live mode is on: use 'live observe-atomic'" in dispatch
    assert "& 0x00ffffffffffffffULL" in cpp_function(source, "process_range_is_readable")


def test_summary_and_validation_report_instead_of_raising() -> None:
    capture = {
        "ok": True,
        "schema": "native-observation-capture.v1",
        "atomic": True,
        "maxObjects": 256,
        "generation": 1 << 48,
        "stateEpoch": 1 << 48,
        "tick": 420,
        "ordinary": {
            "ok": True,
            "count": 0,
            "returned": 0,
            "truncated": False,
            "objects": [],
            "players": [
                {"owner": 0, "accountId": 123, "elixirRaw": 50000, "hand": [], "deck": []},
                {"owner": 1, "accountId": 456, "elixirRaw": 60000, "hand": [{}, {}, {}, {}], "deck": [{}] * 8},
            ],
        },
        "rich": {"ok": True},
    }

    summary = summarize_capture(capture)
    report = check_capture(capture)

    assert summary["tick"] == 420
    assert [player["hand"] for player in summary["players"]] == [0, 4]
    assert report["atomic"].startswith("RunnerError")
    assert "rich" in report and report["rich"] != "ok"
