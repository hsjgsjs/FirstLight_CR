from __future__ import annotations

import io
import json

import pytest

from native_runner import live_preflight


def test_checkpoint_requires_explicit_local_account_id(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["live_preflight", "capture.json.gz", "--checkpoint", "model.pt"])
    with pytest.raises(SystemExit) as error:
        live_preflight.main()
    assert error.value.code == 2
    assert "--account-id is required with --checkpoint" in capsys.readouterr().err


def test_capture_diagnostics_do_not_print_player_identity(monkeypatch, capsys):
    capture = {
        "tick": 100,
        "generation": 1,
        "ordinary": {"players": [{"owner": 0, "accountId": 12345678, "name": "private-player"}]},
    }
    monkeypatch.setattr("sys.argv", ["live_preflight", "capture.json.gz"])
    monkeypatch.setattr(live_preflight.gzip, "open", lambda *_a, **_kw: io.StringIO(json.dumps(capture)))
    live_preflight.main()
    output = capsys.readouterr().out
    assert "'owner': 0" in output
    assert "12345678" not in output
    assert "private-player" not in output
    assert "accountId" not in output
