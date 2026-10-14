import json
from types import SimpleNamespace as NS

import pytest

from native_runner.duel import engine_diagnostics, wait_evaluations


@pytest.mark.parametrize('code, expected', [(-9, 'SIGKILL'), (-15, 'SIGTERM'), (99, '退出码 99'), (0, '退出码 0')])
def test_engine_exit_reports_original_status(tmp_path, code, expected):
    worker = {'process': NS(poll=lambda: None), 'results': tmp_path}
    engine = NS(poll=lambda: code, returncode=code)
    with pytest.raises(RuntimeError, match=expected):
        wait_evaluations([worker], engine, 8)


def test_diagnostics_preserve_exit_and_surface_guest_logs(tmp_path, capsys):
    logs = tmp_path / 'engine' / 'engine-logs-test'
    logs.mkdir(parents=True)
    (logs / 'emulator-0.log').write_text('guest failure detail\n')
    engine_diagnostics(tmp_path, NS(poll=lambda: -9))
    record = json.loads((tmp_path / 'engine-diagnostics.json').read_text())
    assert record['returncode_before_cleanup'] == -9
    assert record['signal'] == 9
    assert 'guest failure detail' in capsys.readouterr().out
