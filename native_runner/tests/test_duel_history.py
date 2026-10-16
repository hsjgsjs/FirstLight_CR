import json
from types import SimpleNamespace

from native_runner.duel import completed_history, evaluation_key, read_options, split_blocks


def options(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'weights')
    return dict(checkpoint_a=str(checkpoint), checkpoint_b=str(checkpoint),
                match_config=None, deck_pool=None, seed=1, blocks=1, engines=1,
                cpu_threads=8, max_score_drop=.05, deterministic=False)


def write_run(parent, settings, key, *, block=0, count=8, legacy=False, shard=False):
    run = parent / 'duel-test'
    results = run / 'results'
    if shard:
        results /= 'shard-00'
    results.mkdir(parents=True)
    (run / 'settings.json').write_text(json.dumps(settings))
    if not legacy:
        (run / 'study.json').write_text(json.dumps(key))
    (results / 'identity.json').write_text(json.dumps({'checkpoints': [
        {'sha256': key['checkpoint_a']}, {'sha256': key['checkpoint_b']},
    ]}))
    rows = [dict(block=block, condition=c, variant=v, terminated=True, truncated=False)
            for c in ('natural', 'forced') for v in range(4)]
    path = results / 'matches.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows[:count]))
    return path


def test_legacy_run_is_found_when_run_size_changes(tmp_path):
    settings = options(tmp_path)
    key = evaluation_key(settings)
    write_run(tmp_path, settings, key, legacy=True)
    current = dict(settings, blocks=4, engines=2, cpu_threads=4)
    assert evaluation_key(current) == key
    assert completed_history(tmp_path, current, key) == {0}
    start = max(completed_history(tmp_path, current, key)) + 1
    assert [(start + offset, count) for offset, count in split_blocks(4, 2)] == [(1, 2), (3, 2)]


def test_partial_or_truncated_block_does_not_advance_history(tmp_path):
    settings = options(tmp_path)
    key = evaluation_key(settings)
    path = write_run(tmp_path, settings, key, count=7)
    with path.open('a') as stream:
        stream.write('{"block":')
    assert completed_history(tmp_path, settings, key) == set()
    rows = [dict(block=0, condition=c, variant=v, terminated=True, truncated=False)
            for c in ('natural', 'forced') for v in range(4)]
    rows[-1]['truncated'] = True
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert completed_history(tmp_path, settings, key) == set()


def test_checkpoint_contents_and_conditions_separate_studies(tmp_path):
    settings = options(tmp_path)
    key = evaluation_key(settings)
    write_run(tmp_path, settings, key)
    assert completed_history(tmp_path, settings, key) == {0}
    changed = dict(settings, deterministic=True)
    assert completed_history(tmp_path, changed, evaluation_key(changed)) == set()
    (tmp_path / 'model.pt').write_bytes(b'new weights')
    assert completed_history(tmp_path, settings, evaluation_key(settings)) == set()


def test_shard_history_is_recovered_without_merged_report(tmp_path):
    settings = options(tmp_path)
    key = evaluation_key(settings)
    write_run(tmp_path, settings, key, block=3, shard=True)
    assert completed_history(tmp_path, settings, key) == {3}


def test_legacy_checkpoint_overwrite_is_not_reused(tmp_path):
    settings = options(tmp_path)
    key = evaluation_key(settings)
    write_run(tmp_path, settings, key, legacy=True)
    (tmp_path / 'model.pt').write_bytes(b'new weights')
    assert completed_history(tmp_path, settings, evaluation_key(settings)) == set()


def test_legacy_null_gap_uses_current_default_and_custom_gap_is_preserved(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    settings = dict(checkpoint_a=str(checkpoint), checkpoint_b=str(checkpoint), min_play_gap_ticks=None)
    config = tmp_path / "duel.json"
    args = SimpleNamespace(checkpoint_a=None, checkpoint_b=None, blocks=None, opening_seconds=None, engines=None)
    config.write_text(json.dumps(settings))
    assert read_options(config, args)["min_play_gap_ticks"] == "6-10"
    settings["min_play_gap_ticks"] = "10-14"
    config.write_text(json.dumps(settings))
    assert read_options(config, args)["min_play_gap_ticks"] == "10-14"
