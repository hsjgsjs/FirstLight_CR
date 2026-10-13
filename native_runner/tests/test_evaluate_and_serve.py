from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from native_runner.contracts import ActionV1, ObservationTier, ObservationV1
from native_runner.match_factory import MatchConfig
from native_runner.training.v4 import evaluate, serve_policy


def test_evaluation_counts_complete_matches_and_alternates_models(monkeypatch, tmp_path):
    assignments, seeds, ended = [], [], []
    terminal = NS(winner=0, to_dict=lambda: {"winner": 0, "ended": True})
    observations = {
        owner: NS(tick=100, owner=owner, players=[NS(owner=owner, elixir_exact=5.0)], events=[], terminal=terminal)
        for owner in (0, 1)
    }

    class Environment:
        def __init__(self, **kwargs):
            pass

        def reset(self, *, match_config, options):
            seeds.append(match_config.seed)
            assert options["render_mode"] == "headless"
            return observations, {}

        def step(self, commands, **kwargs):
            assert set(commands) == {0, 1}
            return observations, {}, {"__all__": True}, {"__all__": False}, {}

    def session(environment, model, *, actor_owner, **kwargs):
        assignments.append((model, actor_owner))
        return NS(
            start_episode=lambda *a, **k: None,
            decide=lambda o: NS(decoded=NS(actions=(ActionV1.wait(actor_owner, ticks=5),))),
            end_episode=lambda: ended.append(actor_owner),
        )

    monkeypatch.setattr(
        evaluate,
        "load_policy_v4",
        lambda path, **k: NS(model=path, checkpoint_path=Path(path), checkpoint_sha256="hash", checkpoint_id=path),
    )
    monkeypatch.setattr(
        evaluate,
        "production_semantic_bundle",
        lambda: NS(native_card_catalog=None, semantic_subset_contract=None, native_effect_catalog=None),
    )
    monkeypatch.setattr(evaluate.RuntimeEffectCatalog, "from_catalog", lambda c: None)
    monkeypatch.setattr(evaluate, "NativeClashEnv", lambda *a, **k: NS(close_transport=lambda: None))
    monkeypatch.setattr(evaluate, "BattleEnvV1", Environment)
    monkeypatch.setattr(evaluate, "build_policy_session_v4", session)
    result = evaluate.evaluate("a.pt", "b.pt", games=3, match=MatchConfig(seed=7), output=tmp_path)
    assert seeds == [7, 8, 9]
    assert assignments == [("a.pt", 0), ("b.pt", 1), ("b.pt", 0), ("a.pt", 1), ("a.pt", 0), ("b.pt", 1)]
    assert result == {"requested": 3, "completed": 3, "a_wins": 2, "b_wins": 1, "draws": 0, "truncated": 0}
    assert len((tmp_path / "matches.jsonl").read_text().splitlines()) == 3
    assert len(ended) == 6


def test_evaluation_delays_only_model_b_plays(monkeypatch, tmp_path):
    terminal = NS(winner=0, to_dict=lambda: {"winner": 0, "ended": True})
    observations = {
        owner: NS(tick=100, owner=owner, players=[NS(owner=owner, elixir_exact=5.0)], events=[], terminal=terminal)
        for owner in (0, 1)
    }
    sent = []

    class Environment:
        def __init__(self, **kwargs):
            pass

        def reset(self, *, match_config, options):
            return observations, {}

        def step(self, commands, **kwargs):
            sent.append(commands)
            return observations, {}, {"__all__": True}, {"__all__": False}, {}

    def session(environment, model, *, actor_owner, **kwargs):
        plays = (ActionV1.play(actor_owner, 0, (3, 20), execute_offset_ticks=1), ActionV1.wait(actor_owner, ticks=5))
        return NS(start_episode=lambda *a, **k: None, decide=lambda o: NS(decoded=NS(actions=plays)),
                  end_episode=lambda: None)

    monkeypatch.setattr(
        evaluate,
        "load_policy_v4",
        lambda path, **k: NS(model=path, checkpoint_path=Path(path), checkpoint_sha256="hash", checkpoint_id=path),
    )
    monkeypatch.setattr(
        evaluate,
        "production_semantic_bundle",
        lambda: NS(native_card_catalog=None, semantic_subset_contract=None, native_effect_catalog=None),
    )
    monkeypatch.setattr(evaluate.RuntimeEffectCatalog, "from_catalog", lambda c: None)
    monkeypatch.setattr(evaluate, "NativeClashEnv", lambda *a, **k: NS(close_transport=lambda: None))
    monkeypatch.setattr(evaluate, "BattleEnvV1", Environment)
    monkeypatch.setattr(evaluate, "build_policy_session_v4", session)
    evaluate.evaluate("a.pt", "b.pt", games=2, match=MatchConfig(seed=7), output=tmp_path, delay_b_ticks=21)
    # Game 1: A is owner 0; game 2: A is owner 1. Only B's play is delayed; waits are untouched.
    offsets = [{owner: [a.execute_offset_ticks for a in actions] for owner, actions in step.items()} for step in sent]
    assert offsets == [{0: [1, 0], 1: [21, 0]}, {0: [21, 0], 1: [1, 0]}]
    sent.clear()
    evaluate.evaluate(
        "a.pt", "b.pt", games=2, match=MatchConfig(seed=7), output=tmp_path, delay_a_ticks=19, delay_b_ticks=21
    )
    offsets = [{owner: [a.execute_offset_ticks for a in actions] for owner, actions in step.items()} for step in sent]
    assert offsets == [{0: [19, 0], 1: [21, 0]}, {0: [21, 0], 1: [19, 0]}]


def test_parallel_evaluation_splits_even_games_and_pools_results(tmp_path):
    from native_runner.training.v4.evaluate_parallel import parse_ports, pooled, split_games

    assert parse_ports("28000-28002,28005") == [28000, 28001, 28002, 28005]
    assert split_games(10, 8) == [2, 2, 2, 2, 2]
    assert split_games(11, 4) == [4, 3, 2, 2] and sum(split_games(100, 8)) == 100
    for index, rows in enumerate(([(0, "a", [10, 4])], [(1, "b", [6, 12]), (0, None, [5, 5])])):
        shard = tmp_path / f"shard-{index:02d}"
        shard.mkdir()
        shard.joinpath("matches.jsonl").write_text("".join(
            '{"a_owner": %d, "winner": %s, "terminated": true, "truncated": false, "actions_by_owner": %s, '
            '"rejected_actions": 0}\n' % (a_owner, "null" if winner is None else f'"{winner}"', actions)
            for a_owner, winner, actions in rows
        ))
    summary = pooled(tmp_path)
    assert (summary["a_wins"], summary["b_wins"], summary["draws"]) == (1, 1, 1)
    assert summary["a_actions_per_game"] == round((10 + 12 + 5) / 3, 1)
    assert summary["b_actions_per_game"] == round((4 + 6 + 5) / 3, 1)


def test_service_preserves_one_state_until_end_and_rejects_wrong_owner(monkeypatch):
    from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK

    calls = []

    class Session:
        actor_owner = 0
        tensorizer = NS(tensorize=lambda *a, **k: calls.append("observe"))

        def __init__(self, *a, **k):
            calls.append("new")

        def start_episode(self, *a, **k):
            calls.append("start")

        def decide(self, observation):
            calls.append("act")
            return NS(decoded=NS(actions=(ActionV1.wait(0, ticks=5),)), inference_ms=1.0)

        def end_episode(self):
            calls.append("end")

    monkeypatch.setattr(serve_policy.EpisodeConfigV1, "from_mapping", lambda x: x)
    monkeypatch.setattr(serve_policy, "build_episode_tensorizer_v4", lambda *a, **k: None)
    monkeypatch.setattr(serve_policy, "PolicySessionV4", Session)
    service = serve_policy.PolicyService(NS(model=None, checkpoint_id="model", checkpoint_sha256="hash"))
    obs = ObservationV1(tier=ObservationTier.FAIR, owner=0, tick=FIRST_POLICY_DECISION_TICK).to_dict()
    service.handle({"op": "start", "episode": {}, "observation": obs, "initial_elixir": {"0": 5, "1": 5}})
    assert service.handle({"op": "act", "observation": obs})["actions"][0]["owner"] == 0
    assert service.handle({"op": "act", "observation": obs})["ok"]
    with pytest.raises(ValueError, match="owner"):
        service.handle({"op": "act", "observation": {**obs, "owner": 1}})
    service.handle({"op": "end"})
    assert calls == ["new", "start", "act", "act", "end"]
    with pytest.raises(ValueError, match="start"):
        service.handle({"op": "act", "observation": obs})


def test_parallel_evaluation_gives_each_shard_one_gpu(monkeypatch, tmp_path):
    from native_runner.training.v4 import evaluate_parallel

    launched = []

    class Process:
        def __init__(self, command, env=None, **kwargs):
            launched.append((command, env))

        def wait(self):
            return 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # left over in the calling shell
    monkeypatch.setattr(evaluate_parallel.subprocess, "Popen", Process)
    evaluate_parallel.main([
        "--ports", "28000-28002", "--games", "6", "--output", str(tmp_path),
        "--devices", "cuda:0,cuda:1", "--checkpoint-a", "a.pt", "--device", "cuda", "--sample",
    ])
    assert [env["CUDA_VISIBLE_DEVICES"] for _command, env in launched] == ["0", "1", "0"]
    assert all(env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for _command, env in launched)
    for command, _env in launched:
        assert command.count("--device") == 1 and command[command.index("--device") + 1] == "cuda"
        assert "--sample" in command
    assert evaluate_parallel.shard_device("cpu") == ("cpu", None)
