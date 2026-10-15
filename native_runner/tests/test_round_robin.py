"""Engine-free schedule, merge, resume and rating checks; runnable with unittest."""

from dataclasses import asdict
import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from native_runner.match_factory import MatchConfig
from native_runner.round_robin import collect_runs, digest, parse_blocks, report, write_json
from native_runner.training.v4.evaluation_protocol import (
    bradley_terry, complete_natural_blocks, natural_schedule, round_robin_summary,
)


class RoundRobinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = ["General 460", "Update 48", "BC 200"]
        self.match = MatchConfig(seed=1001, level_cap=16, minimum_card_level=16, king_tower_level=16)
        self.identity = dict(schema="round-robin.v1", model_order=self.names, anchor=self.names[0],
                             models={name: str(index) * 64 for index, name in enumerate(self.names)},
                             gameplay=dict(seed=1001, sample=False, delay_ticks=[18, 24],
                                           min_play_gap_ticks=[4, 8], lanes=16))

    def rows(self, blocks=(0, 1), winner="a"):
        return [dict(**case.metadata(), match=asdict(case.match), seed=case.match.seed, a_owner=case.a_owner,
                     winner=winner, terminated=True, truncated=False, game=index,
                     opening_overrides={"a": 0, "b": 0})
                for index, case in enumerate(natural_schedule(self.match, blocks))]

    def run_directory(self, name, blocks=(0, 1), winner="a"):
        directory = self.root / name
        directory.mkdir()
        write_json(directory / "study.json", self.identity)
        shard = directory / "shard-00"
        shard.mkdir()
        write_json(shard / "engine.json", dict(production_ready=True, probe_sha256="engine"))
        for index, pair in enumerate(itertools.combinations(self.names, 2)):
            destination = shard / f"pair-{index:02d}"
            destination.mkdir()
            write_json(destination / "pair.json", dict(models=pair, blocks=list(blocks)))
            write_json(destination / "identity.json", dict(
                checkpoints=[dict(sha256=self.identity["models"][name]) for name in pair],
                sample=False, delay_a_ticks=[18, 24], delay_b_ticks=[18, 24],
                min_play_gap_ticks=[4, 8], resident_lanes=16, transport="resident-compact",
                schedule=[asdict(case) for case in natural_schedule(self.match, blocks)]))
            self.write_rows(destination / "matches.jsonl", self.rows(blocks, winner))
        return directory

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_four_natural_variants_swap_every_deck_component(self):
        cases = natural_schedule(self.match, [8])
        self.assertEqual(len(cases), 4)
        self.assertEqual([(case.a_owner, case.a_deck) for case in cases], [(0, 0), (1, 0), (0, 1), (1, 1)])
        self.assertTrue(all(case.condition == "natural" and case.force_b_seconds is None for case in cases))
        self.assertEqual(cases[0].match.deck0, cases[1].match.deck1)
        self.assertEqual(cases[0].match.tower_troop0_id, cases[1].match.tower_troop1_id)

    def test_deck_sampling_random_access_matches_static_shards(self):
        pool = SimpleNamespace(decks=[SimpleNamespace(
            uses=i + 1, deck_id=f"deck-{i}", card_ids=tuple(range(26000000 + i * 8, 26000008 + i * 8)),
            form_availability=(0, 1, 2, 0, 0, 0, 0, 0), tower_troop_id=159000000)
            for i in range(5)])
        state = random.getstate()
        all_cases = natural_schedule(self.match, range(100), deck_pool=pool)
        shards = natural_schedule(self.match, range(80), deck_pool=pool) + natural_schedule(
            self.match, range(80, 100), deck_pool=pool)
        self.assertEqual(all_cases, shards)
        self.assertEqual(all_cases[83 * 4:84 * 4], natural_schedule(self.match, [83], deck_pool=pool))
        self.assertEqual(state, random.getstate())

    def test_range_is_inclusive_and_rejects_overlap(self):
        self.assertEqual(parse_blocks("80-82,90"), [80, 81, 82, 90])
        for value in ("2-1", "-1", "0-2,2", "bad"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_blocks(value)

    def test_partial_and_truncated_groups_are_not_complete(self):
        rows = self.rows()
        rows[-1]["truncated"] = True
        self.assertEqual(set(complete_natural_blocks(rows)), {0})
        self.assertEqual(complete_natural_blocks(rows[:3]), {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            complete_natural_blocks(rows + rows[:1])

    def test_shard_merge_equals_unsplit_summary_and_copy_is_deduplicated(self):
        first = self.run_directory("rr-first", blocks=(0, 1))
        second = self.run_directory("rr-second", blocks=(2, 3))
        copy = self.run_directory("rr-copy", blocks=(0, 1))
        identity, pairs, counts = collect_runs([first, second, copy])
        self.assertEqual(identity, self.identity)
        self.assertEqual(counts["duplicate_blocks"], 6)
        self.assertTrue(all(set(groups) == {0, 1, 2, 3} for groups in pairs.values()))
        unsplit = self.run_directory("rr-all", blocks=(0, 1, 2, 3))
        _, full, _ = collect_runs([unsplit])
        split_summary = round_robin_summary(pairs, self.names, anchor=self.names[0], bootstrap_samples=20)
        self.assertEqual(split_summary, round_robin_summary(full, self.names, anchor=self.names[0], bootstrap_samples=20))

    def test_resume_ignores_incomplete_attempt_and_recovers_full_retry(self):
        partial = self.run_directory("rr-partial")
        path = partial / "shard-00/pair-00/matches.jsonl"
        self.write_rows(path, self.rows()[:-1])
        with path.open("a") as stream:
            stream.write('{"block":')
        _, groups, counts = collect_runs([partial])
        self.assertEqual(set(groups[(self.names[0], self.names[1])]), {0})
        self.assertEqual(counts["excluded_incomplete_blocks"], 1)
        retry = self.run_directory("rr-retry", blocks=(1,))
        _, groups, _ = collect_runs([partial, retry])
        self.assertTrue(all(set(blocks) == {0, 1} for blocks in groups.values()))

    def test_mismatched_study_or_engine_is_rejected(self):
        first = self.run_directory("rr-first")
        other = self.run_directory("rr-other")
        changed = {**self.identity, "models": {**self.identity["models"], self.names[1]: "f" * 64}}
        write_json(other / "study.json", changed)
        with self.assertRaisesRegex(ValueError, "身份不一致"):
            collect_runs([first, other])
        write_json(other / "study.json", self.identity)
        write_json(other / "shard-00/engine.json", dict(production_ready=True, probe_sha256="different"))
        with self.assertRaisesRegex(ValueError, "引擎"):
            collect_runs([first, other])

    def test_conflicting_results_and_changed_effective_settings_are_rejected(self):
        first = self.run_directory("rr-first")
        other = self.run_directory("rr-other", winner="b")
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            collect_runs([first, other])
        marker = other / "shard-00/pair-00/identity.json"
        settings = json.loads(marker.read_text())
        settings["delay_a_ticks"] = [21, 21]
        write_json(marker, settings)
        with self.assertRaisesRegex(ValueError, "settings mismatch"):
            collect_runs([other])

    def test_changed_deck_across_pairs_is_rejected(self):
        directory = self.run_directory("rr-first")
        path = directory / "shard-00/pair-01/matches.jsonl"
        rows = self.rows()
        rows[0]["match"]["arena"] += 1
        self.write_rows(path, rows)
        with self.assertRaises(ValueError):
            collect_runs([directory])

    def test_known_odds_and_extreme_results_produce_finite_ratings(self):
        ratings = bradley_terry(["a", "b"], [("a", "b", 300, 400)], anchor="a")
        self.assertEqual(ratings["a"], 0)
        self.assertAlmostEqual(ratings["b"], -400 * __import__('math').log10(3), delta=1)
        ratings = bradley_terry(["a", "b"], [("a", "b", 400, 400)], anchor="a")
        self.assertTrue(__import__('math').isfinite(ratings["b"]))
        with self.assertRaisesRegex(ValueError, "disconnected"):
            bradley_terry(["a", "b", "c"], [("a", "b", 1, 2)], anchor="a")

    def test_bootstrap_is_reproducible_and_uses_whole_correlated_blocks(self):
        blocks = {0: self.rows((0,), "a"), 1: self.rows((1,), "b")}
        pairs = {pair: blocks for pair in itertools.combinations(self.names, 2)}
        summary = round_robin_summary(pairs, self.names, anchor=self.names[0], bootstrap_samples=80, seed=9)
        self.assertEqual(summary, round_robin_summary(pairs, self.names, anchor=self.names[0],
                                                      bootstrap_samples=80, seed=9))
        for pair in summary["pairs"]:
            self.assertEqual(pair["a_score"], .5)
            self.assertEqual(pair["a_score_95"], [0, 1])
        self.assertEqual(summary["shared_rating_blocks"], 2)
        anchor = next(row for row in summary["ratings"] if row["model"] == self.names[0])
        self.assertEqual(anchor["elo_95"], [0, 0])

    def test_rating_requires_same_global_blocks_across_all_pairs(self):
        pairs = {pair: {0: self.rows((0,))} for pair in itertools.combinations(self.names, 2)}
        pairs[(self.names[1], self.names[2])] = {1: self.rows((1,))}
        summary = round_robin_summary(pairs, self.names, anchor=self.names[0], bootstrap_samples=5)
        self.assertIsNone(summary["ratings"])
        self.assertEqual(summary["shared_rating_blocks"], 0)

    def test_report_produces_portable_raw_data_and_rating_output(self):
        directory = self.run_directory("rr-source")
        output = self.root / "report"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = report([directory], output, samples=10)
        self.assertEqual(summary["shared_rating_blocks"], 2)
        self.assertEqual(len((output / "matches.jsonl").read_text().splitlines()), 24)
        self.assertIn("Elo", (output / "report.md").read_text())
        self.assertTrue((output / "summary.json").is_file())

    def test_worker_runs_all_ten_pairs_once_on_one_engine_with_16_lanes_and_10_threads(self):
        from native_runner.training.v4.evaluate_round_robin import main
        names = ["General 460", "Update 48", "PPO 104", "PPO 1208", "BC 200"]
        models = {}
        for name in names:
            path = self.root / (name + ".pt")
            path.write_bytes(name.encode())
            models[name] = str(path)
        pool_path = self.root / "pool.json"
        pool_path.write_text("pool")
        options = dict(models=models, deck_pool=str(pool_path), match_config=None, battle_level=16, seed=1001,
                       device="cpu", cpu_threads=10, lanes=16, sample=False, gate_temperature=None,
                       delay_ticks=[18, 24], min_play_gap_ticks=[4, 8])
        study = dict(models={name: digest(Path(path)) for name, path in models.items()},
                     deck_pool_sha256=digest(pool_path))
        run = self.root / "rr-worker"
        run.mkdir()
        shard = run / "shard-00"
        shard.mkdir()
        write_json(run / "settings.json", options)
        write_json(run / "study.json", study)
        pairs = list(itertools.combinations(names, 2))
        write_json(shard / "work.json", [dict(models=pair, blocks=[80, 81, 82, 83]) for pair in pairs])
        calls, threads, connections = [], [], []
        torch = ModuleType("torch")
        torch.device = lambda value: SimpleNamespace(type=value)
        torch.set_num_threads = threads.append
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        native = ModuleType("native_runner.cr_native_env")
        def connect(*args, **kwargs):
            connections.append(args)
            return SimpleNamespace(attest=lambda: SimpleNamespace(to_dict=lambda: dict(production_ready=True)),
                                   _request=lambda command: dict(capacity=16), close_transport=lambda: None)
        native.NativeClashEnv = connect
        pool = SimpleNamespace(decks=[SimpleNamespace(uses=10, deck_id="deck", card_ids=self.match.deck0,
                                                      form_availability=(0,) * 8, tower_troop_id=159000000)])
        decks = ModuleType("native_runner.training.v4.deck_pool")
        decks.DeckPoolManifestV4 = SimpleNamespace(load=lambda path: pool)
        evaluator = ModuleType("native_runner.training.v4.evaluate")
        def evaluate(a, b, **kwargs):
            calls.append((a, b, kwargs))
            schedule = kwargs["schedule"]
            destination = kwargs["output"]
            rows = [dict(**case.metadata(), match=asdict(case.match), seed=case.match.seed,
                         a_owner=case.a_owner, winner="a", terminated=True, truncated=False)
                    for case in schedule]
            self.write_rows(destination / "matches.jsonl", rows)
            write_json(destination / "identity.json", dict(checkpoints=[dict(sha256=digest(Path(path)))
                                                                        for path in (a, b)]))
        evaluator.evaluate = evaluate
        with patch.dict(sys.modules, {"torch": torch, "native_runner.cr_native_env": native,
                                      "native_runner.training.v4.deck_pool": decks,
                                      "native_runner.training.v4.evaluate": evaluator}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--run", str(run), "--prepare-only"]), 0)
                self.assertFalse(connections)
                self.assertEqual(main(["--run", str(run), "--shard", str(shard)]), 0)
        self.assertEqual(threads, [10])
        self.assertEqual(connections, [("127.0.0.1", 28000)])
        self.assertEqual(len(calls), 10)
        for (a, b, kwargs), pair in zip(calls, pairs):
            self.assertEqual((a, b), tuple(models[name] for name in pair))
            self.assertEqual(kwargs["resident_lanes"], 16)
            self.assertEqual(kwargs["delay_a_ticks"], (18, 24))
            self.assertFalse(kwargs["sample"])
            self.assertEqual(len(kwargs["schedule"]), 16)
            self.assertEqual({case.block for case in kwargs["schedule"]}, {80, 81, 82, 83})
            self.assertEqual(kwargs["schedule"], calls[0][2]["schedule"])


if __name__ == "__main__":
    unittest.main()
