"""Engine-free checks of which checkpoints the training curve evaluates; runnable with unittest."""

import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from native_runner import eval_curve, round_robin
from native_runner.round_robin import write_json
from native_runner.training.v4.evaluation_protocol import natural_schedule


class EvalCurveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, stage, update):
        path = self.root / "runs" / "arm" / stage / "checkpoints" / f"update-{update:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{stage}-{update}".encode())
        return path

    def test_parse_orders_by_stage_then_update(self):
        found = eval_curve.parse_checkpoints(
            [
                "/r/train-20261016-163245/checkpoints/update-00000016.pt",
                "/r/train-20261016-124143/checkpoints/update-00000200.pt",
                "/r/train-20261016-163245/checkpoints/update-00000008.pt",
                "/r/train-20261016-163245/checkpoints/update-00000008.pt.tmp",
                "/r/train-20261016-163245/metrics.jsonl",
            ]
        )
        self.assertEqual(
            [(stage[-6:], update) for stage, update, _ in found], [("124143", 200), ("163245", 8), ("163245", 16)]
        )

    def test_choose_newest_unevaluated_with_spacing(self):
        checkpoints = [("train-a", 8, "x"), ("train-a", 16, "y")]
        self.assertEqual(eval_curve.choose(checkpoints, set(), None, 0)[:2], ("train-a", 16))
        self.assertIsNone(eval_curve.choose(checkpoints, {"train-a/update-16"}, None, 0))
        self.assertIsNone(eval_curve.choose(checkpoints, set(), ("train-a", 8), 16))
        self.assertEqual(eval_curve.choose(checkpoints, set(), ("train-a", 8), 8)[:2], ("train-a", 16))
        # A new stage restarts the update count, so spacing applies within one stage only.
        self.assertEqual(eval_curve.choose([("train-b", 8, "z")], set(), ("train-a", 16), 16)[:2], ("train-b", 8))
        self.assertIsNone(eval_curve.choose([], set(), None, 0))

    def test_fetch_keeps_a_private_copy_and_drops_stale_ones(self):
        source = self.checkpoint("train-a", 8)
        destination = self.root / "curve" / "checkpoints" / "train-a" / "update-00000008.pt"
        copy = eval_curve.fetch(str(self.root / "runs" / "arm"), str(source), destination)
        self.assertEqual(copy.read_bytes(), source.read_bytes())
        source.unlink()  # the trainer pruned it; the copy survives
        newer = self.checkpoint("train-a", 16)
        second = eval_curve.fetch(
            str(self.root / "runs" / "arm"),
            str(newer),
            self.root / "curve" / "checkpoints" / "train-a" / "update-00000016.pt",
        )
        self.assertTrue(second.is_file())
        self.assertFalse(copy.exists())

    def test_once_evaluates_baseline_then_newest_checkpoint(self):
        self.checkpoint("train-20261016-124143", 200)
        self.checkpoint("train-20261016-163245", 8)
        self.checkpoint("train-20261016-163245", 16)
        opponent = self.root / "general.pt"
        opponent.write_bytes(b"general")
        baseline = self.root / "start.pt"
        baseline.write_bytes(b"start")
        calls = []

        def fake_evaluate(args, label, checkpoint, curve, rows):
            calls.append((label, Path(checkpoint).read_bytes()))
            rows.append({"label": label})
            return True

        with patch.object(eval_curve, "ROOT", self.root), patch.object(eval_curve, "evaluate", fake_evaluate):
            code = eval_curve.main(
                [str(self.root / "runs" / "arm"), "--once", "--opponent", str(opponent), "--baseline", str(baseline)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            calls, [("baseline/start", b"start"), ("train-20261016-163245/update-16", b"train-20261016-163245-16")]
        )
        self.assertEqual(list((self.root / "evaluations" / "curve-arm" / "checkpoints").glob("*/*.pt")), [])

    def test_point_result_scores_the_checkpoint_not_the_opponent(self):
        (self.root / "native_runner").mkdir()
        shutil.copy(round_robin.ROOT / "round_robin.example.json", self.root)
        for name in ("candidate.pt", "general.pt", "pool.json"):
            (self.root / name).write_bytes(name.encode())
        args = type(
            "Args",
            (),
            {
                "opponent": self.root / "general.pt",
                "opponent_name": "General 460",
                "gate_temperature": 0.2,
                "cpu_threads": None,
                "lanes": None,
            },
        )
        config = self.root / "point.json"
        write_json(
            config,
            {
                **eval_curve.point_config(args, "train-a/update-16", self.root / "candidate.pt"),
                "deck_pool": str(self.root / "pool.json"),
            },
        )
        with (
            patch.object(round_robin, "ROOT", self.root),
            patch.object(eval_curve, "ROOT", self.root),
            patch.object(round_robin, "load_env", lambda: None),
        ):
            self.assertIsNone(eval_curve.point_result(config, "train-a/update-16", {0, 1}))
            options = round_robin.read_options(type("A", (), {"config": config}))
            identity = round_robin.study_identity(options)
            pair = identity["model_order"]
            run = self.root / "evaluations" / "rr-x"
            destination = run / "shard-00" / "pair-00"
            destination.mkdir(parents=True)
            write_json(run / "study.json", identity)
            write_json(run / "shard-00" / "engine.json", {"probe_sha256": "engine"})
            write_json(destination / "pair.json", {"models": pair, "blocks": [0, 1]})
            cases = natural_schedule(round_robin.base_match(options), [0, 1])
            gameplay = identity["gameplay"]
            write_json(
                destination / "identity.json",
                {
                    "checkpoints": [{"sha256": identity["models"][name]} for name in pair],
                    "sample": True,
                    "gate_temperatures": [0.2, 0.2],
                    "delay_a_ticks": gameplay["delay_ticks"],
                    "delay_b_ticks": gameplay["delay_ticks"],
                    "min_play_gap_ticks": gameplay["min_play_gap_ticks"],
                    "resident_lanes": gameplay["lanes"],
                    "transport": "resident-compact",
                    "schedule": [asdict(case) for case in cases],
                },
            )
            # The checkpoint (model a) wins three of the eight games.
            rows = [
                dict(
                    **case.metadata(),
                    match=asdict(case.match),
                    seed=case.match.seed,
                    a_owner=case.a_owner,
                    winner="a" if index < 3 else "b",
                    terminated=True,
                    truncated=False,
                    game=index,
                    opening_overrides={"a": 0, "b": 0},
                )
                for index, case in enumerate(cases)
            ]
            (destination / "matches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = eval_curve.point_result(config, "train-a/update-16", {0, 1})
            self.assertEqual((result["blocks"], result["games"], result["a_wins"], result["b_wins"]), (2, 8, 3, 5))
            self.assertAlmostEqual(result["a_score"], 3 / 8)
            # Blocks outside the requested set are not counted.
            self.assertEqual(eval_curve.point_result(config, "train-a/update-16", {0})["games"], 4)

    def test_render_lists_points(self):
        args = type(
            "Args",
            (),
            {"name": "arm", "opponent_name": "General 460", "blocks": list(range(25)), "gate_temperature": 0.2},
        )
        text = eval_curve.render(
            [
                {
                    "label": "train-a/update-16",
                    "games": 100,
                    "wins": 40,
                    "losses": 58,
                    "draws": 2,
                    "score": 0.41,
                    "score_95": [0.33, 0.5],
                    "evaluated_at": "2026-10-16T20:00",
                }
            ],
            args,
        )
        self.assertIn("| train-a/update-16 | 100 | 40 / 58 / 2 | 41.0% | [33.0%, 50.0%] |", text)
        self.assertIn("温度 0.2", text)


if __name__ == "__main__":
    unittest.main()
