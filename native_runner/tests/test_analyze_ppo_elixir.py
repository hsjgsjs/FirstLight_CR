"""Portable log-analysis checks; no training runtime or GPU required."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.analyze_ppo_elixir import distribution, learner_rows, main, read_jsonl, win_summary


def match(**changes):
    result = {
        "current_owners": [0],
        "policy_matchup": "current-history",
        "opponent_checkpoint_id": "frozen",
        "terminal_tick": 3600,
        "result": [1, -1],
        "first_nonwait_action_tick": [140, 160],
        "first_successful_card_tick": [161, 181],
        "cumulative_elixir_overflow_wasted": [0.00001, 2.0],
        "game_elixir_overflow_wasted": [0.0, 0.0],
        "personal_elixir_overflow_penalty_total": [0.02, 0.0],
        "game_elixir_overflow_penalty_total": [0.03, 0.0],
        "unilateral_elixir_overflow_penalty_total": [0.0, 0.0],
        "elixir_overflow_penalty_total": [0.02, 0.0],
    }
    result.update(changes)
    return result


class ElixirAnalysisTests(unittest.TestCase):
    def test_execution_time_and_no_double_counting_of_cost(self):
        row = learner_rows("stage", [match()], {})[0]
        self.assertEqual(row["first_submit_s"], 7.0)
        self.assertEqual(row["first_card_s"], 8.05)
        self.assertAlmostEqual(row["overflow_cost"], 0.05)

    def test_missing_is_not_zero_and_disabled_tracking_is_unavailable(self):
        row = learner_rows("stage", [match()], {}, game_tracking_enabled=False)[0]
        self.assertIsNone(row["actions"])
        self.assertIsNone(row["after_first_waste"])
        report = distribution([row, {"waste": None}, {"waste": 0}], "waste")
        self.assertIn("n=2/3", report)
        self.assertIn("positive=1/2", report)
        self.assertIn("max=0.000010", report)

    def test_self_play_and_truncations_do_not_enter_win_rate(self):
        rows = learner_rows("stage", [
            match(), match(result=[-1, 1]), match(result=[0, 0]),
            match(current_owners=[0, 1], policy_matchup="current-current"),
            match(truncated=True),
        ], {})
        report = win_summary(rows)
        self.assertIn("n=3 win=33.3% draw=33.3%", report)
        self.assertIn("Wilson95%", report)

    def test_live_partial_tail_is_skipped_but_corrupt_complete_line_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.jsonl"
            path.write_text('{"sequence":1}\n{"sequence":', encoding="utf-8")
            with redirect_stderr(io.StringIO()) as warnings:
                self.assertEqual(read_jsonl(path), [{"sequence": 1}])
            self.assertIn("unfinished", warnings.getvalue())
            path.write_text('{"sequence":1}\ninvalid\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                read_jsonl(path)

    def test_cli_separates_stages_opponents_and_exports_selected_seats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage_name, winner in (("train-1", 1), ("train-2", -1)):
                stage = root / stage_name
                stage.mkdir()
                rows = [match(sequence=0), match(sequence=1, result=[winner, -winner]),
                        match(sequence=2, opponent_checkpoint_id="other")]
                (stage / "matches.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                (stage / "config.json").write_text(
                    json.dumps({"arguments": {"game_elixir_overflow_penalty": 0.01}}), encoding="utf-8"
                )
            output = root / "seats.csv"
            with patch("sys.argv", ["analyze", str(root), "--last", "2", "--details", "--csv", str(output)]):
                with redirect_stdout(io.StringIO()) as captured:
                    self.assertEqual(main(), 0)
            text = captured.getvalue()
            self.assertIn("=== train-1 ===", text)
            self.assertIn("=== train-2 ===", text)
            self.assertEqual(text.count("recorded matches=3 selected=2"), 2)
            self.assertEqual(text.count("opponent: frozen"), 2)
            self.assertEqual(text.count("opponent: other"), 2)
            with output.open(newline="", encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 4)
            self.assertEqual({row["sequence"] for row in exported}, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
