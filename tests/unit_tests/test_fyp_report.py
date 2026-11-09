# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate evaluation reports without importing the simulator or workers."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fyp_report", _ROOT / "evaluations/fyp/report.py"
)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

_TASK = {"key": "S1", "suite": "libero_spatial", "name": "pick_up_cup", "task_id": 1}
_METADATA = {
    "git_commit": "abc123",
    "evo_git_commit": "evo123",
    "libero_version": "0.1.0",
    "model_path": "/models/evo1",
    "vlm_path": "/models/vlm",
    "evo_repo": "/code/evo1",
    "protocol": {"image_size": 448, "max_steps": 350, "chunk": 14, "seed": 0},
}


def _line(trial: int, success: bool = True, task_id: int = 1) -> str:
    return (
        f"\x1b[36m(EnvGroup pid=123)\x1b[0m [INFO 11:22:33 EnvGroup-Rank-0] "
        f"[utils.py:1] [libero eval] task_id={task_id}, trial_id={trial}, "
        f"success={success}, episode_len=350, elapsed_seconds=12.345678, "
        f"success_at_end={success}\n"
    )


class TestFypReport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def inspect(self, text, trials=None, **kwargs):
        directory = kwargs.pop("directory", self.root / "S1")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "console.log").write_text(text, encoding="utf-8")
        return report.inspect_task(
            directory,
            _TASK,
            [0, 1] if trials is None else trials,
            kwargs.pop("returncode", 0),
            kwargs.pop("elapsed_seconds", 30.0),
            **kwargs,
        )

    def batch(self, name, trials, **kwargs):
        root = self.root / name
        result = self.inspect(
            "".join(_line(trial, trial % 2 == 0) for trial in trials),
            trials,
            directory=root / "S1",
            **kwargs,
        )
        (root / "run.json").write_text(json.dumps(_METADATA), encoding="utf-8")
        return root, result

    def test_complete_report_counts_policy_failure_and_writes_episode_fields(self):
        summary = self.inspect(_line(0) + _line(1, False))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_trials"], [0, 1])
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["successes"], 1)
        self.assertIsNone(summary["error"])
        with (self.root / "S1/episodes.csv").open(newline="") as source:
            episodes = list(csv.DictReader(source))
        self.assertEqual(episodes[1]["success"], "False")
        self.assertEqual(episodes[1]["success_at_end"], "False")
        self.assertEqual(episodes[0]["episode_len"], "350")
        self.assertEqual(episodes[0]["elapsed_seconds"], "12.345678")
        saved = json.loads((self.root / "S1/summary.json").read_text())
        self.assertEqual(saved, summary)

    def test_success_at_end_is_optional_and_distinct_from_sticky_success(self):
        text = _line(0).replace(", success_at_end=True", "")
        text += _line(1).replace("success_at_end=True", "success_at_end=False")
        summary = self.inspect(text)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertIsNone(summary["rows"][0]["success_at_end"])
        self.assertFalse(summary["rows"][1]["success_at_end"])

    def test_missing_episode_is_incomplete_not_a_policy_failure(self):
        summary = self.inspect(_line(0))
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(summary["partial_success_rate"], 1.0)
        self.assertIn("Missing completed trials: [1]", summary["error"])

    def test_all_policy_failures_have_a_valid_zero_rate(self):
        summary = self.inspect(_line(0, False) + _line(1, False))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["num_completed"], 2)

    def test_empty_log_has_no_measured_success_rate(self):
        summary = self.inspect("worker initialized\n")
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["num_completed"], 0)
        self.assertIsNone(summary["success_rate"])
        self.assertIsNone(summary["partial_success_rate"])

    def test_process_and_budget_errors_suppress_even_complete_success_rate(self):
        for kwargs in (
            {"returncode": 1},
            {"returncode": None},
            {"error": "Wall-clock budget exhausted"},
        ):
            with self.subTest(kwargs=kwargs):
                summary = self.inspect(_line(0) + _line(1), **kwargs)
                self.assertEqual(summary["status"], "error")
                self.assertEqual(summary["num_completed"], 2)
                self.assertIsNone(summary["success_rate"])

    def test_duplicate_and_conflicting_completions_are_errors(self):
        for second in (True, False):
            with self.subTest(second=second):
                summary = self.inspect(_line(0) + _line(0, second), [0])
                self.assertEqual(summary["status"], "error")
                self.assertEqual(summary["num_completed"], 1)
                self.assertEqual(len(summary["rows"]), 2)
                self.assertIsNone(summary["success_rate"])
                self.assertIn("Duplicate completion", summary["error"])

    def test_worker_deduplication_or_ray_log_folding_is_an_error(self):
        for suffix in (
            "[libero eval] duplicate episode skipped: task_id=1, trial_id=0\n",
            _line(0).rstrip() + " [repeated 2x across cluster]\n",
        ):
            with self.subTest(suffix=suffix):
                summary = self.inspect(_line(0) + suffix, [0])
                self.assertEqual(summary["status"], "error")
                self.assertIsNone(summary["success_rate"])

    def test_unexpected_task_and_trial_are_errors(self):
        for text in (_line(0, task_id=2), _line(2)):
            with self.subTest(text=text):
                summary = self.inspect(text)
                self.assertEqual(summary["status"], "error")
                self.assertEqual(summary["num_completed"], 0)
                self.assertIsNone(summary["success_rate"])

    def test_legacy_or_invalid_measurements_fail_clearly(self):
        legacy = "[libero eval] task_id=1, trial_id=0, success=True\n"
        for text in (
            legacy,
            _line(0).replace("12.345678", "nan"),
            _line(0).replace("episode_len=350", "episode_len=0"),
            _line(0).replace("success=True", "success=unknown"),
        ):
            with self.subTest(text=text):
                summary = self.inspect(text, [0])
                self.assertEqual(summary["status"], "error")
                self.assertIsNone(summary["success_rate"])
        self.assertIn("completion logger", self.inspect(legacy, [0])["error"])

    def test_missing_log_is_reported_and_creates_empty_episode_csv(self):
        summary = report.inspect_task(self.root / "missing", _TASK, [0], 1, 1.0)
        self.assertEqual(summary["status"], "error")
        self.assertIn("Cannot read console.log", summary["error"])
        self.assertTrue((self.root / "missing/episodes.csv").is_file())

    def test_disjoint_batches_merge_and_summary_csv_contains_only_observed_tasks(self):
        first, _ = self.batch("screen", list(range(10)))
        second, _ = self.batch("extend", list(range(10, 20)))
        summaries = report.summarize_runs([first, second])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["status"], "complete")
        self.assertEqual(summaries[0]["num_completed"], 20)
        self.assertEqual(summaries[0]["success_rate"], 0.5)
        report.write_summary(self.root / "summary.csv", summaries)
        with (self.root / "summary.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_completed"], "20")

    def test_overlapping_smoke_and_screen_trials_are_rejected(self):
        smoke, _ = self.batch("smoke", [0])
        screen, _ = self.batch("screen", list(range(10)))
        with self.assertRaisesRegex(ValueError, "Overlapping trial IDs"):
            report.summarize_runs([smoke, screen])

    def test_variants_with_same_source_task_are_separate_groups(self):
        root = self.root / "variants"
        for key in ("B1B", "B2P"):
            directory = root / key
            directory.mkdir(parents=True)
            (directory / "console.log").write_text(_line(0))
            report.inspect_task(directory, {**_TASK, "key": key}, [0], 0, 1.0)
        (root / "run.json").write_text(json.dumps(_METADATA))
        summaries = report.summarize_runs([root])
        self.assertEqual(len(summaries), 2)
        self.assertEqual({s["task"]["key"] for s in summaries}, {"B1B", "B2P"})

    def test_metadata_changes_and_missing_metadata_are_rejected(self):
        first, _ = self.batch("first", [0])
        second, _ = self.batch("second", [1])
        for key in _METADATA:
            with self.subTest(key=key):
                changed = dict(_METADATA)
                changed[key] = "different"
                (second / "run.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "metadata/protocol mismatch"):
                    report.summarize_runs([first, second])
        (second / "run.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Missing required run metadata"):
            report.summarize_runs([second])

    def test_merge_preserves_process_error_observations_without_official_rate(self):
        first, _ = self.batch("failed", [0], returncode=1)
        second, _ = self.batch("second", [1])
        summary = report.summarize_runs([first, second])[0]
        self.assertEqual(summary["num_completed"], 2)
        self.assertEqual(summary["status"], "error")
        self.assertIsNone(summary["success_rate"])

    def test_merge_preserves_missing_trials_and_ignores_unstarted_tasks(self):
        root = self.root / "partial"
        self.inspect(_line(0), [0, 1], directory=root / "S1")
        (root / "run.json").write_text(json.dumps(_METADATA))
        (root / "S2").mkdir()
        summaries = report.summarize_runs([root])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["status"], "incomplete")
        self.assertIsNone(summaries[0]["success_rate"])


if __name__ == "__main__":
    unittest.main()
