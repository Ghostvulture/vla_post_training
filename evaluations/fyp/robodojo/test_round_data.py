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

"""CPU contract tests for independent episodes and supervised chunk eligibility."""

import json
import tempfile
import unittest
from pathlib import Path

from round_data import load_episodes, make_chunks


class RoundDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(env=i, layout=i, layout_id=7, episode_id=f"e{i}", steps=31,
                          success=i == 0, unstable=False) for i in range(2)]
        for i in range(2):
            folder = self.root / "batch-0" / f"env-{i}"
            folder.mkdir(parents=True)
            for j in range(2):
                (folder / f"{j:04d}.npz").touch()
        self.result = dict(eval_time=2, success_rate=.5, details={str(i): dict(layout_id=i, success=i == 0) for i in range(2)})

    def load(self):
        (self.root / "batches.jsonl").write_text(json.dumps(self.rows) + "\n")
        return load_episodes(self.root, self.result, 30)

    def test_repeated_logical_layout_retains_independent_episodes(self):
        episodes = self.load()
        chunks, skip = make_chunks(episodes, 30, "rl")
        self.assertIsNone(skip)
        self.assertEqual([x["used"] for x in chunks], [30, 1, 30, 1])
        self.assertEqual({x["episode_id"] for x in chunks}, {"e0", "e1"})
        self.assertEqual([x["advantage"] for x in chunks], [1., 1., -1., -1.])

    def test_fsft_excludes_failed_episodes_and_unexecuted_tail(self):
        chunks, skip = make_chunks(self.load(), 30, "fsft")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["episode_id"], "e0")
        self.assertIsNone(skip)

    def test_duplicate_identity_rejected(self):
        self.rows[1]["episode_id"] = "e0"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_missing_chunk_rejected(self):
        (self.root / "batch-0/env-0/0001.npz").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load()

    def test_mismatched_reward_rejected(self):
        self.rows[1]["success"] = True
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.load()

    def test_empty_success_skips_both_methods(self):
        episodes = self.load()
        for row in episodes:
            row["success"] = False
        for method in ["rl", "fsft"]:
            chunks, skip = make_chunks(episodes, 30, method)
            self.assertEqual(chunks, [])
            self.assertTrue(skip)

    def test_legacy_identity_is_unique(self):
        for row in self.rows:
            del row["episode_id"]
        episodes = self.load()
        self.assertEqual(len({e["episode_id"] for e in episodes}), 2)


if __name__ == "__main__":
    unittest.main()
