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

"""CPU checks for layout-matched phase summaries; all fixtures are synthetic."""

import json
import tempfile
import unittest
from pathlib import Path

from summarize_rounds import paired, read_evaluation, summarize, wilson


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def result(self, ids):
        data = dict(eval_time=len(ids), success_rate=.5, details={str(i):dict(layout_id=x,success=i % 2 == 0) for i,x in enumerate(ids)})
        path = self.root / "result.json"
        path.write_text(json.dumps(data))
        return path

    def test_primary_development_ids(self):
        row = read_evaluation(self.result(range(10,30)), range(10,30))
        self.assertEqual((row["wins"], row["n"]), (10,20))

    def test_extra_or_missing_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "membership"):
            read_evaluation(self.result(range(30)), range(10,30))

    def test_duplicate_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            read_evaluation(self.result([10,10]), range(10,12))

    def test_wrong_success_rate_rejected(self):
        path = self.result(range(20))
        data = json.loads(path.read_text())
        data["success_rate"] = .7
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "disagree"):
            read_evaluation(path, range(20))

    def test_paired_gains_and_losses(self):
        result = paired(dict(outcomes={0:True,1:False,2:False}), dict(outcomes={0:False,1:True,2:True}))
        self.assertEqual(result["gained_layouts"], [1,2])
        self.assertEqual(result["lost_layouts"], [0])
        self.assertAlmostEqual(result["delta"], 1/3)

    def test_mismatched_paired_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            paired(dict(outcomes={0:True}),dict(outcomes={1:True}))

    def test_no_completion_marker_rejected(self):
        with self.assertRaisesRegex(ValueError, "completion marker"):
            summarize([self.root])

    def test_wilson_reference(self):
        low, high = wilson(10,20)
        self.assertAlmostEqual(low, .299298, places=5)
        self.assertAlmostEqual(high, .700702, places=5)


if __name__ == "__main__":
    unittest.main()
