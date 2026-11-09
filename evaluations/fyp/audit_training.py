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

"""Require finite updates, complete interaction budgets, and paired source resets."""

import argparse
import json
import math
import re
from pathlib import Path


def validate_records(metrics, counters, iterations):
    """Reject incomplete, nonfinite, or entirely ineffective training runs."""
    required = [
        "train/actor/grad_norm",
        "train/actor/approx_kl",
        "rollout/group_mixed_fraction",
        "rollout/advantages_max",
    ]
    for tag in required:
        rows = metrics.get(tag, [])
        if [row["step"] for row in rows] != list(range(iterations)):
            raise ValueError(f"Incomplete training metrics: {tag}")
    if any(
        not math.isfinite(row["value"]) for rows in metrics.values() for row in rows
    ):
        raise ValueError("Nonfinite training metric")
    if not any(row["value"] > 0 for row in metrics["train/actor/grad_norm"]):
        raise ValueError("No nonzero gradient update")
    if not any(row["value"] > 0 for row in metrics["rollout/group_mixed_fraction"]):
        raise ValueError("No mixed-reward group")
    if counters != [2800 * step for step in range(1, iterations + 1)]:
        raise ValueError(f"Unexpected simulator interaction counts: {counters}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for event in args.run.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            metrics.setdefault(tag, []).extend(
                {"step": row.step, "value": row.value}
                for row in accumulator.Scalars(tag)
            )
    for rows in metrics.values():
        rows.sort(key=lambda row: row["step"])
    log = (args.run / "console.log").read_text()
    counters = list(map(int, re.findall(r"simulator_action_steps=(\d+)", log)))
    validate_records(metrics, counters, args.iterations)
    sources = [
        [int(value) for value in row.split()]
        for row in re.findall(r"\[libero train reset\].*?trial_ids=\[([0-9 ]+)\]", log)
    ]
    if not sources or any(
        len(row) != 8 or len(set(row)) != 1 or not 0 <= row[0] < 20 for row in sources
    ):
        raise ValueError("Invalid source initialization trace")
    if args.compare:
        previous = json.loads(args.compare.read_text())
        if (
            previous["source_groups"] != sources
            or previous["simulator_action_steps"] != counters[-1]
        ):
            raise ValueError(
                "Methods do not share source initialization sequence and interaction budget"
            )
    report = {
        "valid": True,
        "iterations": args.iterations,
        "simulator_action_steps": counters[-1],
        "source_groups": sources,
        "metrics": metrics,
        "git_commit": (args.run / "git-commit.txt").read_text().strip(),
    }
    (args.run / "training-audit.json").write_text(json.dumps(report, indent=2))
    print(
        f"Verified {args.iterations} iterations, {counters[-1]} simulator action steps"
    )


if __name__ == "__main__":
    main()
