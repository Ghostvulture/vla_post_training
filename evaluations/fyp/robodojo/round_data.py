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

"""Validate recorded episodes independently of their logical layout identity."""

import json
import math
from pathlib import Path


def load_episodes(root, official, horizon):
    root = Path(root)
    details = list(official["details"].values())
    rows = []
    for batch, line in enumerate((root / "batches.jsonl").read_text().splitlines()):
        for entry in json.loads(line):
            row = dict(entry)
            row["episode_id"] = row.get("episode_id", f"{root.parent.name}:batch{batch}:env{row['env']}")
            row["layout_id"] = row.get("layout_id", row["layout"])
            row["paths"] = sorted((root / f"batch-{batch}" / f"env-{row['env']}").glob("*.npz"))
            if row["unstable"] or row["steps"] <= 0 or len(row["paths"]) != math.ceil(row["steps"] / horizon):
                raise ValueError(f"Invalid or incomplete episode {row['episode_id']}")
            rows.append(row)
    if len(rows) != official["eval_time"] or len(rows) != len(details):
        raise ValueError("Episode count differs from official result")
    if len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episode_id")
    # Official layout numbers identify asset slots; repeated logical layouts have separate slots.
    remaining = list(details)
    for row in rows:
        matches = [i for i, d in enumerate(remaining) if d["layout_id"] == row["layout"] and d["success"] == row["success"]]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous official result for {row['episode_id']}")
        remaining.pop(matches[0])
    if not math.isclose(sum(r["success"] for r in rows) / len(rows), official["success_rate"]):
        raise ValueError("Success count differs from official result")
    return rows


def make_chunks(episodes, horizon, method):
    wins = sum(r["success"] for r in episodes)
    count = len(episodes)
    if method == "rl" and (wins == 0 or wins == count):
        return [], "all terminal rewards equal"
    chunks = []
    for row in episodes:
        if method == "fsft" and not row["success"]:
            continue
        advantage = float(row["success"]) - (wins - int(row["success"])) / (count - 1) if count > 1 else 0.
        for i, path in enumerate(row["paths"]):
            used = min(horizon, row["steps"] - i * horizon)
            if method == "fsft" and used != horizon:
                continue
            chunks.append(dict(path=path, used=used, advantage=advantage, episode_id=row["episode_id"], layout_id=row["layout_id"]))
    return chunks, None if chunks else "no complete successful action chunks"
