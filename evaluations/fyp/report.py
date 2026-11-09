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

"""Inspect LIBERO episode logs and combine disjoint evaluation batches."""

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_MARKER = "[libero eval]"
_FIELD = re.compile(
    r"\b(task_id|trial_id|success|episode_len|elapsed_seconds|success_at_end)"
    r"\s*=\s*([^,\s\]]+)"
)
_TASK_FIELDS = ("key", "suite", "name", "task_id")
_EPISODE_FIELDS = (
    "suite",
    "name",
    "key",
    "task_id",
    "trial_id",
    "success",
    "episode_len",
    "elapsed_seconds",
    "success_at_end",
    "source_line",
)
_SUMMARY_FIELDS = (
    *_TASK_FIELDS,
    "status",
    "expected_trials",
    "completed_trials",
    "num_expected",
    "num_completed",
    "successes",
    "success_rate",
    "partial_success_rate",
    "wall_seconds",
    "error",
)
_METADATA_FIELDS = (
    "git_commit",
    "evo_git_commit",
    "libero_version",
    "model_path",
    "vlm_path",
    "evo_repo",
    "protocol",
)


def _boolean(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Expected True or False, got {value!r}")
    return value == "True"


def _trial_ids(values: list[int]) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a nonempty list of trial IDs")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"Expected nonnegative integer trial IDs, got {values!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"Expected unique trial IDs, got {values!r}")
    return sorted(values)


def _episode(fields: dict[str, str], task: dict, line_number: int) -> dict:
    required = {"task_id", "trial_id", "success", "episode_len", "elapsed_seconds"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(
            f"Missing episode fields {sorted(missing)}; "
            "use the LIBERO completion logger with episode_len and elapsed_seconds"
        )
    row = {key: task[key] for key in _TASK_FIELDS}
    row.update(
        {key: int(fields[key]) for key in ("task_id", "trial_id", "episode_len")}
    )
    if row["task_id"] < 0 or row["trial_id"] < 0 or row["episode_len"] <= 0:
        raise ValueError("Expected nonnegative task/trial IDs and positive episode_len")
    elapsed = float(fields["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Expected finite, nonnegative episode elapsed_seconds")
    row.update(
        success=_boolean(fields["success"]),
        elapsed_seconds=elapsed,
        success_at_end=(
            _boolean(fields["success_at_end"]) if fields.get("success_at_end") else None
        ),
        source_line=line_number,
    )
    return row


def _summarize(
    task: dict,
    expected: list[int],
    rows: list[dict],
    errors: list[str],
    wall_seconds: float | None,
) -> dict:
    observed: dict[int, dict] = {}
    expected_set = set(expected)
    for row in rows:
        trial = row["trial_id"]
        if row["task_id"] != task["task_id"] or trial not in expected_set:
            errors.append(
                f"Unexpected task/trial ({row['task_id']}, {trial}); "
                f"expected task {task['task_id']} and trials {expected}"
            )
        elif trial in observed:
            errors.append(
                f"Duplicate completion for task {task['task_id']}, trial {trial}"
            )
        else:
            observed[trial] = row
    completed = sorted(observed)
    missing = sorted(expected_set - observed.keys())
    status = "error" if errors else "incomplete" if missing else "complete"
    if missing:
        errors.append(f"Missing completed trials: {missing}")
    successes = sum(row["success"] for row in observed.values())
    partial_rate = successes / len(observed) if observed else None
    return {
        **{key: task[key] for key in _TASK_FIELDS},
        "task": dict(task),
        "expected_trials": expected,
        "completed_trials": completed,
        "num_expected": len(expected),
        "num_completed": len(observed),
        "successes": successes,
        "success_rate": partial_rate if status == "complete" else None,
        "partial_success_rate": partial_rate,
        "status": status,
        "error": "; ".join(errors) if errors else None,
        "wall_seconds": wall_seconds,
        "rows": rows,
    }


def inspect_task(
    directory: Path,
    task: dict,
    trial_ids: list[int],
    returncode: int | None,
    elapsed_seconds: float,
    error: str | None = None,
) -> dict:
    """Write episode observations and a validated task summary.

    A policy failure is a completed episode with ``success=False``. Process,
    logging, and coverage errors retain observations but suppress success_rate.
    Trial lists in the summary contain IDs; num_expected/num_completed are counts.
    """
    directory = Path(directory)
    expected = _trial_ids(trial_ids)
    errors = [error] if error else []
    rows = []
    if returncode != 0:
        errors.append(f"Evaluation process did not exit successfully: {returncode}")
    wall_seconds = float(elapsed_seconds)
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        errors.append("Expected finite, nonnegative process elapsed_seconds")
        wall_seconds = None
    try:
        with (directory / "console.log").open(
            encoding="utf-8", errors="replace"
        ) as log:
            for line_number, raw_line in enumerate(log, 1):
                line = _ANSI.sub("", raw_line)
                if _MARKER not in line:
                    continue
                payload = line.split(_MARKER, 1)[1].strip()
                if "duplicate episode skipped" in payload:
                    errors.append(f"Line {line_number}: {payload}")
                    continue
                if not payload.startswith("task_id="):
                    continue
                try:
                    pairs = _FIELD.findall(payload)
                    fields = dict(pairs)
                    if len(pairs) != len(fields) or "[repeated" in payload:
                        raise ValueError("Repeated or folded episode completion log")
                    rows.append(_episode(fields, task, line_number))
                except ValueError as exc:
                    errors.append(f"Line {line_number}: {exc}")
    except OSError as exc:
        errors.append(f"Cannot read console.log: {exc}")
    summary = _summarize(task, expected, rows, errors, wall_seconds)
    summary["returncode"] = returncode
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=_EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def _read_episodes(path: Path, task: dict) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not set(_EPISODE_FIELDS).issubset(reader.fieldnames or []):
                raise ValueError("Missing episode CSV columns")
            rows = []
            for number, fields in enumerate(reader, 2):
                if any(
                    fields[key] != str(task[key]) for key in ("suite", "name", "key")
                ):
                    raise ValueError(f"Task identity mismatch on CSV line {number}")
                rows.append(_episode(fields, task, int(fields["source_line"])))
            return rows
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def summarize_runs(roots: list[Path]) -> list[dict]:
    """Combine task reports with identical provenance and disjoint trial IDs.

    Raises:
        ValueError: Metadata is missing or differs, task reports are malformed,
            or planned/observed trials overlap. Failed runs retain their status.
    """
    metadata_reference = None
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for root in roots:
        root = Path(root)
        metadata = _read_json(root / "run.json")
        if any(
            key not in metadata or metadata[key] is None for key in _METADATA_FIELDS
        ):
            raise ValueError(f"Missing required run metadata in {root / 'run.json'}")
        fingerprint = json.dumps(
            {key: metadata[key] for key in _METADATA_FIELDS}, sort_keys=True
        )
        if metadata_reference is None:
            metadata_reference = fingerprint
        elif fingerprint != metadata_reference:
            raise ValueError(f"Run metadata/protocol mismatch: {root}")
        for path in sorted(root.glob("*/summary.json")):
            report = _read_json(path)
            if any(key not in report for key in _TASK_FIELDS):
                raise ValueError(f"Missing task identity in {path}")
            if report.get("status") not in ("complete", "incomplete", "error"):
                raise ValueError(f"Unexpected task status in {path}")
            task = {key: report[key] for key in _TASK_FIELDS}
            expected = _trial_ids(report.get("expected_trials"))
            rows = _read_episodes(path.with_name("episodes.csv"), task)
            group_key = (task["suite"], task["name"], task["key"])
            group = groups.setdefault(
                group_key,
                {
                    "task": task,
                    "expected": set(),
                    "seen": set(),
                    "rows": [],
                    "errors": [],
                    "wall_seconds": 0.0,
                    "source_runs": [],
                },
            )
            if group["task"] != task:
                raise ValueError(f"Task mapping differs across reports: {group_key}")
            overlap = group["expected"].intersection(expected)
            if overlap:
                raise ValueError(
                    f"Overlapping trial IDs for {group_key}: {sorted(overlap)}"
                )
            for row in rows:
                identity = (row["task_id"], row["trial_id"])
                if identity in group["seen"]:
                    raise ValueError(
                        f"Duplicate completion for {group_key}: {identity}"
                    )
                group["seen"].add(identity)
            group["expected"].update(expected)
            group["rows"].extend(rows)
            group["source_runs"].append(str(root))
            if report["status"] == "error":
                group["errors"].append(f"{path}: {report.get('error') or 'task error'}")
            wall = report.get("wall_seconds")
            if (
                not isinstance(wall, (int, float))
                or not math.isfinite(wall)
                or wall < 0
            ):
                group["errors"].append(f"Invalid wall_seconds in {path}")
            else:
                group["wall_seconds"] += wall
    results = []
    for group in groups.values():
        result = _summarize(
            group["task"],
            sorted(group["expected"]),
            group["rows"],
            group["errors"],
            group["wall_seconds"],
        )
        result["source_runs"] = group["source_runs"]
        results.append(result)
    return results


def write_summary(path: Path, rows: list[dict]) -> None:
    """Write one CSV row per observed task report, including incomplete runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            for key in ("expected_trials", "completed_trials"):
                rendered[key] = json.dumps(row[key])
            writer.writerow(rendered)
