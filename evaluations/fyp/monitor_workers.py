# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Record this job's worker limits and ancestor cgroup memory events."""

import json
import os
from pathlib import Path
import time


def read(path):
    try:
        return path.read_text()
    except (OSError, PermissionError):
        return None


def main():
    seen = set()
    job = os.environ["SLURM_JOB_ID"]
    while True:
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                command = read(proc / "comm") or ""
                if not command.startswith("ray::"):
                    continue
                env = (proc / "environ").read_bytes().split(b"\0")
                if f"SLURM_JOB_ID={job}".encode() not in env:
                    continue
                group = read(proc / "cgroup")
                row = {"time": time.time(), "pid": int(proc.name), "comm": command.strip(), "cgroup": group}
                if proc.name not in seen:
                    row["limits"] = read(proc / "limits")
                    seen.add(proc.name)
                row["memory"] = {}
                for line in (group or "").splitlines():
                    if not line.startswith("0::"):
                        continue
                    root = Path("/sys/fs/cgroup")
                    directory = root / line[3:].lstrip("/")
                    while directory == root or root in directory.parents:
                        row["memory"][str(directory)] = {
                            key: read(directory / key)
                            for key in ("memory.current", "memory.max", "memory.events")
                        }
                        if directory == root:
                            break
                        directory = directory.parent
                print(json.dumps(row), flush=True)
            except (OSError, ProcessLookupError):
                continue
        time.sleep(15)


if __name__ == "__main__":
    main()
