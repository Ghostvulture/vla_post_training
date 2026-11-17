#!/bin/bash
set -euo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=90
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
storage="${FYP_STORAGE:?}"
runtime="${FYP_RUNTIME:?}"
export HF_HOME="$runtime/cache/robodojo-hf"
"$runtime/envs/rlinf-evo1/bin/python" - "$storage/robodojo" <<'PY'
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import requests
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

root = Path(sys.argv[1])
repo = "RoboDojo-Benchmark/RoboDojo"
revision = "1127bda94f172d757386945af3eb3a6754d1a4e6"
checkpoint = "ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000"
api = HfApi(token=False)
for folder in [checkpoint, "Assets"]:
    files = [entry for entry in api.list_repo_tree(repo, path_in_repo=folder, recursive=True, repo_type="dataset", revision=revision) if isinstance(entry, RepoFile)]
    if not files:
        raise RuntimeError(f"No published files in {folder}")
    print(folder, len(files), "files", sum(f.size for f in files), "bytes", flush=True)
    def fetch(entry):
        path = root / entry.path
        if path.is_file() and path.stat().st_size == entry.size:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        for attempt in range(4):
            start = partial.stat().st_size if partial.exists() else 0
            if start == entry.size and partial.exists():
                partial.replace(path)
                return
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(entry.path)}"
            try:
                with requests.get(url, params={"download": "true", "offset": start, "attempt": attempt}, headers={"Range": f"bytes={start}-"} if start else {}, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    if response.status_code == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError(f"Unexpected range for {entry.path}")
                    with partial.open("ab" if response.status_code == 206 else "wb") as dest:
                        for chunk in response.iter_content(1024 * 1024):
                            dest.write(chunk)
                if partial.stat().st_size != entry.size:
                    raise IOError(f"Incomplete download of {entry.path}")
                partial.replace(path)
                return
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(fetch, files), 1):
            if i % 100 == 0 or i == len(files):
                print(folder, i, "/", len(files), flush=True)
with (root / checkpoint / "config.json").open() as reader:
    config = json.load(reader)
print("verified published sizes", revision, "model_type", config.get("model_type"))
PY
