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

"""Download the official Pi05 base checkpoint on the server, with resumable files."""

import argparse
import fcntl
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

PREFIX = "checkpoints/pi05_base/"
API = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"


def relative_name(name):
    """Reject objects outside the fixed checkpoint prefix."""
    if not name.startswith(PREFIX):
        raise ValueError(f"Unexpected object: {name}")
    relative = PurePosixPath(name[len(PREFIX) :])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid object path: {name}")
    return Path(*relative.parts)


def fetch_manifest():
    items = []
    token = None
    while True:
        query = {"prefix": PREFIX, "maxResults": 1000}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(
            API + "?" + urllib.parse.urlencode(query), timeout=60
        ) as response:
            page = json.load(response)
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            break
    if not items or not any(x["name"].endswith("params/_METADATA") for x in items):
        raise ValueError("Official checkpoint listing is incomplete")
    return [{k: x[k] for k in ("name", "size", "generation")} for x in items]


def download(item, root):
    target = root / relative_name(item["name"])
    target.parent.mkdir(parents=True, exist_ok=True)
    size = int(item["size"])
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError(f"Unexpected completed file size: {target}")
        print(f"Present {target.relative_to(root)}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    if size == 0:
        target.touch(exist_ok=False)
        return
    url = API + "/" + urllib.parse.quote(item["name"], safe="")
    url += "?" + urllib.parse.urlencode(
        {"alt": "media", "generation": item["generation"]}
    )
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ValueError(f"Oversized partial file: {partial}")
            if offset < size:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={offset}-",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {offset}-"):
                        raise ValueError(
                            "Server did not honor the requested byte range"
                        )
                    with partial.open("ab") as output:
                        while chunk := response.read(8 * 1024 * 1024):
                            output.write(chunk)
            if partial.stat().st_size != size:
                raise OSError(f"Incomplete object: {item['name']}")
            partial.replace(target)
            print(f"Downloaded {target.relative_to(root)} ({size} bytes)", flush=True)
            return
        except (OSError, ValueError) as error:
            print(f"Attempt {attempt + 1} failed: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.output / "download-manifest.json"
        if manifest_path.exists():
            items = json.loads(manifest_path.read_text())
        else:
            if list(args.output.glob("params/*")):
                raise ValueError(
                    "Existing weights have no recorded download provenance"
                )
            items = fetch_manifest()
            with manifest_path.open("x") as output:
                json.dump(items, output, indent=2)
        print(
            f"Official pi05_base: {len(items)} objects, {sum(int(x['size']) for x in items)} bytes",
            flush=True,
        )
        for item in items:
            download(item, args.output)
        (args.output / "download-complete.json").write_text(
            json.dumps(
                {
                    "source": "gs://openpi-assets/checkpoints/pi05_base",
                    "objects": len(items),
                    "validation": "generation-pinned HTTPS downloads and byte counts; model loading not yet verified",
                },
                indent=2,
            )
        )
        print(
            "Download complete; conversion and policy validation remain pending.",
            flush=True,
        )


if __name__ == "__main__":
    main()
