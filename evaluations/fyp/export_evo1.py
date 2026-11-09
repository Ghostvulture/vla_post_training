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

"""Audit and export RLinf full weights for the official Evo-1 evaluator."""

import argparse
import json
import shutil
from pathlib import Path


def unwrap_weights(weights: dict) -> dict:
    """Remove only the known RLinf Evo-1 wrapper prefix."""
    if not weights or any(not key.startswith("evo1.") for key in weights):
        raise ValueError("Expected only evo1.* keys in RLinf full_weights.pt")
    return {key.removeprefix("evo1."): value for key, value in weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Export directory must not exist")
    import torch

    original = torch.load(
        args.base / "mp_rank_00_model_states.pt", map_location="cpu", weights_only=False
    )
    original = original.get("module", original)
    trained = unwrap_weights(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    if set(original) != set(trained):
        raise ValueError(
            f"Checkpoint keys differ: missing={set(original) - set(trained)}, extra={set(trained) - set(original)}"
        )
    changed, frozen_changed = [], []
    max_delta = 0.0
    for key, value in trained.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite weight: {key}")
        reference = original[key].to(value.dtype)
        if not torch.equal(reference, value):
            changed.append(key)
            max_delta = max(
                max_delta, float((reference.float() - value.float()).abs().max())
            )
            if not key.startswith("action_head."):
                frozen_changed.append(key)
    if frozen_changed:
        raise ValueError(f"Frozen weights changed: {frozen_changed}")
    if not changed:
        raise ValueError("No action-head weights changed; pilot did not pass")
    args.output.mkdir(parents=True)
    for name in ["config.json", "norm_stats.json"]:
        shutil.copy2(args.base / name, args.output / name)
    torch.save({"module": trained}, args.output / "mp_rank_00_model_states.pt")
    audit = {
        "source_weights": str(args.weights.resolve()),
        "base": str(args.base.resolve()),
        "changed_action_tensors": changed,
        "frozen_tensors_changed": frozen_changed,
        "max_abs_delta": max_delta,
        "comparison": "base cast to saved dtype before exact equality",
    }
    (args.output / "export-audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
