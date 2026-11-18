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

"""Factor input images and saved RNG states for actual bowl first chunks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from audit_policy_replay import difference, restore_rng
from flow_policy import Model


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "adapter", "diagnostic", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    assert (args.diagnostic / "COMPLETE").is_file()
    args.output.mkdir(parents=True, exist_ok=False)
    source_protocol = json.loads((args.diagnostic / "protocol.json").read_text())
    records = {}
    for repeat in (1, 2, 3):
        root = args.diagnostic / f"repeat-{repeat}" / "policy"
        calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        for call in calls:
            if call["chunk"] != 0:
                continue
            path = root / f"{call['call']:05d}.npz"
            with np.load(path, allow_pickle=False) as data:
                obs = dict(images=[data["image"].copy()], proprio=data["proprio"].copy(),
                           prompt=data["prompt"].item())
                action = data["action"].copy()
            rng_path = root / f"{call['call']:05d}-rng.pt"
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            records[repeat, call["env"]] = dict(obs=obs, action=action, rng=rng,
                                                input_path=str(path), rng_path=str(rng_path))
    assert set(records) == {(r, e) for r in (1, 2, 3) for e in range(4)}
    os.environ.pop("FYP_ROLLOUT_DIR", None)
    os.environ["FYP_RL_ADAPTER"] = str(args.adapter)
    os.environ["FYP_POLICY_SEED"] = "1701"
    model = Model(dict(model_path=str(args.base), processor_path=str(args.base / "processor"),
                       task_name="stack_bowls", action_type="ee", steps=10, domain_id=0, device="cuda"))
    rows = []
    for env, layout in enumerate(source_protocol["slot_to_layout"]):
        common = records[1, env]
        for repeat in (1, 2, 3):
            record = records[repeat, env]
            np.testing.assert_array_equal(record["obs"]["proprio"], common["obs"]["proprio"])
            assert record["obs"]["prompt"] == common["obs"]["prompt"]
            actions = {}
            for condition, observation, rng in [
                ("own", record["obs"], record["rng"]),
                ("own_repeat", record["obs"], record["rng"]),
                ("common_rng", record["obs"], common["rng"]),
                ("common_input", common["obs"], record["rng"]),
            ]:
                restore_rng(rng)
                actions[condition] = model.infer(observation).copy()
            np.savez_compressed(args.output / f"layout-{layout}-repeat-{repeat}.npz", **actions,
                                stored=record["action"], reference_stored=common["action"])
            rows.append(dict(layout=layout, repeat=repeat, input_path=record["input_path"],
                rng_path=record["rng_path"],
                own_vs_stored=difference(actions["own"], record["action"]),
                repeated=difference(actions["own"], actions["own_repeat"]),
                rng_effect_same_input=difference(actions["own"], actions["common_rng"]),
                common_input_vs_reference=difference(actions["common_input"], common["action"]),
                image_effect_common_rng=difference(actions["common_rng"], common["action"])))
    audit = dict(complete=True, source=str(args.diagnostic), source_protocol=source_protocol,
                 base=str(args.base), adapter=str(args.adapter), rows=rows,
                 scope="first bowl action chunks only; no simulation or outcome intervention",
                 conditions="own input/RNG replay; exact repeat; own image with repeat1 RNG; repeat1 image with own RNG",
                 runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                              gpu=torch.cuda.get_device_name(0),
                              deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
