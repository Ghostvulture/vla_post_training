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

"""Replay saved encoded observations under controlled RNG and invocation order."""

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from XPolicyLab.policy.X_VLA.model import Model as OfficialPolicy

from flow_policy import Model as FlowPolicy
from round_data import load_episodes


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def difference(first, second):
    if first.shape != (30, 20) or second.shape != first.shape:
        raise ValueError("Unexpected action chunk shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite action prediction")
    delta = first - second
    return dict(exact=bool(np.array_equal(first, second)), max_abs=float(np.abs(delta).max()),
                rms=float(np.sqrt(np.mean(delta**2))),
                xyz_max_abs_m=float(np.abs(delta[:, [0, 1, 2, 10, 11, 12]]).max()),
                rotation6d_max_abs=float(np.abs(delta[:, [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]]).max()),
                gripper_probability_max_abs=float(np.abs(delta[:, [9, 19]]).max()))


def main():
    parser = argparse.ArgumentParser()
    for key in ("base", "data", "result", "video-adapter", "rl-adapter", "fsft-model", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = load_episodes(args.data, json.loads(args.result.read_text()), 30)
    chosen = sorted(episodes, key=lambda r: r["layout_id"])[:8]
    if len(chosen) != 8:
        raise ValueError("Need eight real source episodes")
    observations, provenance = [], []
    for row in chosen:
        path = row["paths"][0]
        with np.load(path, allow_pickle=False) as data:
            observations.append(dict(images=[data["image"].copy()], prompt=data["prompt"].item(), proprio=data["proprio"].copy()))
        provenance.append(dict(episode_id=row["episode_id"], layout_id=row["layout_id"], source=str(path)))
    config = dict(model_path=str(args.base), processor_path=str(args.base / "processor"),
                  task_name="stack_blocks", action_type="ee", steps=10, domain_id=0, device="cuda")
    configurations = [
        ("base_native", OfficialPolicy, args.base, None),
        ("base_flow", FlowPolicy, args.base, None),
        ("video_shared42_low", FlowPolicy, args.base, args.video_adapter),
        ("round2_rl32", FlowPolicy, args.base, args.rl_adapter),
        ("round2_fsft32", OfficialPolicy, args.fsft_model, None),
    ]
    audit = dict(seed=1701, input_task="stack_blocks", input_selection="first chunk of logical layout IDs 0-7 from round2 RL collection",
                 observations=provenance, configurations={},
                 scope="encoded-observation infer() replay only; no simulator, new episodes, training, or reconstruction of historic bowl inputs",
                 conditions={
                     "reset_repeat": "same observation and restored complete RNG state",
                     "prefix_call": "same initial RNG, one other observed input is inferred before the target",
                     "prefix_restore": "same prefix call, then restore RNG immediately before the target",
                 })
    audit["runtime"] = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                            gpu=torch.cuda.get_device_name(0),
                            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                            cudnn_benchmark=torch.backends.cudnn.benchmark,
                            cudnn_deterministic=torch.backends.cudnn.deterministic,
                            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    baseline = {}
    for name, cls, model_path, adapter in configurations:
        os.environ.pop("FYP_ROLLOUT_DIR", None)
        os.environ.pop("FYP_RL_ADAPTER", None)
        os.environ["FYP_POLICY_SEED"] = "1701"
        if adapter is not None:
            os.environ["FYP_RL_ADAPTER"] = str(adapter)
        reset_seed(1701)
        policy = cls(dict(config, model_path=str(model_path)))
        rows = []
        for index, observation in enumerate(observations):
            image_before = observation["images"][0].copy()
            proprio_before = observation["proprio"].copy()
            reset_seed(1701)
            start = capture_rng()
            direct = policy.infer(observation).copy()
            restore_rng(start)
            repeat = policy.infer(observation).copy()
            restore_rng(start)
            prefix = observations[(index + 1) % len(observations)]
            policy.infer(prefix)
            after_prefix = policy.infer(observation).copy()
            restore_rng(start)
            policy.infer(prefix)
            restore_rng(start)
            after_restore = policy.infer(observation).copy()
            row = dict(index=index, layout_id=provenance[index]["layout_id"],
                       reset_repeat=difference(direct, repeat), prefix_call=difference(direct, after_prefix),
                       prefix_restore=difference(direct, after_restore))
            if name == "base_native":
                baseline[index] = direct.copy()
            elif name == "base_flow":
                row["native_parity"] = difference(baseline[index], direct)
            np.savez_compressed(args.output / f"{name}-{index}.npz", direct=direct, repeat=repeat,
                                after_prefix=after_prefix, after_restore=after_restore)
            if not np.array_equal(image_before, observation["images"][0]) or not np.array_equal(proprio_before, observation["proprio"]):
                raise ValueError("Policy mutated the stored input during replay")
            rows.append(row)
        audit["configurations"][name] = dict(model_path=str(model_path), adapter=str(adapter) if adapter else None, observations=rows)
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    audit["complete"] = True
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
