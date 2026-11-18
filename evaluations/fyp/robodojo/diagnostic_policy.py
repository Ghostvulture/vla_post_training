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

"""Record actual encoded inputs, RNG states and actions without resampling."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from audit_policy_replay import capture_rng
from flow_policy import Model as FlowPolicy


class Model(FlowPolicy):
    def __init__(self, model_cfg):
        if os.environ.get("FYP_ROLLOUT_DIR"):
            raise ValueError("Diagnosis must not enable extra exploration")
        super().__init__(model_cfg)
        self.trace_root = Path(os.environ["FYP_DIAGNOSTIC_DIR"]) / "policy"
        self.trace_root.mkdir(parents=True, exist_ok=False)
        self.call_index = 0

    def get_action_batch(self, env_idx_list=None, **kwargs):
        self.diagnostic_ids = iter(env_idx_list or self._latest_env_idx_list)
        return super().get_action_batch(env_idx_list=env_idx_list, **kwargs)

    def infer(self, observation, steps=None):
        env = int(next(self.diagnostic_ids))
        index = self.call_index
        rng = capture_rng()
        image = np.asarray(observation["images"][0]).copy()
        proprio = np.asarray(observation["proprio"]).copy()
        action = super().infer(observation, steps=steps)
        torch.save(rng, self.trace_root / f"{index:05d}-rng.pt")
        np.savez_compressed(self.trace_root / f"{index:05d}.npz", image=image,
                            proprio=proprio, prompt=observation["prompt"], action=action)
        with (self.trace_root / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(call=index, env=env, batch=self.batch,
                                         chunk=self.chunk,
                                         active_envs=self._latest_env_idx_list)) + "\n")
        self.call_index += 1
        return action
