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

"""Seeded full-flow FSFT or final-step RL inference and round recording."""

import os
from pathlib import Path

import numpy as np
import torch
from XPolicyLab.policy.X_VLA.model import Model as OfficialPolicy

from flow_policy import Model as FlowPolicy

_METHOD = os.environ["FYP_METHOD"]
if _METHOD not in {"rl", "fsft"}:
    raise ValueError(_METHOD)


class Model(FlowPolicy if _METHOD == "rl" else OfficialPolicy):
    def __init__(self, model_cfg):
        if _METHOD == "fsft" and os.environ.get("FYP_FSFT_MODEL"):
            model_cfg = dict(model_cfg, checkpoint_path=os.environ["FYP_FSFT_MODEL"])
        torch.manual_seed(int(os.environ["FYP_POLICY_SEED"]))
        super().__init__(model_cfg)
        torch.manual_seed(int(os.environ["FYP_POLICY_SEED"]))
        if _METHOD == "fsft":
            self.batch = -1
            self.new_batch = True

    def reset(self):
        super().reset()
        self.new_batch = True

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if _METHOD == "rl":
            return super().get_action_batch(env_idx_list=env_idx_list, **kwargs)
        if self.new_batch:
            self.batch += 1
            self.chunk = 0
            self.new_batch = False
        self.record_ids = iter(self._latest_env_idx_list)
        actions = super().get_action_batch(env_idx_list=env_idx_list, **kwargs)
        self.chunk += 1
        return actions

    def infer(self, observation, steps=None):
        if _METHOD == "rl":
            # The parent consumes one environment ID when recording.
            if os.environ.get("FYP_ROLLOUT_DIR"):
                import itertools
                self.record_ids, copy_ids = itertools.tee(self.record_ids)
                env = next(copy_ids)
            action = super().infer(observation, steps=steps)
        else:
            action = super().infer(observation, steps=steps)
            if os.environ.get("FYP_ROLLOUT_DIR"):
                env = next(self.record_ids)
        if os.environ.get("FYP_ROLLOUT_DIR"):
            directory = Path(os.environ["FYP_ROLLOUT_DIR"]) / f"batch-{self.batch}" / f"env-{env}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.chunk:04d}.npz"
            if _METHOD == "rl":
                with np.load(path, allow_pickle=False) as previous:
                    record = dict(previous)
            else:
                record = dict(image=observation["images"][0], prompt=observation["prompt"], proprio=observation["proprio"])
            record.update(action=action, behavior_policy=os.environ["FYP_BEHAVIOR_POLICY"], round=int(os.environ["FYP_ROUND"]))
            with path.open("wb") as stream:
                np.savez_compressed(stream, **record)
        return action
