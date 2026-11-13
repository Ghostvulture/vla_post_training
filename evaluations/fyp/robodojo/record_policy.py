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

"""Record X-VLA's encoded observations and raw predictions at chunk boundaries."""

import os
from pathlib import Path

import numpy as np
import torch
from XPolicyLab.policy.X_VLA.model import Model as OfficialPolicy


class Model(OfficialPolicy):
    def __init__(self, model_cfg):
        torch.manual_seed(int(os.environ.get("FYP_POLICY_SEED", "42")))
        super().__init__(model_cfg)
        self.record_dir = Path(os.environ["FYP_ROLLOUT_DIR"])
        self.batch = -1
        self.new_batch = True

    def reset(self):
        super().reset()
        self.new_batch = True

    def get_action_batch(self, env_idx_list=None, **kwargs):
        # Delegate action generation and conversion. infer() records before conversion.
        if self.new_batch:
            self.batch += 1
            self.chunk = 0
            self.new_batch = False
        self.record_ids = iter(self._latest_env_idx_list)
        actions = super().get_action_batch(env_idx_list=env_idx_list, **kwargs)
        self.chunk += 1
        return actions

    def infer(self, observation, steps=None):
        action = super().infer(observation, steps=steps)
        env_id = next(self.record_ids)
        directory = self.record_dir / f"batch-{self.batch}" / f"env-{env_id}"
        directory.mkdir(exist_ok=True, parents=True)
        with (directory / f"{self.chunk:04d}.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                image=observation["images"][0],
                proprio=observation["proprio"],
                prompt=observation["prompt"],
                action=action,
            )
        return action
