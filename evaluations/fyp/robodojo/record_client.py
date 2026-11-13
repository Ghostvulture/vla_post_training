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

"""Attach official layout IDs and executed action counts to the policy trace."""

import json
import os
from pathlib import Path

from XPolicyLab.policy.X_VLA import deploy


def eval_one_episode_batch(TASK_ENV, model_client):
    deploy.eval_one_episode_batch(TASK_ENV, model_client)
    records = [
        {
            "env": env_idx,
            "layout": int(layout),
            "steps": int(TASK_ENV.take_action_cnt[env_idx]),
            "success": bool(TASK_ENV.success[env_idx]),
            "unstable": env_idx in TASK_ENV.unstable_envs,
        }
        for env_idx, layout in TASK_ENV.current_env_seed_map.items()
    ]
    with (Path(os.environ["FYP_ROLLOUT_DIR"]) / "batches.jsonl").open("a") as stream:
        stream.write(json.dumps(records) + "\n")


def eval_one_episode(TASK_ENV, model_client):
    raise RuntimeError("The recorder requires the official batched X-VLA protocol.")
