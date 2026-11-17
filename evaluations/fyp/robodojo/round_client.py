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

"""Record episode identity and the policy that generated each round."""

import json
import os
from pathlib import Path

from XPolicyLab.policy.X_VLA import deploy


def eval_one_episode_batch(TASK_ENV, model_client):
    deploy.eval_one_episode_batch(TASK_ENV, model_client)
    path = Path(os.environ["FYP_ROLLOUT_DIR"]) / "batches.jsonl"
    previous = path.read_text().splitlines() if path.exists() else []
    batch = len(previous)
    prior = [r for line in previous for r in json.loads(line)]
    mapping = json.loads(os.environ.get("FYP_LAYOUT_MAP", "{}"))
    records = []
    for env, slot in TASK_ENV.current_env_seed_map.items():
        layout = mapping.get(str(slot), int(slot))
        records.append(dict(env=env, layout=int(slot), layout_id=layout,
            episode_id=f"{os.environ['ROBODOJO_RUN_ID']}:batch{batch}:env{env}",
            round=int(os.environ["FYP_ROUND"]),
            repeat_index=sum(r["layout_id"] == layout for r in prior + records),
            behavior_policy=os.environ["FYP_BEHAVIOR_POLICY"],
            steps=int(TASK_ENV.take_action_cnt[env]), success=bool(TASK_ENV.success[env]),
            unstable=env in TASK_ENV.unstable_envs,
            termination="success" if TASK_ENV.success[env] else "horizon_or_failure"))
    with path.open("a") as stream:
        stream.write(json.dumps(records) + "\n")


def eval_one_episode(TASK_ENV, model_client):
    raise RuntimeError("Round recording requires the official batched protocol")
