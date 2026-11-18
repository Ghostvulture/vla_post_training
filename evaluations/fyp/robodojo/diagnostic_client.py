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

"""Observe the official control loop at observation and action boundaries."""

import json
import os
from pathlib import Path

from XPolicyLab.policy.X_VLA import deploy


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def scene_state(env, ids):
    records = {}
    for idx in ids:
        objects = {}
        for name, obj in env.scene_manager._rigid_and_dynamic_objects[idx].items():
            position, orientation = obj.get_world_pose()
            objects[name] = dict(position=position, orientation=orientation,
                                 linear_velocity=obj.get_linear_velocity(),
                                 angular_velocity=obj.get_angular_velocity())
        if not objects:
            raise ValueError("No rigid objects recorded for bowl diagnosis")
        records[str(idx)] = dict(objects=objects,
                                 step=int(env.take_action_cnt[idx]),
                                 success=bool(env.success[idx]),
                                 end=bool(env.end_flag[idx]))
    return plain(records)


def eval_one_episode_batch(TASK_ENV, model_client):
    root = Path(os.environ["FYP_DIAGNOSTIC_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "client.jsonl"
    if path.exists():
        raise ValueError("Expected one four-layout batch per diagnostic process")
    action_count = 0
    obs_count = 0
    original_action = TASK_ENV.take_action_batch
    original_obs = TASK_ENV.get_obs_batch
    with path.open("x") as stream:
        def record(row):
            stream.write(json.dumps(plain(row), allow_nan=False) + "\n")
            stream.flush()

        ids = list(TASK_ENV.current_env_seed_map)
        record(dict(event="initial", mapping=TASK_ENV.current_env_seed_map,
                    run_id=os.environ.get("ROBODOJO_RUN_ID"), num_envs=TASK_ENV.num_envs,
                    state=scene_state(TASK_ENV, ids)))

        def observe(env_idx_list=None, last_frame=False):
            nonlocal obs_count
            observations = original_obs(env_idx_list, last_frame=last_frame)
            ids = [int(o["env_idx"]) for o in observations]
            record(dict(event="observation", index=obs_count,
                        state=scene_state(TASK_ENV, ids),
                        robot_states=[o["state"] for o in observations], envs=ids))
            obs_count += 1
            return observations

        def act(actions_list, env_idx_list=None):
            nonlocal action_count
            ids = list(range(TASK_ENV.num_envs)) if env_idx_list is None else list(env_idx_list)
            record(dict(event="before_action", index=action_count, envs=ids,
                        commands=actions_list, state=scene_state(TASK_ENV, ids)))
            result = original_action(actions_list, env_idx_list)
            record(dict(event="after_action", index=action_count, envs=ids,
                        state=scene_state(TASK_ENV, ids)))
            action_count += 1
            return result

        TASK_ENV.get_obs_batch = observe
        TASK_ENV.take_action_batch = act
        try:
            deploy.eval_one_episode_batch(TASK_ENV, model_client)
            record(dict(event="complete", actions=action_count, observations=obs_count,
                        state=scene_state(TASK_ENV, ids)))
        finally:
            TASK_ENV.get_obs_batch = original_obs
            TASK_ENV.take_action_batch = original_action


def eval_one_episode(TASK_ENV, model_client):
    raise RuntimeError("Diagnosis requires the official batch evaluation loop")
