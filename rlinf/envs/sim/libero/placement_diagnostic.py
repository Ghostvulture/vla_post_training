# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only, opt-in SG6 simulator diagnostics; never changes reward or done."""

import json
import os
from pathlib import Path

import numpy as np


def instrument_placement(env, directory):
    """Record simulator state after each step, including reset settling steps."""
    from libero.libero.envs.object_states.base_object_states import ObjectState

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"sim-{os.getpid()}.jsonl"
    core = env.env
    names = ("cream_cheese_1", "akita_black_bowl_1")
    states = [ObjectState(core, name) for name in names]
    objects = [core.get_object(name) for name in names]
    original_step = env.step
    original_set = env.set_init_state
    counters = {"reset": 0, "step": 0}

    def write(row):
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def snapshot():
        sim = env.sim
        poses = [state.get_geom_state() for state in states]
        positions = [pose["pos"] for pose in poses]
        contact = bool(states[1].check_contact(states[0]))
        gripper = core.robots[0].gripper
        return {
            "sim_time": float(sim.data.time),
            "positions": {n: p.tolist() for n, p in zip(names, positions)},
            "quaternions": {n: p["quat"].tolist() for n, p in zip(names, poses)},
            "linear_velocity": {
                n: sim.data.get_body_xvelp(obj.root_body).tolist()
                for n, obj in zip(names, objects)
            },
            "object_contact": contact,
            "xy_distance": float(np.linalg.norm(positions[0][:2] - positions[1][:2])),
            "height_condition": bool(positions[0][2] >= positions[1][2]),
            "on_predicate": bool(states[1].check_ontop(states[0])),
            "check_success": bool(env.check_success()),
            "gripper_qpos": [float(sim.data.get_joint_qpos(j)) for j in gripper.joints],
            "gripper_contact": {
                n: bool(core.check_contact(gripper.contact_geoms, obj))
                for n, obj in zip(names, objects)
            },
        }

    def set_state(state):
        result = original_set(state)
        counters["reset"] += 1
        counters["step"] = 0
        write({"event": "set_init_state", **counters, **snapshot()})
        return result

    def step(action):
        result = original_step(action)
        counters["step"] += 1
        write({
            "event": "step", **counters, "action": np.asarray(action).tolist(),
            "returned_reward": float(result[1]), "returned_done": bool(result[2]),
            **snapshot(),
        })
        return result

    env.set_init_state = set_state
    env.step = step
    return env
