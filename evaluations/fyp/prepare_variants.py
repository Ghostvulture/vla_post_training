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

"""Build SG6 task variants on the server without modifying installed assets."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def variant_bddl(source: str, key: str) -> str:
    """Change one target identity while retaining the SG6 scene and regions."""
    objects = {"B1B": "butter", "B1A": "alphabet_soup", "B1T": "tomato_sauce"}
    containers = {"B2W": "wooden_tray", "B2B": "basket"}
    if key in objects:
        name = objects[key]
        text = source.replace("cream_cheese", name)
        language = f"put the {name.replace('_', ' ')} in the bowl"
    elif key in containers:
        name = containers[key]
        text = source.replace("akita_black_bowl", name)
        language = f"put the cream cheese in the {name.replace('_', ' ')}"
    elif key == "B2P":
        text = source.replace(
            "(On cream_cheese_1 akita_black_bowl_1)", "(On cream_cheese_1 plate_1)"
        )
        text = text.replace(
            "    cream_cheese_1\n    akita_black_bowl_1",
            "    cream_cheese_1\n    plate_1",
        )
        language = "put the cream cheese on the plate"
    else:
        raise ValueError(f"Unknown variant: {key}")
    return re.sub(r"\(:language[^)]+\)", f"(:language {language})", text)


def validate_goals(root):
    """Check that a centered placement can satisfy each variant's predicate."""
    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    sources = {"B1B": "butter_1", "B1A": "alphabet_soup_1", "B1T": "tomato_sauce_1"}
    targets = {"B2P": "plate_1", "B2W": "wooden_tray_1", "B2B": "basket_1"}
    for directory in sorted(root.iterdir()):
        if not (directory / "validation.json").is_file():
            continue
        record = json.loads((directory / "validation.json").read_text())
        if not record["valid"]:
            continue
        env = OffScreenRenderEnv(
            bddl_file_name=str(directory / "task.bddl"),
            camera_heights=448,
            camera_widths=448,
        )
        try:
            env.reset()
            states = torch.load(directory / "states.pt", weights_only=False)
            stability = []
            for trial, state in enumerate(states):
                env.set_init_state(state)
                ids = [env.env.obj_body_id[name] for name in env.env.objects_dict]
                initial_positions = env.sim.data.body_xpos[ids].copy()
                worst_contact = 0.0
                max_motion = 0.0
                for _ in range(10):
                    worst_contact = min(
                        worst_contact,
                        min(
                            (
                                float(env.sim.data.contact[i].dist)
                                for i in range(env.sim.data.ncon)
                            ),
                            default=0.0,
                        ),
                    )
                    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
                    motion = np.linalg.norm(
                        env.sim.data.body_xpos[ids, :2] - initial_positions[:, :2],
                        axis=1,
                    ).max()
                    max_motion = max(max_motion, float(motion))
                stability.append(
                    {
                        "trial_id": trial,
                        "worst_contact": worst_contact,
                        "max_planar_motion": max_motion,
                        "valid": worst_contact > -0.005 and max_motion < 0.02,
                    }
                )
            record["settling_checks"] = stability
            record["settling_valid"] = all(row["valid"] for row in stability)
            env.set_init_state(states[0])
            source = env.env.get_object(sources.get(directory.name, "cream_cheese_1"))
            target_name = targets.get(directory.name, "akita_black_bowl_1")
            target = env.env.get_object(target_name)
            target_pos = env.sim.data.body_xpos[env.env.obj_body_id[target_name]].copy()
            qpos = env.sim.data.get_joint_qpos(source.joints[0]).copy()
            qpos[:3] = target_pos
            qpos[2] += float(target.top_offset[2] - source.bottom_offset[2]) + 0.01
            env.sim.data.set_joint_qpos(source.joints[0], qpos)
            env.sim.forward()
            success = False
            for _ in range(40):
                obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
                success = success or bool(env.check_success())
            record["goal_witness_success"] = success
            record["goal_witness_note"] = (
                "Object placed above target and settled; validates predicate feasibility, not policy success or robot reachability"
            )
            Image.fromarray(obs["agentview_image"][::-1, ::-1]).save(
                directory / "goal-witness.png"
            )
            (directory / "validation.json").write_text(json.dumps(record, indent=2))
            print(directory.name, "goal witness", success, flush=True)
        finally:
            env.close()


def main():
    """Generate immutable state pools and inspect development initializations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-goals", action="store_true")
    args = parser.parse_args()
    if args.validate_goals:
        validate_goals(args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    source = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal/put_the_cream_cheese_in_the_bowl.bddl"
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for key in ("B1B", "B1A", "B1T", "B2P", "B2W", "B2B"):
        directory = args.output_dir / key
        directory.mkdir()
        bddl = directory / "task.bddl"
        bddl.write_text(variant_bddl(source.read_text(), key))
        env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=448, camera_widths=448
            )
            states, records = [], []
            for trial in range(50):
                seed = 110900 + trial
                env.seed(seed)
                env.reset()
                state = env.get_sim_state().copy()
                initial_success = bool(env.check_success())
                for _ in range(10):
                    obs, _, _, _ = env.step(
                        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                    )
                min_contact = min(
                    (
                        float(env.sim.data.contact[i].dist)
                        for i in range(env.sim.data.ncon)
                    ),
                    default=0.0,
                )
                valid = bool(
                    np.isfinite(state).all()
                    and not initial_success
                    and not env.check_success()
                    and min_contact > -0.005
                )
                records.append(
                    {
                        "trial_id": trial,
                        "seed": seed,
                        "initial_success": initial_success,
                        "settled_min_contact": min_contact,
                        "valid": valid,
                    }
                )
                states.append(state)
                if trial < 2:
                    Image.fromarray(obs["agentview_image"][::-1, ::-1]).save(
                        directory / f"initial-{trial}.png"
                    )
            unique = len({tuple(state.tolist()) for state in states}) == 50
            torch.save(np.asarray(states), directory / "states.pt")
            result = {
                "valid": all(r["valid"] for r in records) and unique,
                "key": key,
                "git_commit": commit,
                "source_bddl": str(source),
                "state_generation": "fresh seeded reset; not original SG6 states",
                "unique_states": unique,
                "states": records,
            }
        except Exception as exc:
            result = {
                "valid": False,
                "key": key,
                "git_commit": commit,
                "error": repr(exc),
            }
        finally:
            if env is not None:
                env.close()
        (directory / "validation.json").write_text(json.dumps(result, indent=2))
        print(key, result["valid"], result.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
