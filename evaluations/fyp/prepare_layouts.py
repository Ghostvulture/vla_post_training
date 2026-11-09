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

"""Validate a finite bank of SG6 training-only layout perturbations on the server."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

SOURCE_TRIALS = list(range(20))
REGIONS = {
    "cream_cheese_1": (-0.06, 0.12, -0.04, 0.14),
    "akita_black_bowl_1": (-0.10, -0.01, -0.08, 0.01),
}


def sample_xy(original, bounds, generator):
    """Sample at most 1 cm per axis, intersected with the task's placement region."""
    original = np.asarray(original)
    lower = np.asarray(bounds[:2])
    upper = np.asarray(bounds[2:])
    if np.any(original < lower - 1e-6) or np.any(original > upper + 1e-6):
        raise ValueError(f"Source position {original} outside region {bounds}")
    return generator.uniform(
        np.maximum(lower, original - 0.01), np.minimum(upper, original + 0.01)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(6)
    if task.name != "put_the_cream_cheese_in_the_bowl":
        raise ValueError(f"Unexpected SG6: {task.name}")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=448, camera_widths=448
    )
    source_states = suite.get_task_init_states(6)
    generator = np.random.default_rng(110942)
    records, bank = [], []
    try:
        env.reset()
        for trial in SOURCE_TRIALS:
            variations = []
            for variant in range(4):
                accepted = False
                for attempt in range(30):
                    env.set_init_state(source_states[trial])
                    displacements = {}
                    for name, bounds in REGIONS.items():
                        obj = env.env.get_object(name)
                        qpos = env.sim.data.get_joint_qpos(obj.joints[0]).copy()
                        old_xy = qpos[:2].copy()
                        qpos[:2] = sample_xy(old_xy, bounds, generator)
                        displacements[name] = (qpos[:2] - old_xy).tolist()
                        env.sim.data.set_joint_qpos(obj.joints[0], qpos)
                    env.sim.forward()
                    candidate = env.get_sim_state().copy()
                    initially_successful = bool(env.check_success())
                    ids = [env.env.obj_body_id[name] for name in env.env.objects_dict]
                    positions = env.sim.data.body_xpos[ids].copy()
                    worst_contact, max_motion = 0.0, 0.0
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
                        obs, _, _, _ = env.step(
                            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                        )
                        max_motion = max(
                            max_motion,
                            float(
                                np.linalg.norm(
                                    env.sim.data.body_xpos[ids, :2] - positions[:, :2],
                                    axis=1,
                                ).max()
                            ),
                        )
                    accepted = bool(
                        np.isfinite(candidate).all()
                        and not initially_successful
                        and not env.check_success()
                        and worst_contact > -0.005
                        and max_motion < 0.02
                    )
                    if accepted:
                        variations.append(candidate)
                        records.append(
                            {
                                "source_trial_id": trial,
                                "layout_id": variant,
                                "attempts": attempt + 1,
                                "displacements": displacements,
                                "worst_contact": worst_contact,
                                "max_planar_motion": max_motion,
                            }
                        )
                        if trial == 0:
                            Image.fromarray(obs["agentview_image"][::-1, ::-1]).save(
                                args.output_dir / f"layout-{variant}.png"
                            )
                        break
                if not accepted:
                    raise ValueError(
                        f"No legal layout for source {trial}, variant {variant}"
                    )
            bank.append(np.stack(variations))
            print(f"source={trial} validated_layouts={len(variations)}", flush=True)
        states = np.stack(bank)
        if any(len(np.unique(row, axis=0)) != 4 for row in states):
            raise ValueError("Duplicate layouts within a source state")
        record = {
            "valid": True,
            "task_id": 6,
            "source_trial_ids": SOURCE_TRIALS,
            "layouts_per_source": 4,
            "seed": 110942,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_bddl": str(bddl),
            "regions": REGIONS,
            "max_xy_axis_delta": 0.01,
            "yaw_changed": False,
            "records": records,
        }
        torch.save(
            {"source_trial_ids": SOURCE_TRIALS, "states": states, "validation": record},
            args.output_dir / "bank.pt",
        )
        (args.output_dir / "validation.json").write_text(json.dumps(record, indent=2))
    except Exception as exc:
        (args.output_dir / "validation.json").write_text(
            json.dumps(
                {"valid": False, "error": repr(exc), "records": records}, indent=2
            )
        )
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
