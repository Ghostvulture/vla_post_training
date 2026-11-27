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

"""Qualify an existing RoboTwin environment on one Slurm-allocated GPU."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

LOG = logging.getLogger(__name__)
SOURCE_COMMIT = "eaa1920287708edd0f1bafb61f337309df9d1039"
ASSETS_REVISION = "25ed6fa76343d7b900e2b44d39f32a0eae1950b1"


def stream_digest(path: Path) -> str:
    """Use the upstream asset validation algorithm on Python 3.10 as well."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def bind_allocated_gpu(torch) -> dict:
    """Select the Vulkan device by the UUID of the single visible CUDA GPU."""
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one GPU visible to PyTorch")
    gpu = torch.cuda.get_device_properties(0)
    uuid = str(gpu.uuid).lower().removeprefix("gpu-")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,driver_version", "--format=csv,noheader"],
        text=True,
    )
    assigned = [line.split(", ") for line in inventory.splitlines()
                if line.split(",")[0].lower().removeprefix("gpu-") == uuid]
    if len(assigned) != 1:
        raise RuntimeError(f"Cannot map allocated CUDA GPU UUID {uuid} to Vulkan")
    match = re.fullmatch(r"[0-9a-fA-F]+:([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-9a-fA-F])", assigned[0][1])
    if match is None:
        raise RuntimeError(f"Unexpected PCI bus address: {assigned[0][1]}")
    alias = "pci:" + ":".join(match.groups()[:2]) + "." + match.group(3)
    os.environ["FYP_ROBOTWIN_VULKAN_DEVICE"] = alias.lower()
    return dict(gpu=gpu.name, gpu_uuid=uuid, vulkan_device=alias,
                driver=assigned[0][2], python=sys.version, prefix=sys.prefix)


def main() -> None:
    """Extract existing easy assets, render, and exercise the native planner."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "robotwin", "inputs", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not os.environ.get("SLURM_JOB_ID") or os.uname().nodename == "4090node1":
        raise RuntimeError("Run inside a GPU Slurm allocation on a compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "job_id": os.environ["SLURM_JOB_ID"],
              "host": os.uname().nodename, "evaluation_episodes": 0}

    def save() -> None:
        (args.output / "qualification.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        import numpy as np
        from PIL import Image
        import torch

        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
        report.update(bind_allocated_gpu(torch))
        packages = ("torch", "torchvision", "transformers", "numpy", "sapien", "mplib", "nvidia-curobo", "warp-lang")
        report["versions"] = {p: importlib.metadata.version(p) for p in packages}
        save()
        sys.path.insert(0, str(args.source.resolve()))
        from tools.xvla.robotwin import job_runtime, protocol
        from tools.xvla.robotwin.bind_renderer import bind_renderer, probe

        if job_runtime.commit(args.source) != SOURCE_COMMIT:
            raise RuntimeError("Policy/environment source revision differs from the inspected revision")
        if job_runtime.commit(args.robotwin) != protocol.ROBOTWIN_COMMIT:
            raise RuntimeError("RoboTwin revision differs from the pinned task implementation")
        report.update(source_commit=SOURCE_COMMIT, robotwin_commit=protocol.ROBOTWIN_COMMIT)
        # The pinned asset helper uses hashlib.file_digest (added in Python 3.11).
        job_runtime.digest = stream_digest
        job_runtime.prepare_easy_assets(args.data.resolve(), args.inputs.resolve(), ASSETS_REVISION, args.robotwin.resolve())
        report["assets_ready"] = True
        save()

        generator = torch.Generator().manual_seed(2053)
        x = torch.randn(32, 32, generator=generator)
        actual = (x.cuda() @ x.cuda().T).cpu()
        torch.testing.assert_close(actual, x @ x.T, rtol=1e-4, atol=1e-4)
        bind_renderer(pin_legacy_renderer=True)
        probe()
        report["cuda_and_render_probe"] = "passed"
        save()

        # Load the real standalone adapter without distributed-worker registration.
        spec = importlib.util.spec_from_file_location(
            "fyp_native_robotwin", args.source / "rlinf/envs/sim/robotwin/xvla_env.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task = module.NativeTask(SimpleNamespace(
            robotwin_root=str(args.robotwin.resolve()), task_name="move_stapler_pad", condition="easy"
        ))
        try:
            task.reset(index=0, seed=2053)
            raw = task.observe()
            images, proprio = protocol.observation_values(raw)
            report["initial_scene"] = task.scene_signature()
            report["initial_success"] = bool(task.env.check_success())
            if report["initial_success"]:
                raise RuntimeError("Qualification scene unexpectedly starts in a success state")
            for camera, rgb in zip(protocol.CAMERAS, images):
                Image.fromarray(rgb).save(args.output / f"{camera}-before.png")
                if np.ptp(rgb) == 0:
                    raise RuntimeError(f"Native camera {camera} produced a constant frame")
            # Hold the measured end-effector poses through the actual action converter.
            command = protocol.actions_to_env(np.repeat(proprio[None], protocol.ANCHORS, axis=0))[0]
            with task.action_context():
                task.take_action(command)
            after_images, _ = protocol.observation_values(task.observe())
            Image.fromarray(after_images[0]).save(args.output / "head_camera-after.png")
            report["native_action_budget"] = vars(task.budget)
            if task.budget.physics_ticks <= 0:
                raise RuntimeError("Native action did not advance the physics simulation")
            if task.budget.left_plan_failures or task.budget.right_plan_failures:
                raise RuntimeError("Native hold-pose action reported a planner failure")
            report["status"] = "simulation_witness_passed"
            report["scope"] = "Real cameras and one hold-pose action; no model rollout or success-rate evidence"
        finally:
            task.close()
        save()
        LOG.info("SIMULATION_WITNESS_PASSED %s", args.output)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
