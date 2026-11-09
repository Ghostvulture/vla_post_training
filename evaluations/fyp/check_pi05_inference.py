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

"""Server-only synthetic inference check; never report it as LIBERO evaluation."""

import argparse
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this model-loading check in a Slurm GPU allocation")
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    from rlinf.models.embodiment.openpi.checkpoint import load_base_safetensors
    from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    torch.set_num_threads(8)
    torch.manual_seed(42)
    set_torch_compile(False)
    config = Pi0Config(pi05=True, action_horizon=50, action_dim=32, max_token_len=200)
    policy = Pi0(config)
    load_base_safetensors(policy, args.model / "model.safetensors")
    for name, parameter in policy.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Nonfinite checkpoint parameter: {name}")
    policy.to_bfloat16_for_selected_params("bfloat16")
    policy = policy.to("cuda").eval()
    observation = config.fake_obs(batch_size=1)
    values = {}
    for field in dataclasses.fields(observation):
        value = getattr(observation, field.name)
        if isinstance(value, dict):
            value = {key: tensor.to("cuda") for key, tensor in value.items()}
        elif isinstance(value, torch.Tensor):
            value = value.to("cuda")
        values[field.name] = value
    observation = dataclasses.replace(observation, **values)
    noise = torch.randn(1, 50, 32, device="cuda")
    times = []
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(2):
            torch.cuda.synchronize()
            start = time.monotonic()
            actions = policy.sample_actions(
                observation, noise=noise.clone(), num_steps=10
            )
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            if actions.shape != (1, 50, 32) or not torch.isfinite(actions).all():
                raise ValueError("Unexpected shape or nonfinite inference actions")
            outputs.append(actions.cpu())
    delta = (outputs[0] - outputs[1]).abs().max().item()
    report = {
        "stage": "synthetic_inference_complete",
        "model": str(args.model),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in policy.parameters()),
        "action_shape": list(outputs[0].shape),
        "finite_actions": True,
        "fixed_noise_repeat_max_difference": delta,
        "seconds": times,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "libero_episodes": 0,
        "limitation": "Synthetic images and token IDs; no tokenizer, control mapping, or task performance validated",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
