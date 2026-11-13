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

"""X-VLA policy with an adaptable last flow step and optional Gaussian exploration."""

import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from XPolicyLab.policy.X_VLA.model import Model as OfficialPolicy


def exploration_scale(tensor):
    # XYZ in metres, 6D rotation components, and pre-sigmoid gripper logits.
    return tensor.new_tensor(([0.003] * 3 + [0.025] * 6 + [0.1]) * 2)


def encode_inputs(processor, image, prompt, proprio, device):
    inputs = processor(images=[Image.fromarray(image)], language_instruction=prompt)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs.pop("image_input")
    state = torch.as_tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0)
    return inputs, state


def flow_step(net, enc, state, xt, t):
    state, xt = net.action_space.preprocess(state, xt)
    return net.transformer(
        domain_id=torch.zeros(1, dtype=torch.long, device=xt.device),
        action_with_noise=xt, proprio=state, t=t, **enc,
    )


class Model(OfficialPolicy):
    def __init__(self, model_cfg):
        torch.manual_seed(int(os.environ["FYP_POLICY_SEED"]))
        super().__init__(model_cfg)
        adapter = os.environ.get("FYP_RL_ADAPTER")
        self.tuned = PeftModel.from_pretrained(self.model, adapter).eval() if adapter else None
        self.collect = bool(os.environ.get("FYP_ROLLOUT_DIR"))
        self.batch = -1
        self.new_batch = True

    def reset(self):
        super().reset()
        self.new_batch = True

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.new_batch:
            self.batch += 1
            self.chunk = 0
            self.new_batch = False
        self.record_ids = iter(self._latest_env_idx_list)
        actions = super().get_action_batch(env_idx_list=env_idx_list, **kwargs)
        self.chunk += 1
        return actions

    @torch.no_grad()
    def infer(self, observation, steps=None):
        from contextlib import nullcontext

        steps = int(self.model_cfg.get("steps", 10) if steps is None else steps)
        assert steps == 10 and int(self.model_cfg.get("domain_id", 0)) == 0
        image = observation["images"][0]
        prompt = observation["prompt"]
        inputs, state = encode_inputs(self.processor, image, prompt, observation["proprio"], self.device)
        net = self.model
        with self.tuned.disable_adapter() if self.tuned is not None else nullcontext():
            enc = net.forward_vlm(**inputs)
            noise = torch.randn(1, net.num_actions, net.action_space.dim_action, device=self.device)
            prediction = torch.zeros_like(noise)
            for i in range(steps, 0, -1):
                t = state.new_full((1,), i / steps)
                xt = noise * t.view(1, 1, 1) + prediction * (1 - t).view(1, 1, 1)
                if i > 1:
                    prediction = flow_step(net, enc, state, xt, t)
        if self.tuned is not None:
            enc = net.forward_vlm(**inputs)
        mean = flow_step(net, enc, state, xt, t)
        raw = mean
        if self.collect:
            raw = mean + torch.randn_like(mean) * exploration_scale(mean)
            directory = Path(os.environ["FYP_ROLLOUT_DIR"]) / f"batch-{self.batch}" / f"env-{next(self.record_ids)}"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{self.chunk:04d}.npz").open("xb") as stream:
                np.savez_compressed(stream, image=image, prompt=prompt, proprio=observation["proprio"],
                                    xt=xt[0].cpu().numpy(), mean=mean[0].cpu().numpy(), raw=raw[0].cpu().numpy())
        return net.action_space.postprocess(raw)[0].cpu().numpy()
