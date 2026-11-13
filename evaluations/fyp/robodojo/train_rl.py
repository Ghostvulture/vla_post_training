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

"""One on-policy collection round, followed by clipped terminal-return policy updates."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from XPolicyLab.policy.X_VLA.model import XVLA, XVLAProcessor

from flow_policy import encode_inputs, exploration_scale, flow_step


def trajectory_chunks(root, official, horizon):
    details = {v["layout_id"]: v["success"] for v in official["details"].values()}
    seen = set()
    chunks = []
    steps_total = 0
    wins = sum(details.values())
    count = len(details)
    if count != 20 or not 0 < wins < count:
        raise ValueError(f"Need mixed verified terminal rewards, obtained {wins}/{count}")
    for batch, line in enumerate((root / "batches.jsonl").read_text().splitlines()):
        for row in json.loads(line):
            layout, steps, reward = row["layout"], row["steps"], row["success"]
            if row["unstable"] or layout in seen or details.get(layout) != reward:
                raise ValueError(f"Invalid trajectory: {row}")
            seen.add(layout)
            steps_total += steps
            advantage = float(reward) - (wins - int(reward)) / (count - 1)
            files = sorted((root / f"batch-{batch}" / f"env-{row['env']}").glob("*.npz"))
            if steps <= 0 or len(files) != math.ceil(steps / horizon):
                raise ValueError(f"Invalid recorded horizon: {row}")
            for i, path in enumerate(files):
                chunks.append((path, min(horizon, steps-i*horizon), advantage))
    if seen != set(details):
        raise ValueError("Incomplete terminal reward mapping")
    return chunks, steps_total


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "data", "result", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--arm", choices=("head", "shared"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    rng = random.Random(42)
    net = XVLA.from_pretrained(str(args.base), torch_dtype=torch.float32).cuda()
    processor = XVLAProcessor.from_pretrained(str(args.base / "processor"))
    results = json.loads(args.result.read_text())
    chunks, interactions = trajectory_chunks(args.data, results, net.num_actions)
    head = [k for k, _ in net.named_modules() if k.startswith("transformer.blocks.") and k.endswith(".attn.qkv")]
    vlm = [k for k, _ in net.named_modules() if k.startswith("vlm.language_model.model.encoder.layers.") and k.endswith((".self_attn.q_proj", ".self_attn.v_proj"))]
    if len(head) != 24 or len(vlm) != 24:
        raise ValueError(f"Unexpected projection counts: {len(head)}, {len(vlm)}")
    rank = 4 if args.arm == "head" else 3
    extra = vlm if args.arm == "shared" else []
    model = get_peft_model(net, LoraConfig(r=rank, lora_alpha=rank*2, target_modules=head+extra,
        rank_pattern={k: 2 for k in extra}, alpha_pattern={k: 4 for k in extra}))
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    if n_params != 393216:
        raise ValueError(f"Unmatched trainable parameter budget: {n_params}")
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0)
    history = []
    for update in range(32):
        opt.zero_grad()
        log_ratios = []
        for path, used, advantage in rng.sample(chunks, 4):
            with np.load(path, allow_pickle=False) as ex:
                inputs, state = encode_inputs(processor, ex["image"], ex["prompt"].item(), ex["proprio"], "cuda")
                xt, old, raw = [torch.from_numpy(ex[k]).cuda().unsqueeze(0) for k in ("xt", "mean", "raw")]
            mean = flow_step(net, net.forward_vlm(**inputs), state, xt, state.new_full((1,), 0.1))
            scale = exploration_scale(mean)
            if update == 0 and not torch.allclose(mean, old, rtol=1e-5, atol=2e-5):
                raise ValueError("Behavior mean replay mismatch before update")
            # Joint density of the executed prefix; unexecuted tail has zero credit.
            z = ((raw-old)/scale)[:, :used]
            shift = ((mean-old)/scale)[:, :used]
            log_ratio = (z * shift - 0.5 * shift.square()).sum()
            ratio = log_ratio.clamp(-20, 20).exp()
            loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
            (loss / 4).backward()
            log_ratios.append(log_ratio.detach().item())
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        history.append({"grad_norm": grad.item(), "log_ratios": log_ratios})
    if not any(x["grad_norm"] > 0 for x in history) or not all(torch.isfinite(p).all() for p in params):
        raise ValueError("Invalid policy update")
    model.save_pretrained(args.output)
    (args.output / "training.json").write_text(json.dumps({"arm": args.arm, "trainable": n_params,
        "env_steps": interactions, "chunks": len(chunks), "train_episodes": results["eval_time"],
        "train_success_rate": results["success_rate"], "updates": history}, indent=2))


if __name__ == "__main__":
    main()
