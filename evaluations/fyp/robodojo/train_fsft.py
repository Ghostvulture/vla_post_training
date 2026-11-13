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

"""Fit action-transformer LoRA on complete chunks from successful X-VLA trials."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from XPolicyLab.policy.X_VLA.model import XVLA, XVLAProcessor


def verify_rollouts(data, official, horizon):
    details = {r["layout_id"]: r["success"] for r in official["details"].values()}
    seen = set()
    examples = []
    chunks_by_layout = {}
    for batch, line in enumerate((data / "batches.jsonl").read_text().splitlines()):
        for row in json.loads(line):
            layout = row["layout"]
            if row["unstable"] or layout in seen or details.get(layout) != row["success"]:
                raise ValueError(f"Ambiguous or unverified rollout {layout}: {row}")
            seen.add(layout)
            paths = sorted((data / f"batch-{batch}" / f"env-{row['env']}").glob("*.npz"))
            steps = row["steps"]
            if steps <= 0 or len(paths) != (steps + horizon - 1) // horizon:
                raise ValueError(f"Action count mismatch at layout {layout}: {steps}, {len(paths)}")
            chunks_by_layout[layout] = {"steps": steps, "chunks": len(paths), "success": row["success"]}
            if row["success"]:
                examples.extend(paths[:steps // horizon])
    if seen != set(details) or len(seen) != official["eval_time"] or not examples:
        raise ValueError(f"Insufficient verified success data: {len(seen)} layouts, {len(examples)} chunks")
    return examples, chunks_by_layout


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "data", "result", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.manual_seed(42)
    rng = random.Random(42)
    base = XVLA.from_pretrained(str(args.base), torch_dtype=torch.float32).cuda()
    official = json.loads(args.result.read_text())
    examples, rollouts = verify_rollouts(args.data, official, base.num_actions)
    processor = XVLAProcessor.from_pretrained(str(args.base / "processor"))
    targets = [n for n, m in base.named_modules() if n.startswith("transformer.blocks.") and n.endswith(".attn.qkv") and isinstance(m, torch.nn.Linear)]
    if len(targets) != len(base.transformer.blocks):
        raise RuntimeError(f"Unexpected action head layout: {targets}")
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=targets))
    model.eval()  # Freeze stochastic dropout, keep gradients on the adapters.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0)
    losses = []
    for _ in range(32):
        optimizer.zero_grad()
        total = 0.0
        for _ in range(4):
            with np.load(rng.choice(examples), allow_pickle=False) as ex:
                inputs = processor(images=[Image.fromarray(ex["image"])], language_instruction=ex["prompt"].item())
                inputs = {k: v.cuda() for k, v in inputs.items()}
                inputs["proprio"] = torch.from_numpy(ex["proprio"]).unsqueeze(0).cuda()
                inputs["domain_id"] = torch.zeros(1, device="cuda", dtype=torch.long)
                action = torch.from_numpy(ex["action"]).unsqueeze(0).cuda()
            if action.shape != (1, base.num_actions, 20) or not torch.isfinite(action).all():
                raise ValueError("Invalid continuous action target")
            loss = sum(model(**inputs, action=action).values())
            (loss / 4).backward()
            total += loss.item() / 4
        grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"loss": total, "grad_norm": grad.item()})
    if not any(x["grad_norm"] > 0 for x in losses) or not all(torch.isfinite(p).all() for p in trainable):
        raise RuntimeError("Invalid LoRA update")
    model.save_pretrained(args.output / "lora")
    # Use the same official loader for both arms of the paired evaluation.
    # Keeping the adapter metadata in a subdirectory avoids implicit PEFT loading.
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    (args.output / "processor").symlink_to((args.base / "processor").resolve())
    audit = {"seed": 42, "rank": 4, "source": str(args.result), "full_success_chunks": len(examples),
             "rollouts": rollouts, "steps": losses, "trainable_parameters": sum(p.numel() for p in trainable)}
    (args.output / "training.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
