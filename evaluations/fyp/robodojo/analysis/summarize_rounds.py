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

"""Summarize completed two-round phases without mixing extra development layouts."""

import argparse
import json
import math
import random
from pathlib import Path

SETS = {"A-train": range(20), "stack_blocks": range(10, 30), "stack_bowls": range(10, 30)}
LABELS = {"A-train": "A-Train", "stack_blocks": "A-Val", "stack_bowls": "B-dev"}


def wilson(wins, count):
    z = 1.959963984540054
    p = wins / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def read_evaluation(path, ids):
    data = json.loads(path.read_text())
    details = list(data["details"].values())
    actual = [r["layout_id"] for r in details]
    if len(actual) != len(set(actual)) or set(actual) != set(ids):
        raise ValueError(f"Wrong or duplicated layout membership: {path}")
    if data["eval_time"] != len(actual) or not all(type(r["success"]) is bool for r in details):
        raise ValueError(f"Invalid episode count or terminal reward: {path}")
    wins = sum(r["success"] for r in details)
    if not math.isclose(data["success_rate"], wins / len(actual), abs_tol=1e-9):
        raise ValueError(f"Rate and terminal rewards disagree: {path}")
    return dict(source=str(path.resolve()), wins=wins, n=len(actual), rate=wins / len(actual),
                wilson95=wilson(wins, len(actual)), outcomes={r["layout_id"]: r["success"] for r in details})


def paired(base, candidate):
    if set(base["outcomes"]) != set(candidate["outcomes"]):
        raise ValueError("Paired comparison requires identical layout membership")
    ids = sorted(base["outcomes"])
    gains = [i for i in ids if candidate["outcomes"][i] and not base["outcomes"][i]]
    losses = [i for i in ids if base["outcomes"][i] and not candidate["outcomes"][i]]
    differences = [int(candidate["outcomes"][i]) - int(base["outcomes"][i]) for i in ids]
    rng = random.Random(1701)
    draws = sorted(sum(rng.choices(differences, k=len(ids))) / len(ids) for _ in range(20000))
    return dict(delta=(len(gains)-len(losses))/len(ids), gained_layouts=gains, lost_layouts=losses,
                paired_bootstrap95=[draws[499], draws[19499]])


def read_training(path):
    data = json.loads(path.read_text())
    updates = data["updates"]
    if len(updates) != data["completed_updates"]:
        raise ValueError(f"Inconsistent update count: {path}")
    if data["skip_reason"] and updates:
        raise ValueError(f"Skipped round contains updates: {path}")
    if not data["skip_reason"] and len(updates) != data["planned_updates"]:
        raise ValueError(f"Incomplete training round: {path}")
    if updates and not data.get("resume_validation", {}).get("optimizer_and_rng_exact"):
        raise ValueError(f"Resume verification missing: {path}")
    if any(not math.isfinite(row["loss"]) or not math.isfinite(row["grad_norm"]) for row in updates):
        raise ValueError(f"Nonfinite update: {path}")
    return dict(source=str(path.resolve()), method=data["settings"]["method"], round=data["round"],
                learning_rate=data["settings"]["learning_rate"], episodes=len(data["episodes"]), successes=data["successes"],
                chunks=data["chunks"], env_steps=data["env_steps"], planned_updates=data["planned_updates"],
                completed_updates=data["completed_updates"], skip_reason=data["skip_reason"],
                zero_gradient_steps=[r["step"] for r in updates if r["grad_norm"] == 0],
                replay=data["replay"], action_conversion=data["action_conversion"],
                resume_validation=data.get("resume_validation"), export_validation=data.get("export_validation"))


def summarize(directories):
    evaluations, training, protocols = [], [], {}
    stages = {"eval0": 0, "eval16": 16, "round2": 32}
    for root in directories:
        if not (root / "COMPLETE").is_file():
            raise ValueError(f"Phase has no completion marker: {root}")
        protocol = json.loads((root / "protocol.json").read_text())
        phase = protocol["phase"]
        if phase in protocols:
            raise ValueError(f"Separate repeated evaluations instead of overwriting phase {phase}")
        protocols[phase] = protocol
        if phase in ("train1", "round2"):
            for method in ("rl", "fsft"):
                row = read_training(root / method / "training.json")
                if row["method"] != method or row["round"] != (1 if phase == "train1" else 2):
                    raise ValueError(f"Mismatched training identity: {root}")
                training.append(row)
        if phase in stages:
            step = stages[phase]
            for method in ("rl", "fsft"):
                for key, ids in SETS.items():
                    result = read_evaluation(root / f"{method}-step{step:03d}-{key}.json", ids)
                    evaluations.append(dict(method=method, step=step, split=LABELS[key], **result))
        elif phase != "train1":
            raise ValueError(f"Unknown phase {phase}")
    eval_protocols = [p for phase, p in protocols.items() if phase in stages]
    if eval_protocols:
        for key in ("evaluation_policy_seed", "train_layouts", "development_layouts", "evaluation_batches", "base"):
            if any(p[key] != eval_protocols[0][key] for p in eval_protocols):
                raise ValueError(f"Evaluation protocols differ: {key}")
    for candidate in evaluations:
        if candidate["step"]:
            base = next((r for r in evaluations if r["step"] == 0 and r["method"] == candidate["method"] and r["split"] == candidate["split"]), None)
            if base:
                candidate["vs_base"] = paired(base, candidate)
    return dict(protocols=protocols, training=training, evaluations=evaluations)


def write_report(summary, root):
    lines = ["# 两轮同源后训练阶段结果", "", "仅汇总所提供且通过完整性校验的阶段。额外开发布局 0–9 不计入主开发成绩；各方法使用自己的推理路径基座。", ""]
    lines += ["| 方法 | 轮次 | 采集成功 | 可用块 | 环境步 | 实际/计划更新 | 零梯度步 |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in summary["training"]:
        lines.append(f"| {r['method']} | {r['round']} | {r['successes']}/{r['episodes']} | {r['chunks']} | {r['env_steps']} | {r['completed_updates']}/{r['planned_updates']} | {r['zero_gradient_steps']} |")
        if r["skip_reason"]:
            lines.append(f"\n跳过 {r['method']} 第 {r['round']} 轮：{r['skip_reason']}。\n")
    lines += ["", "| 方法 | 累计预算步 | 集合 | 成功数 | 成功率及 Wilson 95% 区间 | 相对对应基座 | 改善/退步布局 |", "|---|---:|---|---:|---|---|---|"]
    for r in sorted(summary["evaluations"], key=lambda r: (r["method"], r["split"], r["step"])):
        low, high = r["wilson95"]
        delta, flips = "—", "—"
        if "vs_base" in r:
            comp = r["vs_base"]
            delta = f"{100*comp['delta']:+.1f} pp"
            flips = f"{comp['gained_layouts']} / {comp['lost_layouts']}"
        lines.append(f"| {r['method']} | {r['step']} | {r['split']} | {r['wins']}/{r['n']} | {100*r['rate']:.1f}% [{100*low:.1f}, {100*high:.1f}] | {delta} | {flips} |")
    lines += ["", "统计边界：Wilson 区间与配对 bootstrap 仅描述这批布局的二元结果；不包含重新采集训练、仿真随机性和检查点选择造成的不确定性。单次训练链不证明稳定改善。预算步与实际优化步分别报告，跳过更新不能当作完成了优化。采集成功率带各自探索机制，不能直接与评测成功率比较。", "", "所有来源、逐布局结果、配对区间及 GPU 验收数据见 `summary.json`。"]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")


def plot(summary, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, split in zip(axes, LABELS.values()):
        for method in ("rl", "fsft"):
            rows = sorted([r for r in summary["evaluations"] if r["method"] == method and r["split"] == split], key=lambda r: r["step"])
            if not rows:
                continue
            ax.errorbar([r["step"] for r in rows], [100*r["rate"] for r in rows],
                yerr=[[100*(r["rate"]-r["wilson95"][0]) for r in rows], [100*(r["wilson95"][1]-r["rate"]) for r in rows]],
                marker="o", capsize=3, label=method.upper())
        ax.set_title(split)
        ax.set_xlabel("Cumulative planned updates")
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_ylim(0, 100)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend()
    fig.suptitle("Development pilot: one training chain per method; Wilson intervals")
    fig.tight_layout()
    fig.savefig(root / "learning-curves.png", dpi=180)
    fig.savefig(root / "learning-curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = summarize(args.phase_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    write_report(result, args.output)
    if args.plot:
        plot(result, args.output)


if __name__ == "__main__":
    main()
