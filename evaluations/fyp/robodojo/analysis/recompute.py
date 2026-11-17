"""Rebuild the numerical appendix from archived official episode and update records."""
import json
import math
import statistics
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report_dir", type=Path, help="Report output directory beside the archived run directories")
HERE = parser.parse_args().report_dir.resolve()
HERE.mkdir(parents=True, exist_ok=True)
RUNS = {
    19987: "xvla", 20003: "fsft", 20006: "rl", 20010: "rl",
    20016: "toggle", 20080: "lr", 20143: "rep43", 20306: "repeat",
}
data = {}
output = []
totals = {"采集": 0, "评测": 0}


def table(headers, rows):
    output.extend(["", "| " + " | ".join(headers) + " |",
                   "|" + "---|" * len(headers)])
    for row in rows:
        output.append("| " + " | ".join(map(str, row)) + " |")
    output.append("")


output.append("# 数值附录\n\n脚本读取各作业的 results.json。成功数取 details 中的 success，部分得分不计成功。每种设置只在对应表格行计算一次。")
for job, label in RUNS.items():
    path = Path(f"../robodojo-{label}-{job}/results.json")
    data[job] = json.loads((HERE / path).read_text())
    output.append(f"## {job}\n\n[原始 JSON]({path})")
    rows = []
    for key, value in data[job].items():
        if not isinstance(value, dict):
            continue
        result = value.get("result", value)
        if "details" not in result:
            continue
        rr = list(result["details"].values())
        count = len(rr)
        ids = {r["layout_id"] for r in rr}
        wins = {r["layout_id"] for r in rr if r["success"]}
        assert count == result["eval_time"] == len(ids)
        assert all(type(r["success"]) is bool for r in rr)
        assert math.isclose(len(wins) / count, result["success_rate"], abs_tol=1e-9)
        assert math.isclose(100 * statistics.mean(r["score"] for r in rr), result["score"], abs_tol=1e-8)
        phase = "采集" if key in ("collect", "official") else "评测"
        totals[phase] += count
        parts = []
        if job >= 20080 and phase == "评测":
            assert ids == set(range(30))
            parts = [f"{len(wins & set(range(10)))}/10", f"{len(wins & set(range(10, 30)))}/20"]
        rows.append([key,phase,f"{len(wins)}/{count}",f"{result['score']:.2f}",*(parts or ["—","—"]),
                     "、".join(map(str,sorted(wins))) or "无"])
    table(["记录键","用途","全组成功","官方均分（0–100）","0–9 成功","10–29 成功","成功布局 ID"],rows)
output.append(f"汇总为 {totals['采集']} 个采集回合和 {totals['评测']} 个评测回合。20070 的 25 回合在 20080 中仅计一次。")
assert totals == {"采集":80, "评测":831}


def wins(result):
    return {r["layout_id"] for r in result["details"].values() if r["layout_id"] >= 10 and r["success"]}


first = {42: data[20080], 43: data[20143]}
output.append("## 独立布局复测与跨采集批次汇总")
paired = []
means = []
for task in ("stack_blocks", "stack_bowls"):
    for rate in ("original", "low"):
        scores = [[], []]
        for seed, record in first.items():
            a = wins(record[f"lr_{rate}-{task}"])
            b = wins(data[20306][f"seed{seed}_{rate}-{task}"])
            scores[0].append(len(a)*5)
            scores[1].append(len(b)*5)
            paired.append([task,seed,rate,len(a),len(b),sorted(b-a),sorted(a-b),len(a ^ b)])
        for label, values in zip(("首测","复测"), scores):
            means.append([task,rate,label,*values,f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"])
table(["任务","采集种子","学习率组","首测成功/20","复测成功/20","新增 ID","失去 ID","翻转"],paired)
table(["任务","学习率组","测次","种子42（%）","种子43（%）","均值 ± 样本标准差（百分点）"],means)
output.append("标准差以两个采集批次为单位，分母 n−1。复测沿用原权重和原布局。original 为 2e-5，low 为 2e-6。")
output.append("## 每次优化更新")
train = { "20010-head":data[20010]["training"]["head"], "20010-shared":data[20010]["training"]["shared"],
          "20080-low":data[20080]["low_training"], "20143-original":data[20143]["lr_original_training"],
          "20143-low":data[20143]["lr_low_training"]}
fsft = data[20003]["training"]["steps"]
assert len(fsft)==32 and all(len(v["updates"])==32 for v in train.values())
def outside(update):
    ratios = update["log_ratios"]
    assert len(ratios)==4 and all(math.isfinite(x) for x in ratios)
    return sum(not math.log(.8)<=r<=math.log(1.2) for r in ratios)
table(["RL 检查点","超界次数/128","零梯度更新/32"],
      [[k,sum(map(outside,v["updates"])),sum(u["grad_norm"]==0 for u in v["updates"])] for k,v in train.items()])
output.append("下表 RL 单元格为该次更新的裁剪前梯度范数 / 超界动作块数（每步 4 个）。FSFT 单元格为监督损失和裁剪前梯度范数。")
rows=[]
for i,step in enumerate(fsft):
    row=[i+1, f"{step['loss']:.6f}",f"{step['grad_norm']:.6g}"]
    for record in train.values():
        u=record["updates"][i]
        row.append(f"{u['grad_norm']:.6g} / {outside(u)}")
    rows.append(row)
table(["更新序号","FSFT 损失","FSFT 梯度",*train],rows)
(HERE/"数值附录.md").write_text("\n".join(output)+"\n")
