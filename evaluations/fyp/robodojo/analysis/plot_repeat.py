"""Plot the four paired outcomes at each held-out layout."""
import json
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report_dir", type=Path, help="Report output directory beside the archived run directories")
here = parser.parse_args().report_dir.resolve()
here.mkdir(parents=True, exist_ok=True)
def load(folder):
    return json.loads((here.parent / folder / "results.json").read_text())

old = {42: load("robodojo-lr-20080"), 43: load("robodojo-rep43-20143")}
new = load("robodojo-repeat-20306")
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
colors = ["#eceff1", "#e7a341", "#4b8fad", "#183b4a"]
fig, axes = plt.subplots(2, 1, figsize=(10, 4.8))
for ax, task, title in zip(axes, ("stack_blocks", "stack_bowls"), ("堆积木", "叠碗")):
    values, labels = [], []
    for seed in old:
        for rate, name in (("original", "2e-5"), ("low", "2e-6")):
            a = old[seed][f"lr_{rate}-{task}"]["details"]
            b = new[f"seed{seed}_{rate}-{task}"]["details"]
            values.append([2*int(a[str(i)]["success"])+int(b[str(i)]["success"]) for i in range(10,30)])
            labels.append(f"{seed}   /   {name}")
    ax.imshow(values, cmap=ListedColormap(colors), vmin=-.5, vmax=3.5, aspect="auto")
    ax.set_title(title,loc="left",fontsize=12,fontweight="bold")
    ax.set_yticks(range(4), labels)
    ax.set_xticks(range(20),range(10,30))
    ax.set_xticks([i-.5 for i in range(21)],minor=True)
    ax.set_yticks([i-.5 for i in range(5)],minor=True)
    ax.grid(which="minor",color="white",linewidth=1)
    ax.tick_params(which="both",length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[1].set_xlabel("评测布局 ID")
handles = [Patch(color=c,label=l) for c,l in zip(colors,("两次均失败","复测新增成功","复测失去成功","两次均成功"))]
fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False)
fig.subplots_adjust(left=.14,right=.99,bottom=.20,top=.92,hspace=.48)
fig.savefig(here/"layout-repeat.png",dpi=190,bbox_inches="tight")
