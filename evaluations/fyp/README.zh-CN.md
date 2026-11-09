# FYP：Evo-1 原模型评估

[English](README.md)

本页说明如何从本地登录服务器、恢复已安装的 Evo-1 环境、通过 Slurm 提交 LIBERO 评估，以及查看和合并结果。本地用于修改代码和运行 CPU 检查；模型加载、仿真和评估在服务器上运行。当前使用的是针对 LIBERO 训练过的 `Evo1_LIBERO` checkpoint，本项目没有在评估前再次进行 SFT 或 RL。

## 每次登录后准备环境

在本地电脑的终端连接服务器。将示例替换为自己的 SSH 用户名和地址，或使用已配置的 SSH 别名：

```bash
ssh your-user@your-server
```

以下命令在服务器上执行。个人配置文件位于仓库外，记录服务器的数据目录、运行环境和节点；加载它会激活已安装的虚拟环境。每次开启新的服务器终端都需要重新运行这一组命令：

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
mkdir -p "$FYP_STORAGE/results"
FYP_SLURM=(
  --partition=GPU
  --nodelist="$FYP_NODE"
  --gres=gpu:4090:1
  --cpus-per-task=8
  --time=01:40:00
  --output="$FYP_STORAGE/results/slurm-eval-%j.log"
)
command -v python
squeue -u "$USER"
```

`FYP_STORAGE` 保存模型和结果，`FYP_RUNTIME` 保存环境与缓存；个人配置文件中的具体路径可用 `cat "$HOME/.config/fyp/evo1-env.sh"` 查看。`FYP_SLURM` 申请一张 4090 和 8 个 CPU，最长运行 100 分钟。`squeue` 显示当前账户的作业；若提交时出现 `AssocMaxSubmitJobLimit`，等待已有作业结束后再提交。服务器分配规则变化时，先根据集群规定调整该数组。

评估启动脚本是仓库外的 `$FYP_RUNTIME/fyp-eval-job.sh`。它会在作业内加载同一个环境，以离线方式读取服务器已有模型，再调用本项目的 [评估入口](evaluate.py)。作业内使用节点的 `/tmp` 存放临时文件，避免 Ray 的 socket 路径过长；模型、缓存和结果仍保存在用户数据目录。下文的 `--budget-minutes` 是单次评估的时间上限，需要短于 Slurm 的 `--time` 并留出启动和清理时间。

## 提交评估

### 小规模检查和视频

先用一个任务的两个初始状态检查模型、仿真和视频。每次使用新的输出目录，避免覆盖以前的结果：

```bash
FYP_RUN_ID="$(date +%Y%m%d-%H%M%S)"
sbatch "${FYP_SLURM[@]}" "$FYP_RUNTIME/fyp-eval-job.sh" \
  --tasks S1 --episodes 2 --trial-start 0 --video \
  --output-dir "$FYP_STORAGE/results/smoke-$FYP_RUN_ID" \
  --budget-minutes 60
```

`sbatch` 会返回作业号。`--video` 最多支持每个任务两集；作业结束后，视频位于该输出目录的 `S1/official_logs/video/eval/` 下。成功率为零也可能是有效结果；缺失轨迹、崩溃或无动作需要先排查。smoke test 与正式评估若使用相同任务和初始状态，不能作为额外样本合并。

### 六任务筛查

任务按完整名称匹配服务器安装的 LIBERO；标签保存在 [tasks.json](tasks.json)。

| 标签 | 任务 | Suite |
|---|---|---|
| S1 | 把桌面中央黑碗放到盘子上 | LIBERO-Spatial |
| S2 | 把饼干盒旁黑碗放到盘子上 | LIBERO-Spatial |
| O1 | 把 alphabet soup 放进篮子 | LIBERO-Object |
| O2 | 把 cream cheese 放进篮子 | LIBERO-Object |
| G1 | 打开中间抽屉 | LIBERO-Goal |
| G2 | 把盘子推到炉子前 | LIBERO-Goal |

省略 `--tasks` 会依次运行全部六个任务。下面从初始状态 0–9 中各跑 10 集；六个任务分别调用 RLinf 官方评估入口：

```bash
FYP_RUN_ID="$(date +%Y%m%d-%H%M%S)"
sbatch "${FYP_SLURM[@]}" "$FYP_RUNTIME/fyp-eval-job.sh" \
  --episodes 10 --trial-start 0 \
  --output-dir "$FYP_STORAGE/results/screen-0-9-$FYP_RUN_ID" \
  --budget-minutes 90
```

如果只需要指定任务，可以在同一命令中加入 `--tasks S1 S2`，输出目录仍需使用新名称。评估沿用 [Evo-1 配置](../libero/libero_spatial_evo1_eval.yaml) 的 448×448 图像、14 步动作块和每集 350 仿真步，使用 `libero_robot` 归一化 key。模型、代码版本或评估配置变化后，应单独记录并重新核对可比性。

扩展候选任务的标签为 `SG1 SG3 SG6 SG7 SS4 SS7`，具体目标见 [tasks.json](tasks.json)。筛查这些任务时，将命令改为 `--tasks SG1 SG3 SG6 SG7 SS4 SS7 --episodes 5 --trial-start 0`；如需补测 5 个不同状态，使用新的输出目录和 `--trial-start 5`。省略 `--tasks` 仍只运行上表中的六个原始基线任务。

时间允许时，等第一批完成后再提交状态 10–19：

```bash
FYP_RUN_ID="$(date +%Y%m%d-%H%M%S)"
sbatch "${FYP_SLURM[@]}" "$FYP_RUNTIME/fyp-eval-job.sh" \
  --episodes 10 --trial-start 10 \
  --output-dir "$FYP_STORAGE/results/screen-10-19-$FYP_RUN_ID" \
  --budget-minutes 90
```

仅改变输出目录不会产生新的初始状态；`--trial-start` 才控制状态编号。两批必须使用相同 checkpoint、代码版本和评估协议，并且不得重复状态编号，才能用下文的报告命令合并。输出目录已存在时，启动器会拒绝覆盖。

## 查看作业、日志和结果

将作业号替换为 `sbatch` 返回的数字。`squeue` 查看运行状态，`tail` 跟踪 Slurm 日志，`sacct` 查看结束状态；按 `Ctrl+C` 只会退出日志查看。

```bash
FYP_JOB_ID=12345
squeue -j "$FYP_JOB_ID"
tail -f "$FYP_STORAGE/results/slurm-eval-$FYP_JOB_ID.log"
sacct -j "$FYP_JOB_ID" --format=JobID,State,ExitCode,Elapsed
```

每次评估的输出目录包含 `run.json`、`summary.csv`，以及各任务的 `episodes.csv`、`summary.json`、`console.log` 和 `gpu-memory.csv`。先核对 `status`、完成集数和实际状态编号，再解释成功率。环境异常、超时和未完成的轨迹不能按正常失败计入成功率。

查看本次已经使用的两批目录，并在两批均结束后合并：

```bash
cat "$FYP_STORAGE/results/evo1-baseline-batch1/summary.csv"
cat "$FYP_STORAGE/results/evo1-baseline-batch2/summary.csv"
python evaluations/fyp/evaluate.py --report \
  "$FYP_STORAGE/results/evo1-baseline-batch1" \
  "$FYP_STORAGE/results/evo1-baseline-batch2"
```

其他批次将路径替换为实际输出目录。报告命令只读取已有文件，不加载模型或启动仿真；它检查来源与协议，并拒绝重复的初始状态。逐集 `success` 记录 350 步内是否曾达到成功条件，`success_at_end` 记录最后一步是否仍满足条件。两者含义不同；每任务 10–20 个固定初始状态适合初筛，不足以证明真实成功率或微小差异。

需要停止自己的作业时，先用 `squeue -u "$USER"` 核对作业号，再运行 `scancel "$FYP_JOB_ID"`。停止后检查已有 `run.json` 和逐集记录；未完成的批次应标记为不完整。

## 同步代码与本地检查

本地项目是独立的 `RLinf` 仓库。修改后先检查并只暂存需要提交的文件，例如：

```bash
cd /home/xyz/code/works/VLA_test/RLinf
git status --short
git add evaluations/fyp/README.zh-CN.md evaluations/fyp/README.md
git commit -s -m "docs(fyp): update evaluation guide"
git push origin main
```

服务器任务运行时不要更新代码或模型。所有任务结束后，在服务器仓库检查状态并快进更新：

```bash
cd "$HOME/RLinf"
squeue -u "$USER"
git status --short
git pull --ff-only
```

提交与拉取仅同步代码；模型、结果、个人路径和 SSH 密钥留在仓库外。本地 CPU 检查可运行：

```bash
python3 -m pytest -q \
  tests/unit_tests/test_fyp_evaluation.py \
  tests/unit_tests/test_fyp_report.py \
  tests/unit_tests/test_libero_env.py
```

这些检查不加载模型或执行仿真。环境已经安装时，无需在每次登录后重复运行 `requirements/install.sh` 或重新下载模型。后续 RL 需先根据基线选择任务并制定训练预算；评估脚本不会启动训练。

## 迁移或恢复服务器配置

当前服务器已经准备好个人配置和启动脚本。如果更换服务器，先在仓库外重新设置 `FYP_STORAGE`、`FYP_RUNTIME`、`FYP_NODE`、缓存目录，并激活实际安装的 `rlinf-evo1` 环境。参考下面的模板，将示例路径和节点替换为新服务器实际值后，保存为 `~/.config/fyp/evo1-env.sh`：

```bash
export FYP_STORAGE='/path/to/fyp'
export FYP_RUNTIME='/path/to/fyp-runtime'
export FYP_NODE='your-gpu-node'
export HF_HOME="$FYP_RUNTIME/cache/huggingface"
export TORCH_HOME="$FYP_RUNTIME/cache/torch"
export TORCH_EXTENSIONS_DIR="$FYP_RUNTIME/cache/torch-extensions"
export UV_CACHE_DIR="$FYP_RUNTIME/cache/uv"
export PIP_CACHE_DIR="$FYP_RUNTIME/cache/pip"
export TMPDIR="$FYP_RUNTIME/tmp"
source "$FYP_RUNTIME/envs/rlinf-evo1/bin/activate"
```

在新服务器首次安装时，先设置模板中的路径变量，确认系统依赖已由管理员安装，再运行官方安装器。已有环境不需要重复安装：

```bash
cd "$HOME/RLinf"
mkdir -p "$FYP_RUNTIME/tmp" "$FYP_STORAGE/models" "$FYP_STORAGE/results"
export PATH="$HOME/.local/bin:$PATH"
TMPDIR="$FYP_RUNTIME/tmp" UV_PYTHON_INSTALL_DIR="$FYP_RUNTIME/uv-python" \
  bash requirements/install.sh embodied --model evo1 --env libero \
  --no-root --use-mirror --venv "$FYP_RUNTIME/envs/rlinf-evo1"
source "$HOME/.config/fyp/evo1-env.sh"
```

`--no-root` 不会安装缺失的系统依赖。需要下载权重时，在服务器的个人数据目录运行下面的命令；运行中的模型目录不得被替换：

```bash
hf download MINT-SJTU/Evo1_LIBERO --local-dir "$FYP_STORAGE/models/Evo1_LIBERO"
hf download OpenGVLab/InternVL3-1B --local-dir "$FYP_STORAGE/models/InternVL3-1B"
```

确认 Evo-1 fork 位于 `$FYP_RUNTIME/envs/rlinf-evo1/Evo-1`。如果仓库外的作业脚本丢失，在加载个人配置后可按以下内容重建：

```bash
cat > "$FYP_RUNTIME/fyp-eval-job.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$HOME/RLinf"
python evaluations/fyp/evaluate.py \
  --model-path "$FYP_STORAGE/models/Evo1_LIBERO" \
  --evo-repo "$FYP_RUNTIME/envs/rlinf-evo1/Evo-1" \
  --vlm-path "$FYP_STORAGE/models/InternVL3-1B" \
  --num-envs 1 --gpu 0 "$@"
SCRIPT
chmod 700 "$FYP_RUNTIME/fyp-eval-job.sh"
```

迁移后先检查模型路径、LIBERO 仿真 assets、节点和 GPU 分配，再跑两集 smoke test。模型与 assets 下载、环境安装应在服务器执行，并遵守存储节点的 `user_data` / `public_data` 规则。

## 近任务迁移试验：训练前筛选

本轮预算为训练 48 GPU 小时（最多两卡）及评估 8 GPU 小时（单卡）。先筛选 SG6 场景的六个目标变体：B1B/B1A/B1T 将 cream cheese 换成 butter/alphabet soup/tomato sauce；B2P 改为放到已有盘子；B2W/B2B 将碗换为托盘/篮子。物体替换会改变几何；新状态来自固定种子的环境 reset，不能称为原任务的相同物理状态。生成器记录初始化成功与接触检查，失败变体不能进入评分。视觉和训练布局扩展只在筛选通过后实施。

在已分配 GPU 的 Slurm 作业内运行以下生成命令；目录必须不存在。生成状态不加载 VLA。然后按前文单卡 sbatch 流程调用评估启动脚本，增加下面的参数：

```bash
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-transfer-v1"
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-transfer-v1" --validate-goals
```

```bash
--tasks B1B B1A B1T B2P B2W B2B \
--variant-root "$FYP_STORAGE/variants/sg6-transfer-v1" \
--episodes 10 --trial-start 0
```

### 2026-11-09 筛选结果与停止原因

本轮未启动 RL。唯一有效容器候选 B2P（同场景 cream cheese 放盘子）在开发状态 0–9 上曾成功 1/10，最后成功 0/10，没有达到预设 2–8/10 门槛。B2W 木托盘存在明显接触穿透；B2B 篮子会在初始化时挤飞酒瓶，即使静置后的接触检查通过也无效。生成器现检查静置过程的穿透和水平位移，并验证成功谓词可满足。三个物体候选通过环境检查，但因容器侧已无法组成配对，没有继续消耗算力评分。

完整成绩在 `$FYP_STORAGE/results/sg6-transfer-container-v2`，准入报告在 `$FYP_STORAGE/results/sg6-transfer-gate-v2.json`。`sg6-transfer-screen-v1` 是取消的旧批次，不计分；`sg6-transfer-container-video-v2` 是重复初始化的视频复核，不增加筛查样本。

训练初始化白名单 `env.train.train_trial_ids` 已加入环境接口，不能与评估模式或固定 reset ID 混用。布局随机化、训练启动配置、checkpoint 恢复与三模型比较尚未执行或验收；本页没有提供未经验证的训练命令。先修订任务配对，再继续这些步骤。此结果不能证明 RL 没有迁移能力。

## 扩展原模型筛查

按用户确认，完整筛查四类各两个候选及 SG6 参照，共九个条件，每个条件 10 次开发初始化。A1/A2 只将墙面改为深蓝/深绿，地面保持 light-gray，并使用原任务初始化；B1B/B1A 替换为 butter/alphabet soup；B2N/B2R 替换为白碗/瓷烤盅；G1/SG7 分别打开抽屉/打开炉灶。B 类使用生成的初始化，不能称为跨条件相同物理状态。保持官方 On 成功谓词，不将其解读为严格容器内部检测。

当前原模型是 MINT-SJTU/Evo1_LIBERO，视觉语言骨干为 OpenGVLab/InternVL3-1B，已针对 LIBERO 适配；本项目未执行 SFT 或 RL。有效场景即使达到 0/10 或 10/10 也完成并报告，不再因一个候选未通过训练门槛而提前结束其他类别。非法场景单列，不能计为策略失败。

在 Slurm GPU 作业内生成并验证四个 B 变体，然后按前文 sbatch 流程给评估启动脚本传入第二段参数。输出目录必须尚不存在；保留旧筛选结果。

```bash
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-expanded-20261109" \
  --keys B1B B1A B2N B2R
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-expanded-20261109" --validate-goals
```

```bash
--tasks SG6 A1 A2 B1B B1A B2N B2R G1 SG7 \
--variant-root "$FYP_STORAGE/variants/sg6-expanded-20261109" \
--episodes 10 --trial-start 0 --num-envs 2
```
