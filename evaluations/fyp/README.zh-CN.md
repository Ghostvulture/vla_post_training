# FYP：Evo-1 原模型评估

[English](README.md)

本文说明如何在本地修改和检查评估代码，再通过 Git 同步到服务器，完成 Evo-1 的六任务原模型评估。先在服务器运行 2 集 smoke test，确认模型、渲染和结果记录能连通，再运行每任务 10 集的基线。

当前还没有配置 SSH、实际服务器目录或项目的 `origin`，也没有完成服务器验证。本页命令是操作步骤，不代表已经获得成功率。服务器示例目录 `/srv/fyp` 必须换成你有写权限的实际目录。

## 本地与服务器分工

把本地作为代码编辑端，把服务器作为模型和仿真运行端。这里的本地检查只能验证参数、命令生成、日志处理和初始状态选择，不能证明模型能加载或机器人能完成任务。

| 工作 | 本地电脑 | 服务器 |
|---|---|---|
| 修改代码、配置、说明文档 | 在 Codex 中修改 | 通过 Git 接收 |
| CPU 单元测试、`--dry-run` | 运行 | 更新后可复查 |
| 下载模型和仿真 assets | 不下载 | 下载并保留 |
| 模型推理、MuJoCo 渲染、smoke test | 不运行 | 运行 |
| 正式评估、后续 RL 训练 | 不运行 | 运行 |
| 查看 CSV、JSON 和日志 | 可拉回小文件分析 | 保存原始记录 |

本阶段复用 [官方评估入口](../run_eval.sh) 和 [Evo-1 评估配置](../libero/libero_spatial_evo1_eval.yaml)。只评估已发布 checkpoint，不做 SFT，也不下载示范轨迹数据。安装器会下载运行 LIBERO 必需的仿真 assets，它们与 SFT 示范数据不同。

## 在本地检查代码

从本地仓库根目录运行 CPU 检查。使用已有的测试环境；如果缺少 `pytest` 等测试依赖，先按本地测试环境的要求安装，勿在本地运行服务器安装命令。

```bash
cd /home/xyz/code/works/VLA_test/RLinf
python3 -m pytest -q \
  tests/unit_tests/test_fyp_evaluation.py \
  tests/unit_tests/test_fyp_report.py \
  tests/unit_tests/test_libero_env.py
```

这些测试覆盖评估配置、结果汇总和 LIBERO 初始状态选择。它们不加载 Evo-1、不执行仿真，也不提供成功率。

用服务器路径生成待执行命令，确认任务和集数配置：

```bash
python3 evaluations/fyp/evaluate.py \
  --model-path /srv/fyp/models/Evo1_LIBERO \
  --evo-repo /srv/fyp/RLinf/.venv/Evo-1 \
  --vlm-path /srv/fyp/models/InternVL3-1B \
  --output-dir /srv/fyp/results/evo1-baseline-batch1 \
  --episodes 10 --trial-start 0 --num-envs 1 --gpu 0 \
  --budget-minutes 1200 --dry-run
```

`--dry-run` 只打印命令，不导入模型或仿真模块，也不写入输出目录。任务 ID 在服务器安装的 LIBERO 中按完整任务名精确匹配；本地打印的是占位符，不要把占位符当成可直接运行的最终命令。执行时使用下文的 Python 入口，让脚本解析任务 ID。

## 通过 Git 同步代码

本地已使用独立分支 `codex/evo1-baseline`，`upstream` 指向官方 RLinf。准备自己的远程仓库 URL，将它设为 `origin`；不要向 `upstream` 推送项目改动。先检查并提交需要同步的文件，提交使用你的 Git 身份和 `git commit -s`。

首次发布分支时，在本地运行：

```bash
FYP_REMOTE_URL='git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git'
git remote add origin "$FYP_REMOTE_URL"
git push -u origin codex/evo1-baseline
```

先替换 `FYP_REMOTE_URL`。`git push` 只上传已提交的代码，不会上传未提交修改、模型或结果。如果 `origin` 已存在，先用 `git remote -v` 核对，避免重复添加。

配置好 SSH 后，从本地登录服务器：

```bash
FYP_SSH_HOST='your-ssh-alias'
ssh "$FYP_SSH_HOST"
```

将 `FYP_SSH_HOST` 换成已配置的 SSH alias，或 `用户名@服务器地址`。SSH 密钥由本地 SSH 配置管理，不写入仓库。后面的安装和评估命令均在服务器终端执行。

首次在服务器获取代码：

```bash
mkdir -p /srv/fyp
cd /srv/fyp
FYP_REMOTE_URL='git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git'
git clone --branch codex/evo1-baseline "$FYP_REMOTE_URL" RLinf
cd /srv/fyp/RLinf
```

服务器需要能读取这个远程仓库。后续更新时，本地完成测试、提交并 `git push`；等服务器当前任务结束后，在服务器仓库中执行 `git pull --ff-only`，再启动下一批。不要在正在运行的评估目录中改代码或拉取更新。实际评估要求服务器工作区干净，未提交修改会阻止启动。

## 在服务器安装环境和模型

先确认分配到的 NVIDIA GPU、可用显存和磁盘空间，再安装与当前 RLinf 版本配套的 Evo-1/LIBERO 环境。

```bash
cd /srv/fyp/RLinf
nvidia-smi
df -h /srv/fyp
bash requirements/install.sh embodied --model evo1 --env libero --no-root
source .venv/bin/activate
```

这是 [官方安装器](../../requirements/install.sh) 支持的组合。`--no-root` 跳过系统依赖安装，仅适用于管理员已备好系统依赖的服务器；它不会补齐缺失的系统库。若出现缺库或编译工具错误，先解决服务器环境，再重试。网络需要镜像时可按官方说明添加 `--use-mirror`。尽量使用不含空格的仓库、模型和输出路径。

安装器把 RLinf 的 Evo-1 fork 安装到 `.venv/Evo-1`，设置过 `EVO1_PATH` 时则使用该位置。确认实际目录后再传给评估入口：

```bash
FYP_EVO_REPO="${EVO1_PATH:-$PWD/.venv/Evo-1}"
test -d "$FYP_EVO_REPO"
realpath "$FYP_EVO_REPO"
```

使用自定义 `--venv` 时，按实际安装位置调整 `FYP_EVO_REPO`。这里的路径来自安装器规则，不代表当前服务器已安装成功。

下载 [Evo1_LIBERO](https://huggingface.co/MINT-SJTU/Evo1_LIBERO) 与其基础 VLM [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B)：

```bash
mkdir -p /srv/fyp/models /srv/fyp/results
hf download MINT-SJTU/Evo1_LIBERO \
  --local-dir /srv/fyp/models/Evo1_LIBERO
hf download OpenGVLab/InternVL3-1B \
  --local-dir /srv/fyp/models/InternVL3-1B
```

`hf` 由安装环境中的 `huggingface_hub` 提供。Evo checkpoint 目录应含 `config.json`、`norm_stats.json` 和 `mp_rank_00_model_states.pt`；基础 VLM 单独存放。两者都传入本地服务器路径，避免模型加载时临时下载。模型及仿真 assets 留在服务器，不加入 Git。

## 在服务器运行小规模验证

先只运行 S1 的 2 集并保存视频。定义共享参数后，后续命令都使用同一组模型路径和 GPU 配置：

```bash
cd /srv/fyp/RLinf
source .venv/bin/activate
FYP_EVO_REPO="${EVO1_PATH:-$PWD/.venv/Evo-1}"
FYP_EVAL_ARGS=(
  --model-path /srv/fyp/models/Evo1_LIBERO
  --evo-repo "$FYP_EVO_REPO"
  --vlm-path /srv/fyp/models/InternVL3-1B
  --num-envs 1
  --gpu 0
)
python evaluations/fyp/evaluate.py "${FYP_EVAL_ARGS[@]}" \
  --tasks S1 --episodes 2 --trial-start 0 --video \
  --output-dir /srv/fyp/results/evo1-smoke \
  --budget-minutes 60
```

这一步检查权重加载、动作执行、448×448 渲染、逐集记录和视频是否正常。成功率为 0 也可能是有效测量；崩溃、缺失集数或没有真实动作则需要先排查。`--video` 只允许每任务最多 2 集，正式基线默认关闭视频。

查看任务摘要中的 `wall_seconds`，用 `wall_seconds / 2` 估计这次观测到的每集时间。六任务各 10 集还会发生六次任务启动，任务长度也可能不同；据此预留启动时间和余量。这是预算估计，不是性能承诺。

`--budget-minutes` 是一次评估命令的整批时间上限，默认 1200 分钟。安装、下载、smoke test 和前一批不会自动计入下一批；本项目的总目标是含准备在内 24 小时，必须先扣除已用时间，再传入剩余预算。

`--gpu` 是 RLinf 可见设备中的编号。例如服务器已设置 `CUDA_VISIBLE_DEVICES=3` 时，用 `--gpu 0`；显存采样会映射到物理 GPU 3。`gpu-memory.csv` 约每 5 秒记录整张卡的用量，可能包含其他进程，不能作为本模型独占显存的精确峰值。每次任务使用独立的本地 Ray 实例和临时目录，不执行全机器的 `ray stop`。

## 在服务器运行六任务基线

smoke test 通过后，对固定的六任务各评估 10 集。任务定义保存在 [tasks.json](tasks.json)，运行前按任务完整名称匹配服务器的 LIBERO 任务 ID；匹配失败时先检查版本与任务表，不凭名称猜 ID。

| 标签 | 任务描述 | Suite |
|---|---|---|
| S1 | 把桌面中央黑碗放到盘子上 | LIBERO-Spatial |
| S2 | 把饼干盒旁黑碗放到盘子上 | LIBERO-Spatial |
| O1 | 把 alphabet soup 放进篮子 | LIBERO-Object |
| O2 | 把 cream cheese 放进篮子 | LIBERO-Object |
| G1 | 打开中间抽屉 | LIBERO-Goal |
| G2 | 把盘子推到炉子前 | LIBERO-Goal |

这些标签只是基线评估任务，不预先指定后训练的 A/B/C。省略 `--tasks` 会使用全部六任务；用 `--tasks S1 S2` 等组合可运行子集。

```bash
python evaluations/fyp/evaluate.py "${FYP_EVAL_ARGS[@]}" \
  --episodes 10 --trial-start 0 \
  --output-dir /srv/fyp/results/evo1-baseline-batch1 \
  --budget-minutes 1200
```

把 `1200` 改成实际剩余分钟数。脚本沿用官方 Evo 配置的 350 仿真步、448×448 图像和 14 步 action chunk；每次结果比较都应保持这些设置一致。若时间预算只够较小规模，改用 `--episodes 5` 和新目录 `evo1-baseline-fallback`，并按每任务 5 集报告。

需要扩充到每任务 20 集时，使用新目录和下一段初始状态编号：

```bash
python evaluations/fyp/evaluate.py "${FYP_EVAL_ARGS[@]}" \
  --episodes 10 --trial-start 10 \
  --output-dir /srv/fyp/results/evo1-baseline-batch2 \
  --budget-minutes 1200
```

第一批使用 trial 0–9，第二批使用 10–19。`--trial-start` 选择初始状态序号；仅改输出目录并不会产生新样本。输出目录必须是新目录，脚本拒绝覆盖已有 run。每次扩充前重新计算时间预算，并保持代码版本、模型文件、模型路径和评估协议一致；不要在同一路径替换权重后把结果合并。

## 查看并合并结果

每个 run 目录保留运行信息、逐集记录和汇总，供后续与后训练模型作相同条件下的比较。

```text
evo1-baseline-batch1/
├── run.json
├── summary.csv
├── S1/
│   ├── console.log
│   ├── command.json
│   ├── gpu-memory.csv
│   ├── episodes.csv
│   ├── summary.json
│   └── official_logs/
└── ...
```

视频启用时保存在官方日志目录内。先检查实际完成集数与错误状态，再解释成功率；超时、崩溃或未完成不能当成完整的 10 集实验。每任务 10 集只适合初步判断，不足以证明微小差异或跨任务迁移。

合并已保存的两批结果时运行：

```bash
python evaluations/fyp/evaluate.py --report \
  /srv/fyp/results/evo1-baseline-batch1 \
  /srv/fyp/results/evo1-baseline-batch2
```

`--report` 只读取已有结果并输出合并摘要，不加载模型、不启动仿真。它检查来源和评估协议是否一致，并拒绝重复或重叠的 trial；第一批与第二批合计每任务最多 20 个不同 trial。smoke test 的 S1 trial 0–1 与第一批重叠，不将 smoke 目录加入正式汇总。也可把 `run.json` 和各任务的 `summary.json`、`episodes.csv` 按原目录结构拉回本地，再运行相同的汇总命令。

## 基线完成后再规划 RL

先依据六任务原模型成绩、失败视频和实际运行成本选择后训练任务与保留评估任务，再确定服务器训练预算。为比较保留能力，保存原模型成绩、任务定义和 trial 范围；之后只在选定训练任务上更新模型，并用相同评估协议检查其他任务。

后续可从 [官方 Evo-1 GRPO 配置](../../examples/embodiment/config/libero_spatial_grpo_evo1.yaml) 制定训练方案：只在 A 上做 RL，冻结视觉语言骨干、更新动作头。先核验奖励、实际参数更新、权重同步与 checkpoint 恢复，再固定预算开展三个训练种子的重复实验。开始训练前需要确认 checkpoint、任务分组、可用 GPU、停止条件和评估频率。本入口只做原模型评估，不自动选择 A/B/C 或启动训练。
