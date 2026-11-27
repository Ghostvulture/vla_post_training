# RoboTwin base/PPO comparison / 基座与 PPO 配对评估

Compare the base checkpoint and supplied PPO full weights on `move_stapler_pad/easy` with the same native adapter, measured proprioception, flow-ODE sampler, ten flow steps and a 300-anchor limit. Run on a server compute node inside Slurm with one visible GPU. The environment must have passed `robotwin_qualify.py` first.

在服务器计算节点上比较基座和收到的 PPO 权重。两个模型统一使用真实 RoboTwin 适配器、实测本体状态、十步 flow-ODE 采样和每回合最多 300 个动作。先通过 `robotwin_qualify.py` 的真实相机与规划检查，再在单 GPU Slurm 作业内执行：

```bash
python evaluations/fyp/robotwin_compare.py \
  --source "$FYP_XVLA_SOURCE" \
  --robotwin "$FYP_ROBOTWIN_ROOT" \
  --base "$FYP_BASE_MODEL" \
  --ppo "$FYP_PPO_DIRECTORY" \
  --output "$FYP_RESULT_DIRECTORY" \
  --seeds 2053 2054
```

Set these paths before running. `source` is the collaborator's RLinf-VLA checkout at `eaa1920287708edd0f1bafb61f337309df9d1039`; `robotwin` is the asset-linked checkout at `6e75b8b80ee32d2f495eb756cf5687e4e16f0eea`. `base` contains the verified `2toINF/X-VLA-RoboTwin2` revision `a157c580cfe6f9f445614490f3bec1b2f9ef9f18`. `ppo` contains `online-config.yaml` and `full_weights.pt`. Choose a new result directory each time. Use the existing environment; this command installs nothing and loads local model files only.

执行前设置上述路径。`source` 是指定版本的对方策略源码，`robotwin` 是链接好场景资源的任务源码，`base` 是已校验的基座目录，`ppo` 是含配置与完整权重的解压目录。输出目录必须是新目录。该命令沿用现有环境并从本地加载模型。

Each model runs in a fresh process. Matching reset seeds and independent policy-noise seeds are used; actual initial scene poses must match within the upstream adapter's 1e-3 tolerance. Exceptions are recorded separately and stop the comparison. They are never scored as task failures. The two pilot seeds can be included in a later ten-seed cohort, but cannot be counted twice as independent trials.

两个模型分别运行于新进程。除固定场景种子和动作噪声种子外，还核对实际初始场景。每回合保存 `episode.json`、`actions.npz` 与 `head.mp4`，每个模型有独立日志和 `run.json`，总表为 `summary.json`。视频记录初始画面及每个实际执行动作后的画面，10 fps 播放速度不代表真实物理时间。异常会停止比较，不计入失败率。扩大为十回合时，前两回合不能重复计数。

This is a standalone evaluation using the real policy, processor and environment adapter, without the distributed runner. It is a new development comparison: the original training code revision and original paired scene manifest were unavailable. The smoke witness covers rendering and one hold-pose action; complete model trajectories remain the acceptance test for this reused environment.

这是调用真实策略、处理器与环境适配器的独立评估，未启动分布式训练框架。因原训练版本及原配对场景清单缺失，结果属于当前协议下的新对照，不能称为原报告的精确复现。环境检查只覆盖渲染和一次保持姿态动作，完整模型轨迹仍需通过运行验证。

Local CPU checks / 本地 CPU 检查：

```bash
python3 -m py_compile evaluations/fyp/robotwin_compare.py evaluations/fyp/robotwin_qualify.py
python3 evaluations/fyp/robotwin_compare.py --help
python3 -m pytest -q tests/unit_tests/test_fyp_evaluation.py
```

These check CLI syntax, GPU selection and rejection of invalid paired results without loading a model or simulator. GPU execution is verified by the server run.

这些检查验证命令、GPU 选择及不完整或不匹配结果的拒绝逻辑，不加载模型或仿真器；真实执行由服务器作业验证。
