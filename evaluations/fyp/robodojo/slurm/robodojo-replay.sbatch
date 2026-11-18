#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
[[ "${CUDA_VISIBLE_DEVICES:?}" != *,* ]]
work="${FYP_RUNTIME:?}/robodojo-official"
dojo="$work/RoboDojo"
results="${FYP_STORAGE:?}/results"
archive="$results/robodojo-policy-replay-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
src="$HOME/RLinf/evaluations/fyp/robodojo"
export CONDA_ENVS_PATH="$work/envs" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate XVLA-np126
export PYTHONPATH="$src:$dojo:$dojo/XPolicyLab:${PYTHONPATH:-}"
cd "$dojo"
git -C "$HOME/RLinf" rev-parse HEAD > "$archive/code-commit.txt"
python -m pip freeze > "$archive/dependencies.txt"
python "$src/audit_policy_replay.py" --base "$work/ckpt-100000" \
    --data "$results/robodojo-rounds-round2-20533/rl-traces" \
    --result "$results/robodojo-rounds-round2-20533/rl-collect.json" \
    --video-adapter "$results/robodojo-xvla-lr-20080/shared-low" \
    --rl-adapter "$results/robodojo-rounds-round2-20533/rl/step-032/lora" \
    --fsft-model "$results/robodojo-rounds-round2-20533/fsft/step-032/export" \
    --output "$archive/replay"
touch "$archive/COMPLETE"
