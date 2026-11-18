#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
[[ "${CUDA_VISIBLE_DEVICES:?}" != *,* ]]
work="${FYP_RUNTIME:?}/robodojo-official"
dojo="$work/RoboDojo"
results="${FYP_STORAGE:?}/results"
archive="$results/robodojo-bowl-input-replay-$SLURM_JOB_ID"
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
python "$src/audit_bowl_inputs.py" --base "$work/ckpt-100000" \
    --adapter "$results/robodojo-xvla-lr-20080/shared-low" \
    --diagnostic "$results/robodojo-bowls-diagnostic-20624" \
    --output "$archive/replay"
touch "$archive/COMPLETE"
