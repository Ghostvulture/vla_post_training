#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
gpu="${CUDA_VISIBLE_DEVICES:?missing Slurm GPU allocation}"
[[ "$gpu" != *,* ]]
runtime="${FYP_RUNTIME:?}"
storage="${FYP_STORAGE:?}"
work="$runtime/robodojo-official"
dojo="$work/RoboDojo"
archive="$storage/results/robodojo-xvla-fsft-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
src="$HOME/RLinf/evaluations/fyp/robodojo"
git -C "$HOME/RLinf" rev-parse HEAD > "$archive/training-commit.txt"
git -C "$dojo" rev-parse HEAD > "$archive/robodojo-commit.txt"
git -C "$dojo/XPolicyLab" rev-parse HEAD > "$archive/xpolicy-commit.txt"
export CONDA_ENVS_PATH="$work/envs" EVAL_ENV_TYPE=sim OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate XVLA-np126
export PATH="$HOME/miniforge3/condabin:$PATH"
cd "$dojo"
export PYTHONPATH="$dojo:$dojo/XPolicyLab:${PYTHONPATH:-}"
ref=RoboDojo-sim-arx_x5-ee-0
model_name="FypXvlaRecord${SLURM_JOB_ID}"
adapter_dir="$dojo/XPolicyLab/policy/X_VLA"
recorder="$dojo/XPolicyLab/policy/$model_name"
mkdir "$recorder"
for script in setup_eval_env_client.sh setup_eval_policy_server.sh eval.sh; do
    cp "$adapter_dir/$script" "$recorder/$script"
done
sed "s/^policy_name: X_VLA$/policy_name: $model_name/" "$adapter_dir/deploy.yml" > "$recorder/deploy.yml"
cp "$src/record_policy.py" "$recorder/model.py"
cp "$src/record_client.py" "$recorder/deploy.py"
touch "$recorder/__init__.py"
export FYP_ROLLOUT_DIR="$archive/train-traces" FYP_POLICY_SEED=42
mkdir "$FYP_ROLLOUT_DIR"
counts=(--env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu")
run_eval() {
    local policy=$1 task=$2 checkpoint=$3 seed=$4 count=$5 label=$6
    export ROBODOJO_RUN_ID="fyp-${SLURM_JOB_ID}-$label"
    bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$policy" --task "$task" --ckpt "$checkpoint" --seed "$seed" --eval-num "$count" "${counts[@]}"
    local result="eval_result/RoboDojo/$task/$policy/arx_x5/${seed}_ckpt_name=${checkpoint},action_type=ee/$ROBODOJO_RUN_ID/_result.json"
    python - "$result" "$count" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
rows=list(r["details"].values())
n=int(sys.argv[2])
assert n == r["eval_time"] == len(rows) == len({x["layout_id"] for x in rows})
assert abs(sum(x["success"] for x in rows)/n-r["success_rate"]) < 1e-8
print("Verified",sys.argv[1],sum(x["success"] for x in rows),"/",n,flush=True)
PY
    cp "$result" "$archive/$label.json"
}
run_eval "$model_name" stack_blocks "$ref" 0 20 collect
trained="fyp-fsft-$SLURM_JOB_ID"
python "$src/train_fsft.py" --base "$work/ckpt-100000" --data "$FYP_ROLLOUT_DIR" --result "$archive/collect.json" --output "$archive/adapter"
ln -s "$archive/adapter" "$adapter_dir/checkpoints/$trained"
for task in stack_blocks stack_bowls; do
    run_eval X_VLA "$task" "$ref" 2 10 "base-$task"
    run_eval X_VLA "$task" "$trained" 2 10 "fsft-$task"
done
