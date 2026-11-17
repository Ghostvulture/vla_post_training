#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
gpu="${CUDA_VISIBLE_DEVICES:?}"
[[ "$gpu" != *,* ]]
runtime="${FYP_RUNTIME:?}"
storage="${FYP_STORAGE:?}"
work="$runtime/robodojo-official"
dojo="$work/RoboDojo"
parent="$storage/results/robodojo-xvla-rl-20010"
archive="$storage/results/robodojo-xvla-rep43-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
src="$HOME/RLinf/evaluations/fyp/robodojo"
git -C "$HOME/RLinf" diff --quiet a72b1662 -- evaluations/fyp/robodojo/flow_policy.py evaluations/fyp/robodojo/train_rl.py
git -C "$HOME/RLinf" rev-parse HEAD > "$archive/training-commit.txt"
export CONDA_ENVS_PATH="$work/envs" EVAL_ENV_TYPE=sim OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate XVLA-np126
export PATH="$HOME/miniforge3/condabin:$PATH"
cd "$dojo"
export PYTHONPATH="$dojo:$dojo/XPolicyLab:${PYTHONPATH:-}"
test "$(git rev-parse HEAD)" = "$(cat "$parent/robodojo-commit.txt")"
test "$(git -C XPolicyLab rev-parse HEAD)" = "$(cat "$parent/xpolicy-commit.txt")"

python -m pip freeze > "$archive/policy-dependencies.txt"
cat > "$archive/settings.json" <<'JSON'
{"collection_policy_seed":43,"collection_layout_seed":0,"collection_layout_range_inclusive":[0,19],"exploration_multiplier":0.1,"training_seed":42,"evaluation_policy_seed":1701,"evaluation_layout_seed":1,"primary_layout_range_inclusive":[10,29],"reference_job":20080,"learning_rates":[2e-6,2e-5]}
JSON
ref=RoboDojo-sim-arx_x5-ee-0
model_name="FypXvlaRL${SLURM_JOB_ID}"
adapter_dir="$dojo/XPolicyLab/policy/X_VLA"
recorder="$dojo/XPolicyLab/policy/$model_name"
mkdir "$recorder"
for script in setup_eval_env_client.sh setup_eval_policy_server.sh eval.sh; do
    cp "$adapter_dir/$script" "$recorder/$script"
done
sed "s/^policy_name: X_VLA$/policy_name: $model_name/" "$adapter_dir/deploy.yml" > "$recorder/deploy.yml"
cp "$src/flow_policy.py" "$recorder/model.py"
cp "$src/record_client.py" "$recorder/deploy.py"
touch "$recorder/__init__.py"
unset FYP_RL_ADAPTER
export FYP_ROLLOUT_DIR="$archive/train-traces" FYP_POLICY_SEED=43
export FYP_EXPLORATION_MULTIPLIER=0.1
echo "$FYP_EXPLORATION_MULTIPLIER" > "$archive/exploration-multiplier.txt"
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
assert n == r["eval_time"] == len(rows)
assert {x["layout_id"] for x in rows} == set(range(n))
assert abs(sum(x["success"] for x in rows)/n-r["success_rate"]) < 1e-8
print("Verified",sys.argv[1],sum(x["success"] for x in rows),"/",n,flush=True)
PY
    cp "$result" "$archive/$label.json"
}
run_eval "$model_name" stack_blocks "$ref" 0 20 collect
unset FYP_ROLLOUT_DIR

export FYP_POLICY_SEED=1701
unset FYP_EXPLORATION_MULTIPLIER
model_name=FypXvlaRL20010
cmp "XPolicyLab/policy/$model_name/model.py" "$src/flow_policy.py"
cmp "XPolicyLab/policy/$model_name/deploy.py" "$adapter_dir/deploy.py"
tail_seed=20080
layout_root="$dojo/Assets/Eval_Layout/RoboDojo/arx_x5"
for task in stack_blocks stack_bowls; do
    for idx in {25..29}; do
        test "$layout_root/1/${task}_$idx.json" -ef "$layout_root/$tail_seed/${task}_$idx.json"
    done
done
evaluate() {
    local label=$1 task seed count batch f
    export FYP_RL_ADAPTER="$2"
    for task in stack_blocks stack_bowls; do
        for batch in first25 last5; do
            seed=1
            count=25
            if [[ "$batch" = last5 ]]; then seed="$tail_seed"; count=5; fi
            export ROBODOJO_RUN_ID="fyp-$SLURM_JOB_ID-$label-$task-$batch"
            bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$model_name" --task "$task" --ckpt "$ref" --seed "$seed" --eval-num "$count" --env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu"
            f="eval_result/RoboDojo/$task/$model_name/arx_x5/${seed}_ckpt_name=${ref},action_type=ee/$ROBODOJO_RUN_ID/_result.json"
            cp "$f" "$archive/$label-$task-$batch.json"
        done
        python - "$archive/$label-$task" <<'PYCODE'
import json, sys
from pathlib import Path
prefix = sys.argv[1]
details = {}
for name, count, offset in [("first25", 25, 0), ("last5", 5, 25)]:
    path = Path(f"{prefix}-{name}.json")
    result = json.loads(path.read_text())
    rows = list(result["details"].values())
    assert len(rows) == result["eval_time"] == count
    assert {x["layout_id"] for x in rows} == set(range(count))
    assert all(isinstance(x["success"], bool) for x in rows)
    assert abs(sum(x["success"] for x in rows)/count-result["success_rate"]) < 1e-9
    for row in rows:
        canonical_id = offset + row["layout_id"]
        details[str(canonical_id)] = dict(row, native_layout_id=row["layout_id"],
                                           layout_id=canonical_id, batch=name)
assert set(map(int,details)) == set(range(30))
summary = dict(eval_time=len(details), success_rate=sum(x["success"] for x in details.values())/30,
               score=sum(x["score"] for x in details.values())*100/30, details=details)
Path(f"{prefix}.json").write_text(json.dumps(summary, indent=2))
print("Held-out successes", sum(x["success"] for x in details.values() if x["layout_id"] >= 10), "/ 20", flush=True)
PYCODE
    done
}

for setting in lr_low lr_original; do
    rate=2e-6
    if [[ "$setting" = lr_original ]]; then rate=2e-5; fi
    python "$src/train_rl.py" --arm shared --learning-rate "$rate" --base "$work/ckpt-100000" --data "$archive/train-traces" --result "$archive/collect.json" --output "$archive/$setting"
done
evaluate lr_low "$archive/lr_low"
evaluate lr_original "$archive/lr_original"
