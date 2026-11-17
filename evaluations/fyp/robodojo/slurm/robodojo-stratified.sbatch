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
archive="$storage/results/robodojo-xvla-stratified-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
src="$HOME/RLinf/evaluations/fyp/robodojo"
git -C "$HOME/RLinf" diff --quiet a72b1662 -- evaluations/fyp/robodojo/flow_policy.py
git -C "$HOME/RLinf" rev-parse HEAD > "$archive/evaluation-commit.txt"
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

ref=RoboDojo-sim-arx_x5-ee-0
adapter_dir="$dojo/XPolicyLab/policy/X_VLA"
results="$storage/results"
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



collection=${1:?expected collection seed}
case "$collection" in
  43) data="$results/robodojo-xvla-rep43-20143"; control="$data/lr_low" ;;
  42) data="$parent"; control="$results/robodojo-xvla-lr-20080/shared-low" ;;
  *) exit 2 ;;
esac
test -s "$control/adapter_model.safetensors"
printf '%s\n' "$collection" "$data" "$control" > "$archive/sources.txt"
python "$src/train_rl.py" --base "$work/ckpt-100000" --data "$data/train-traces" --result "$data/collect.json" --output "$archive/stratified" --arm shared --learning-rate 2e-6 --success-stratified
for arm in uniform stratified; do
    export FYP_RL_ADAPTER="$control"
    if [[ "$arm" == stratified ]]; then export FYP_RL_ADAPTER="$archive/stratified"; fi
    export ROBODOJO_RUN_ID="fyp-$SLURM_JOB_ID-$arm-training-layouts"
    bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$model_name" --task stack_blocks --ckpt "$ref" --seed 0 --eval-num 20 --env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu"
    cp "eval_result/RoboDojo/stack_blocks/$model_name/arx_x5/0_ckpt_name=${ref},action_type=ee/$ROBODOJO_RUN_ID/_result.json" "$archive/$arm-training-layouts.json"
    evaluate "$arm" "$FYP_RL_ADAPTER"
done
