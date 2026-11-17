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
frozen="$storage/results/robodojo-xvla-toggle-20016/both_off"
archive="$storage/results/robodojo-xvla-lr-$SLURM_JOB_ID"
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
model_name=FypXvlaRL20010
cmp "XPolicyLab/policy/$model_name/model.py" "$src/flow_policy.py"
cmp "XPolicyLab/policy/$model_name/deploy.py" XPolicyLab/policy/X_VLA/deploy.py
unset FYP_ROLLOUT_DIR
export FYP_POLICY_SEED=1701
python -m pip freeze > "$archive/policy-dependencies.txt"
python - <<'PY'
from pathlib import Path
import json
root = Path("Assets/Eval_Layout/RoboDojo/arx_x5")
for task in ("stack_blocks", "stack_bowls"):
    def poses(seed, layout):
        doc = json.loads((root / str(seed) / f"{task}_{layout}.json").read_text())
        return tuple(tuple(obj["default_pos"]) for group in doc["Rigid"].values() for obj in group if "default_pos" in obj)
    prior = {poses(seed, i) for seed, n in [(0, 20), (1, 10), (2, 10)] for i in range(n)}
    held = {poses(1, i) for i in range(10, 30)}
    assert () not in held and len(held) == 20 and held.isdisjoint(prior), task
    print(task, "verified 20 distinct held-out object layouts", flush=True)
PY
ref=RoboDojo-sim-arx_x5-ee-0
# The native tasks cap a call at 25. A second view of the unchanged
# official scene files lets the client run the remaining five for every arm.
layout_root="$dojo/Assets/Eval_Layout/RoboDojo/arx_x5"
tail_seed="$SLURM_JOB_ID"
tail_view="$archive/layouts-25-29"
mkdir "$tail_view"
for task in stack_blocks stack_bowls; do
    for idx in {25..29}; do
        source_layout="$layout_root/1/${task}_$idx.json"
        test -s "$source_layout"
        ln -s "$source_layout" "$tail_view/${task}_$idx.json"
    done
done
ln -sT "$tail_view" "$layout_root/$tail_seed"
previous="eval_result/RoboDojo/stack_blocks/$model_name/arx_x5/1_ckpt_name=${ref},action_type=ee/fyp-20070-frozen1-stack_blocks/_result.json"
evaluate() {
    local label=$1 task seed count batch f
    export FYP_RL_ADAPTER="$2"
    for task in stack_blocks stack_bowls; do
        for batch in first25 last5; do
            seed=1
            count=25
            if [[ "$batch" = last5 ]]; then seed="$tail_seed"; count=5; fi
            if [[ "$label-$task-$batch" = frozen1-stack_blocks-first25 ]]; then
                f="$previous"
            else
                export ROBODOJO_RUN_ID="fyp-$SLURM_JOB_ID-$label-$task-$batch"
                bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$model_name" --task "$task" --ckpt "$ref" --seed "$seed" --eval-num "$count" --env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu"
                f="eval_result/RoboDojo/$task/$model_name/arx_x5/${seed}_ckpt_name=${ref},action_type=ee/$ROBODOJO_RUN_ID/_result.json"
            fi
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
evaluate frozen1 "$frozen"
evaluate frozen2 "$frozen"
python "$src/train_rl.py" --arm shared --learning-rate 2e-6 --base "$work/ckpt-100000" --data "$parent/train-traces" --result "$parent/collect.json" --output "$archive/shared-low"
evaluate lr_low "$archive/shared-low"
evaluate lr_original "$parent/shared"
