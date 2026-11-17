#!/bin/bash
set -eo pipefail
phase=${1:?expected train1, eval0, eval16, or round2}
parent=${2:-}
case "$phase" in train1|eval0|eval16|round2) ;; *) exit 2 ;; esac
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
gpu=${CUDA_VISIBLE_DEVICES:?}
[[ "$gpu" != *,* ]]
runtime=${FYP_RUNTIME:?}
storage=${FYP_STORAGE:?}
work="$runtime/robodojo-official"
dojo="$work/RoboDojo"
src="$HOME/RLinf/evaluations/fyp/robodojo"
archive="$storage/results/robodojo-rounds-$phase-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
export CONDA_ENVS_PATH="$work/envs" EVAL_ENV_TYPE=sim OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate XVLA-np126
export PATH="$HOME/miniforge3/condabin:$PATH"
export PYTHONPATH="$src:$dojo:$dojo/XPolicyLab:${PYTHONPATH:-}"
cd "$dojo"
git -C "$HOME/RLinf" rev-parse HEAD > "$archive/training-commit.txt"
git rev-parse HEAD > "$archive/robodojo-commit.txt"
git -C XPolicyLab rev-parse HEAD > "$archive/xpolicy-commit.txt"
python -m pip freeze > "$archive/policy-dependencies.txt"
base="$work/ckpt-100000"
original="$storage/results/robodojo-xvla-rl-20010"
ref=RoboDojo-sim-arx_x5-ee-0
export FYP_EXPLORATION_MULTIPLIER=0.1
unset FYP_RL_ADAPTER FYP_FSFT_MODEL FYP_ROLLOUT_DIR
python - "$archive" "$phase" "$parent" "$base" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], 'protocol.json').write_text(json.dumps(dict(phase=sys.argv[2], parent=sys.argv[3], base=sys.argv[4],
    initial_collection_seed=42, round2_collection_seed=43, evaluation_policy_seed=1701,
    training_seed=42, steps_per_round=16, rl_lr=2e-6, fsft_lr=1e-4,
    rl_sampling='uniform 4 chunks without replacement', fsft_sampling='4 complete successful chunks with replacement',
    rl_exploration_multiplier=.1, fsft_exploration='native flow noise only',
    train_layouts=dict(seed=0, ids=list(range(20))),
    development_layouts=dict(seed=1, ids=list(range(10,30))),
    evaluation_batches='seed1 first25 + seed20080 last5; exclude canonical0-9 from main development rate'), indent=2))
PY
train() {
    local method=$1 round=$2 data=$3 result=$4 output=$5 behavior=$6 resume=${7:-}
    local continuation=()
    if [[ -n "$resume" ]]; then continuation=(--resume "$resume"); fi
    python "$src/train_rounds.py" --method "$method" --round "$round" --base "$base" --data "$data" --result "$result" --output "$output" --behavior-policy "$behavior" --steps 16 --audit-resume "${continuation[@]}"
}
if [[ "$phase" = train1 ]]; then
    for method in rl fsft; do
        train "$method" 1 "$original/train-traces" "$original/collect.json" "$archive/$method" "official-base-job20010"
    done
    echo "$archive" > "$archive/COMPLETE"
    exit 0
fi

adapter_dir="$dojo/XPolicyLab/policy/X_VLA"
policy="FypRounds$SLURM_JOB_ID"
for suffix in Eval Collect; do
    directory="$dojo/XPolicyLab/policy/$policy$suffix"
    mkdir "$directory"
    for script in setup_eval_env_client.sh setup_eval_policy_server.sh eval.sh; do
        cp "$adapter_dir/$script" "$directory/$script"
    done
    sed "s/^policy_name: X_VLA$/policy_name: $policy$suffix/" "$adapter_dir/deploy.yml" > "$directory/deploy.yml"
    cp "$src/round_policy.py" "$directory/model.py"
    if [[ "$suffix" = Collect ]]; then
        cp "$src/round_client.py" "$directory/deploy.py"
    else
        cp "$adapter_dir/deploy.py" "$directory/deploy.py"
    fi
    touch "$directory/__init__.py"
done
run_eval() {
    local suffix=$1 task=$2 seed=$3 count=$4 label=$5
    export ROBODOJO_RUN_ID="fyp-rounds-$SLURM_JOB_ID-$label"
    bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$policy$suffix" --task "$task" --ckpt "$ref" --seed "$seed" --eval-num "$count" --env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu"
    local result="eval_result/RoboDojo/$task/$policy$suffix/arx_x5/${seed}_ckpt_name=${ref},action_type=ee/$ROBODOJO_RUN_ID/_result.json"
    python - "$result" "$count" "$archive/$label.json" <<'PY'
import json, sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text()); n=int(sys.argv[2]); rows=list(r['details'].values())
assert n == r['eval_time'] == len(rows)
assert {x['layout_id'] for x in rows} == set(range(n))
assert all(isinstance(x['success'],bool) for x in rows)
assert abs(sum(x['success'] for x in rows)/n-r['success_rate']) < 1e-8
Path(sys.argv[3]).write_text(json.dumps(r,indent=2))
print('Verified',sys.argv[3],sum(x['success'] for x in rows),'/',n,flush=True)
PY
}
select_policy() {
    local method=$1 checkpoint=$2
    export FYP_METHOD="$method"
    unset FYP_RL_ADAPTER FYP_FSFT_MODEL
    if [[ "$checkpoint" != base ]]; then
        if [[ "$method" = rl ]]; then
            export FYP_RL_ADAPTER="$checkpoint/lora"
            test -s "$FYP_RL_ADAPTER/adapter_model.safetensors"
        else
            export FYP_FSFT_MODEL="$checkpoint/export"
            test -s "$FYP_FSFT_MODEL/config.json"
        fi
    fi
}
evaluate() {
    local label=$1 task
    unset FYP_ROLLOUT_DIR
    export FYP_POLICY_SEED=1701
    run_eval Eval stack_blocks 0 20 "$label-A-train"
    for task in stack_blocks stack_bowls; do
        for idx in {25..29}; do
            test "$dojo/Assets/Eval_Layout/RoboDojo/arx_x5/1/${task}_$idx.json" -ef "$dojo/Assets/Eval_Layout/RoboDojo/arx_x5/20080/${task}_$idx.json"
        done
        run_eval Eval "$task" 1 25 "$label-$task-first25"
        run_eval Eval "$task" 20080 5 "$label-$task-last5"
        python - "$archive" "$label-$task" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); label=sys.argv[2]; primary=[]; extra=[]
for batch,offset in [('first25',0),('last5',25)]:
    result=json.loads((root/f'{label}-{batch}.json').read_text())
    for row in result['details'].values():
        row=dict(row,native_layout_id=row['layout_id'],layout_id=row['layout_id']+offset,batch=batch)
        (primary if row['layout_id']>=10 else extra).append(row)
assert {r['layout_id'] for r in primary} == set(range(10,30))
summary=dict(eval_time=20,successes=sum(r['success'] for r in primary),success_rate=sum(r['success'] for r in primary)/20,
             details={str(r['layout_id']):r for r in primary},extra_development_0_9=extra)
(root/f'{label}.json').write_text(json.dumps(summary,indent=2))
print('Primary development',label,summary['successes'],'/20',flush=True)
PY
    done
}
for method in rl fsft; do
    case "$phase" in
        eval0)
            select_policy "$method" base
            evaluate "$method-step000"
            ;;
        eval16)
            select_policy "$method" "$parent/$method/step-016"
            evaluate "$method-step016"
            ;;
        round2)
            select_policy "$method" "$parent/$method/step-016"
            export FYP_ROUND=2 FYP_POLICY_SEED=43
            export FYP_ROLLOUT_DIR="$archive/$method-traces"
            export FYP_BEHAVIOR_POLICY="${FYP_RL_ADAPTER:-${FYP_FSFT_MODEL:-}}"
            test -n "$FYP_BEHAVIOR_POLICY"
            mkdir "$FYP_ROLLOUT_DIR"
            run_eval Collect stack_blocks 0 20 "$method-collect"
            data="$FYP_ROLLOUT_DIR"
            unset FYP_ROLLOUT_DIR
            train "$method" 2 "$data" "$archive/$method-collect.json" "$archive/$method" "$FYP_BEHAVIOR_POLICY" "$parent/$method/step-016"
            select_policy "$method" "$archive/$method/step-032"
            evaluate "$method-step032"
            ;;
    esac
done
echo "$archive" > "$archive/COMPLETE"
