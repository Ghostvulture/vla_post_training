#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
gpu=${CUDA_VISIBLE_DEVICES:?}
[[ "$gpu" != *,* ]]
work="${FYP_RUNTIME:?}/robodojo-official"
dojo="$work/RoboDojo"
src="$HOME/RLinf/evaluations/fyp/robodojo"
archive="${FYP_STORAGE:?}/results/robodojo-bowls-diagnostic-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
export CONDA_ENVS_PATH="$work/envs" EVAL_ENV_TYPE=sim OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate XVLA-np126
export PATH="$HOME/miniforge3/condabin:$PATH"
export PYTHONPATH="$src:$dojo:$dojo/XPolicyLab:${PYTHONPATH:-}"
cd "$dojo"
git -C "$HOME/RLinf" rev-parse HEAD > "$archive/code-commit.txt"
git rev-parse HEAD > "$archive/robodojo-commit.txt"
git -C XPolicyLab rev-parse HEAD > "$archive/xpolicy-commit.txt"
python -m pip freeze > "$archive/policy-dependencies.txt"
conda run -n RoboDojo python -m pip freeze > "$archive/simulator-dependencies.txt"
export FYP_POLICY_SEED=1701
export FYP_RL_ADAPTER="$FYP_STORAGE/results/robodojo-xvla-lr-20080/shared-low"
unset FYP_ROLLOUT_DIR FYP_FSFT_MODEL FYP_EXPLORATION_MULTIPLIER
policy="FypDiagnostic$SLURM_JOB_ID"
directory="$dojo/XPolicyLab/policy/$policy"
original="$dojo/XPolicyLab/policy/X_VLA"
mkdir "$directory"
for script in setup_eval_env_client.sh setup_eval_policy_server.sh eval.sh; do
    cp "$original/$script" "$directory/$script"
done
sed "s/^policy_name: X_VLA$/policy_name: $policy/" "$original/deploy.yml" > "$directory/deploy.yml"
cp "$src/diagnostic_policy.py" "$directory/model.py"
cp "$src/diagnostic_client.py" "$directory/deploy.py"
touch "$directory/__init__.py"
# A new view contains links to exactly four existing layouts. The seed label
# selects this view; it does not generate fresh task geometry.
view_seed=$((1000000 + SLURM_JOB_ID))
python - "$dojo" "$view_seed" "$archive" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])/'Assets/Eval_Layout/RoboDojo/arx_x5'
view=root/sys.argv[2]; view.mkdir()
layouts=[13,14,15,25]
for slot,layout in enumerate(layouts):
    source=root/'1'/f'stack_bowls_{layout}.json'
    assert source.is_file()
    (view/f'stack_bowls_{slot}.json').symlink_to(source.resolve())
protocol=dict(task='stack_bowls', source_layout_seed=1, slot_to_layout=layouts,
              view_seed=int(sys.argv[2]), repeats=3, episodes=12, policy_seed=1701,
              adapter='robodojo-xvla-lr-20080/shared-low', extra_exploration=False,
              scope='selected-layout diagnosis, not population variance or historical batch replication',
              intervention='passive logging; original deploy loop and native global RNG retained')
Path(sys.argv[3],'protocol.json').write_text(json.dumps(protocol,indent=2))
PY
ref=RoboDojo-sim-arx_x5-ee-0
for repeat in 1 2 3; do
    export FYP_DIAGNOSTIC_DIR="$archive/repeat-$repeat"
    mkdir "$FYP_DIAGNOSTIC_DIR"
    export ROBODOJO_RUN_ID="fyp-diagnostic-$SLURM_JOB_ID-r$repeat"
    bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$policy" --task stack_bowls --ckpt "$ref" --seed "$view_seed" --eval-num 4 --env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu"
    result="eval_result/RoboDojo/stack_bowls/$policy/arx_x5/${view_seed}_ckpt_name=${ref},action_type=ee/$ROBODOJO_RUN_ID/_result.json"
    python - "$result" "$FYP_DIAGNOSTIC_DIR" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[2]); source=Path(sys.argv[1]); r=json.loads(source.read_text())
rows=list(r['details'].values())
assert r['eval_time']==len(rows)==4
assert {x['layout_id'] for x in rows}==set(range(4))
assert all(isinstance(x['success'],bool) for x in rows)
assert abs(sum(x['success'] for x in rows)/4-r['success_rate']) < 1e-8
client=[json.loads(x) for x in (root/'client.jsonl').read_text().splitlines()]
assert client[0]['event']=='initial' and client[-1]['event']=='complete'
assert len(client[0]['mapping'])==4
calls=[json.loads(x) for x in (root/'policy/calls.jsonl').read_text().splitlines()]
assert {x['env'] for x in calls}==set(range(4))
assert [x['call'] for x in calls]==list(range(len(calls)))
for x in calls:
    assert (root/'policy'/f"{x['call']:05d}.npz").stat().st_size>0
    assert (root/'policy'/f"{x['call']:05d}-rng.pt").stat().st_size>0
(root/'result.json').write_text(json.dumps(r,indent=2))
(root/'source-result.txt').write_text(str(source.resolve())+'\n')
(root/'COMPLETE').write_text('results, state log and policy call files verified\n')
print('Verified', root, 'successes', sum(x['success'] for x in rows), '/4', flush=True)
PY
done
echo "$archive" > "$archive/COMPLETE"
