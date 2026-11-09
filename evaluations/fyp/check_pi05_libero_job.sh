#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
source "$FYP_RUNTIME/envs/rlinf-pi05-check/bin/activate"
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export RAY_ADDRESS=local RAY_DEDUP_LOGS=0
export RAY_TMPDIR
RAY_TMPDIR=$(mktemp -d /tmp/fyp-pi05-ray-XXXXXX)
cd "$HOME/RLinf"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -n "${SLURM_JOB_ID:-}"
test -z "$(git status --porcelain)"
export PI05_CHECK_OUTPUT="$FYP_STORAGE/results/pi05-base-check/libero-$SLURM_JOB_ID"
mkdir "$PI05_CHECK_OUTPUT"
python - <<'PY'
import json, os, subprocess
from pathlib import Path
from evaluations.fyp.evaluate import resolve_tasks
manifest = json.loads(Path('evaluations/fyp/tasks.json').read_text())
task = next(t for t in manifest if t['key'] == 'SG6')
resolved, _ = resolve_tasks([task], [0, 1])
root = Path(os.environ['PI05_CHECK_OUTPUT'])
(root / 'task-id.txt').write_text(str(resolved[0]['task_id']))
(root / 'protocol.json').write_text(json.dumps({
    'protocol': 'pi05_base weights + public LIBERO interface calibration',
    'task': resolved[0], 'trial_ids': [0, 1], 'planned_episodes': 2,
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'model': str(Path(os.environ['FYP_STORAGE']) / 'models/pi05_base_openpi'),
    'action_horizon': 50, 'executed_action_chunk': 5, 'denoising_steps': 10,
    'discrete_state_input': True, 'task_time_limit': 350,
    'weight_training': False,
}, indent=2))
PY
task_id=$(cat "$PI05_CHECK_OUTPUT/task-id.txt")
python -m pip freeze > "$PI05_CHECK_OUTPUT/packages.txt"
bash evaluations/run_eval.sh libero libero_pi05_base_calibrated_eval \
  '~cluster.component_placement' '+cluster.component_placement={env:0,rollout:0}' \
  "+env.eval.task_id_filter=[$task_id]" \
  "runner.logger.log_path=$PI05_CHECK_OUTPUT/official_logs" \
  'runner.logger.experiment_name=pi05-base-calibrated-SG6' \
  'env.eval.video_cfg.save_video=true' 2>&1 | tee "$PI05_CHECK_OUTPUT/console.log"
