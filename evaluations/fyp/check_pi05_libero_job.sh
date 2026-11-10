#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
source "$FYP_RUNTIME/envs/rlinf-pi05-check/bin/activate"
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME="$FYP_RUNTIME/cache/openpi"
export RAY_ADDRESS=local RAY_DEDUP_LOGS=0
export RAY_TMPDIR
RAY_TMPDIR=$(mktemp -d /tmp/fyp-pi05-ray-XXXXXX)
cd "$HOME/RLinf"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -n "${SLURM_JOB_ID:-}"
test -z "$(git status --porcelain)"
export PI05_TASK="${1:-SG6}" PI05_EPISODES="${2:-2}"
case "$PI05_TASK" in SG6|S1|O2|G1) ;; *) echo "Unsupported screening task"; exit 2 ;; esac
case "$PI05_EPISODES" in 1|2|3|4|5) ;; *) echo "Expected 1-5 development episodes"; exit 2 ;; esac
export PI05_CHECK_OUTPUT="$FYP_STORAGE/results/pi05-base-check/libero-$SLURM_JOB_ID-$PI05_TASK"
mkdir "$PI05_CHECK_OUTPUT"
python - <<'PY'
import json, os, subprocess
from pathlib import Path
from evaluations.fyp.evaluate import resolve_tasks
manifest = json.loads(Path('evaluations/fyp/tasks.json').read_text())
task = next(t for t in manifest if t['key'] == os.environ['PI05_TASK'])
trials = list(range(int(os.environ['PI05_EPISODES'])))
resolved, _ = resolve_tasks([task], trials)
root = Path(os.environ['PI05_CHECK_OUTPUT'])
(root / 'task-id.txt').write_text(str(resolved[0]['task_id']))
(root / 'suite.txt').write_text(task['suite'])
(root / 'trials.txt').write_text(json.dumps(trials, separators=(',', ':')))
(root / 'protocol.json').write_text(json.dumps({
    'protocol': 'pi05_base weights + public LIBERO interface calibration',
    'task': resolved[0], 'trial_ids': trials, 'planned_episodes': len(trials),
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'model': str(Path(os.environ['FYP_STORAGE']) / 'models/pi05_base_openpi'),
    'action_horizon': 50, 'executed_action_chunk': 5, 'denoising_steps': 10,
    'discrete_state_input': True, 'task_time_limit': 350,
    'weight_training': False,
}, indent=2))
PY
task_id=$(cat "$PI05_CHECK_OUTPUT/task-id.txt")
suite=$(cat "$PI05_CHECK_OUTPUT/suite.txt")
trials=$(cat "$PI05_CHECK_OUTPUT/trials.txt")
python -m pip freeze > "$PI05_CHECK_OUTPUT/packages.txt"
bash evaluations/run_eval.sh libero libero_pi05_base_calibrated_eval \
  '~cluster.component_placement' '+cluster.component_placement={env:0,rollout:0}' \
  "+env.eval.task_id_filter=[$task_id]" \
  "env.eval.task_suite_name=$suite" \
  "env.eval.eval_trial_ids=$trials" \
  "env.eval.rollout_epoch=$PI05_EPISODES" \
  "runner.logger.log_path=$PI05_CHECK_OUTPUT/official_logs" \
  "runner.logger.experiment_name=pi05-base-calibrated-$PI05_TASK" \
  'env.eval.video_cfg.save_video=true' 2>&1 | tee "$PI05_CHECK_OUTPUT/console.log"
