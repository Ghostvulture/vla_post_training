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
export PI05_TASK="${1:-SG6}" PI05_EPISODES="${2:-2}" PI05_CONDITION="${3:-base}"
case "$PI05_CONDITION" in
  base) config=libero_pi05_base_calibrated_eval; run_prefix=libero ;;
  control) config=libero_pi05_adapted_control_eval; run_prefix=control ;;
  *) echo "Expected base or control condition"; exit 2 ;;
esac
case "$PI05_TASK" in SG6|S1|O2|G1) ;; *) echo "Unsupported screening task"; exit 2 ;; esac
case "$PI05_EPISODES" in 1|2|3|4|5) ;; *) echo "Expected 1-5 development episodes"; exit 2 ;; esac
export PI05_CHECK_OUTPUT="$FYP_STORAGE/results/pi05-base-check/$run_prefix-$SLURM_JOB_ID-$PI05_TASK"
mkdir "$PI05_CHECK_OUTPUT"
python - <<'PY'
import json, os, subprocess
from pathlib import Path
from evaluations.fyp.evaluate import resolve_tasks
# Respect scheduler-visible devices; fail rather than sharing a busy GPU.
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
gpus = [tuple(int(x.strip()) for x in row.split(',')) for row in rows.splitlines()]
allowed = {g[0] for g in gpus}
for variable in ('SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES'):
    raw = os.environ.get(variable, '')
    if raw:
        if not all(x.strip().isdigit() for x in raw.split(',')):
            raise ValueError(f'Cannot safely resolve {variable}={raw}')
        allowed &= {int(x.strip()) for x in raw.split(',')}
candidates = [i for i, used, free in gpus if i in allowed and used < 1024 and free >= 12000]
if not candidates:
    raise RuntimeError('No scheduler-visible idle GPU with at least 12 GiB free')
gpu = candidates[0]
manifest = json.loads(Path('evaluations/fyp/tasks.json').read_text())
task = next(t for t in manifest if t['key'] == os.environ['PI05_TASK'])
trials = list(range(int(os.environ['PI05_EPISODES'])))
resolved, _ = resolve_tasks([task], trials)
root = Path(os.environ['PI05_CHECK_OUTPUT'])
(root / 'gpu-rank.txt').write_text(str(gpu))
(root / 'gpu-allocation.json').write_text(json.dumps({'selected_physical_index': gpu, 'memory_snapshot': gpus, 'SLURM_JOB_GPUS': os.environ.get('SLURM_JOB_GPUS'), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}, indent=2))
(root / 'task-id.txt').write_text(str(resolved[0]['task_id']))
(root / 'suite.txt').write_text(task['suite'])
(root / 'trials.txt').write_text(json.dumps(trials, separators=(',', ':')))
control = os.environ['PI05_CONDITION'] == 'control'
(root / 'protocol.json').write_text(json.dumps({
    'protocol': 'official pi05_libero adapted positive control' if control else 'pi05_base weights + public LIBERO interface calibration',
    'task': resolved[0], 'trial_ids': trials, 'planned_episodes': len(trials),
    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'model': str(Path(os.environ['FYP_STORAGE']) / ('models/pi05_libero_openpi' if control else 'models/pi05_base_openpi')),
    'action_horizon': 10 if control else 50, 'executed_action_chunk': 5, 'denoising_steps': 10,
    'discrete_state_input': not control, 'task_time_limit': 350,
    'weight_training': False,
}, indent=2))
PY
gpu=$(cat "$PI05_CHECK_OUTPUT/gpu-rank.txt")
task_id=$(cat "$PI05_CHECK_OUTPUT/task-id.txt")
suite=$(cat "$PI05_CHECK_OUTPUT/suite.txt")
trials=$(cat "$PI05_CHECK_OUTPUT/trials.txt")
python -m pip freeze > "$PI05_CHECK_OUTPUT/packages.txt"
bash evaluations/run_eval.sh libero "$config" \
  '~cluster.component_placement' "+cluster.component_placement={env:$gpu,rollout:$gpu}" \
  "+env.eval.task_id_filter=[$task_id]" \
  "env.eval.task_suite_name=$suite" \
  "env.eval.eval_trial_ids=$trials" \
  "env.eval.rollout_epoch=$PI05_EPISODES" \
  "runner.logger.log_path=$PI05_CHECK_OUTPUT/official_logs" \
  "runner.logger.experiment_name=pi05-$PI05_CONDITION-$PI05_TASK" \
  'env.eval.video_cfg.save_video=true' 2>&1 | tee "$PI05_CHECK_OUTPUT/console.log"
