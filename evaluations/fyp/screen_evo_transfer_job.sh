#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
export TMPDIR=/tmp
cd "$HOME/RLinf"
test -n "${SLURM_JOB_ID:-}"
test -z "$(git status --porcelain)"
python - <<'CHECK'
import os, subprocess
visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not visible.isdigit():
    raise RuntimeError('Expected one Slurm-visible GPU index')
free = int(subprocess.check_output(['nvidia-smi', '--id='+visible, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
if free < 12000:
    raise RuntimeError('Allocated GPU has less than 12 GiB free; stop before evaluation')
CHECK
root="$FYP_STORAGE/results/evo-transfer-screen-$SLURM_JOB_ID"
bash "$FYP_RUNTIME/fyp-eval-job.sh" --tasks SG6 B2N B2P \
  --episodes 10 --trial-start 10 --num-envs 2 --budget-minutes 25 \
  --variant-root "$FYP_STORAGE/variants/sg6-expanded-20261109" \
  --output-dir "$root/supplement"
bash "$FYP_RUNTIME/fyp-eval-job.sh" --tasks B1T \
  --episodes 10 --trial-start 0 --num-envs 2 --budget-minutes 15 \
  --variant-root "$FYP_STORAGE/variants/sg6-transfer-v1" \
  --output-dir "$root/object-candidate"
bash "$FYP_RUNTIME/fyp-eval-job.sh" --tasks SG6 B2N B2P \
  --episodes 2 --trial-start 10 --num-envs 1 --video --budget-minutes 10 \
  --variant-root "$FYP_STORAGE/variants/sg6-expanded-20261109" \
  --output-dir "$root/video-diagnostics"
