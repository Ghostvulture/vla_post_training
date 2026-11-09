#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
source "$FYP_RUNTIME/envs/rlinf-pi05-check/bin/activate"
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=8
cd "$HOME/RLinf"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -z "$(git status --porcelain)"
python evaluations/fyp/check_pi05_inference.py \
  --model "$FYP_STORAGE/models/pi05_base_openpi" \
  --output "$FYP_STORAGE/results/pi05-base-check/inference-$SLURM_JOB_ID.json"
