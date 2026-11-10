#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/RLinf"
for task in S1 O2 G1; do
  bash evaluations/fyp/check_pi05_libero_job.sh "$task" 5 "${1:-base}"
done
