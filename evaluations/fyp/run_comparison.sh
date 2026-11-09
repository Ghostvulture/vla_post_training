#!/usr/bin/env bash
# Blocking Slurm driver; wake-run waits on this process, without polling.
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
test -z "$(git status --porcelain)"
test -z "$(squeue -h -u "$USER")"
root="$FYP_STORAGE/results/sg6-comparison-v1"
test ! -e "$root" || { echo 'Comparison output exists; inspect before resuming'; exit 1; }
mkdir -p "$root"
git rev-parse HEAD > "$root/git-commit.txt"
for stage in layout-pilot plain diverse eval; do
  case "$stage" in
    layout-pilot) cards=2; limit=00:30:00 ;;
    plain|diverse) cards=2; limit=03:00:00 ;;
    eval) cards=1; limit=03:00:00 ;;
  esac
  echo "Starting $stage ($cards GPUs, time limit $limit)"
  sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
    --gres="gpu:4090:$cards" --cpus-per-task=16 --time="$limit" \
    --job-name="fyp-$stage" --output="$root/$stage-%j.log" \
    evaluations/fyp/comparison_job.sh "$stage"
  echo "Completed $stage"
done
