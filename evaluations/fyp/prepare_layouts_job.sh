#!/usr/bin/env bash
#SBATCH --partition=GPU
#SBATCH --nodelist=4090node1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=fyp-layouts
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
export TMPDIR=/tmp
cd "$HOME/RLinf"
test -z "$(git status --porcelain)"
python evaluations/fyp/prepare_layouts.py --output-dir "$FYP_STORAGE/variants/sg6-layouts-v1"
