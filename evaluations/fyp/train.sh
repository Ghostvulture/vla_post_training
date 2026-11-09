#!/usr/bin/env bash
# Run inside a two-GPU Slurm allocation after sourcing the personal environment.
set -euo pipefail
: "${FYP_STORAGE:?Source the personal FYP environment first}"
: "${FYP_RUNTIME:?Source the personal FYP environment first}"
: "${SLURM_JOB_ID:?Run training within a Slurm allocation}"
output=${1:?Usage: train.sh OUTPUT_DIRECTORY [Hydra overrides...]}
shift
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
test -z "$(git status --porcelain)" || { echo 'Server checkout must be clean'; exit 1; }
test ! -e "$output" || { echo 'Use a fresh output directory'; exit 1; }
mkdir -p "$output"
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard ROBOT_PLATFORM=LIBERO
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
git rev-parse HEAD > "$output/git-commit.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$output/slurm-job.txt"
python -m pip freeze > "$output/packages.txt"
args=(
  python examples/embodiment/train_embodied_agent.py
  --config-name fyp_sg6_grpo_evo1
  "runner.logger.log_path=$output"
  "actor.model.model_path=$FYP_STORAGE/models/Evo1_LIBERO"
  "rollout.model.model_path=$FYP_STORAGE/models/Evo1_LIBERO"
  "actor.model.evo1.repo_path=$FYP_RUNTIME/envs/rlinf-evo1/Evo-1"
  "+actor.model.evo1.vlm_name=$FYP_STORAGE/models/InternVL3-1B"
)
printf '%q ' "${args[@]}" "$@" > "$output/command.sh"
printf '\n' >> "$output/command.sh"
"${args[@]}" "$@" 2>&1 | tee "$output/console.log"
