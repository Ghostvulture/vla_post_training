#!/usr/bin/env bash
# Server-only preparation; the Evo-1 environment supplies read-only dependencies.
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
test -z "$(git status --porcelain)"
test -z "$(squeue -h -u "$USER")"
export TMPDIR=/tmp
export UV_CACHE_DIR="$FYP_RUNTIME/cache/uv"
export JAX_PLATFORMS=cpu
base_site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pi_env="$FYP_RUNTIME/envs/rlinf-pi05-check"
if [[ ! -x "$pi_env/bin/python" ]]; then
  python -m venv "$pi_env"
  pi_site=$("$pi_env/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  printf '%s\n' "$base_site" > "$pi_site/fyp_evo_dependencies.pth"
fi
uv pip install --python "$pi_env/bin/python" \
  'numpy==1.26.4' 'jax==0.5.3' 'jaxlib==0.5.3' 'orbax-checkpoint==0.11.13' \
  'sentencepiece==0.2.1'
"$pi_env/bin/python" -c 'import jax, orbax.checkpoint, torch, numpy; print("versions", jax.__version__, torch.__version__, numpy.__version__)'
mkdir -p "$FYP_STORAGE/results/pi05-base-check"
"$pi_env/bin/python" -m pip freeze > "$FYP_STORAGE/results/pi05-base-check/conversion-packages.txt"
git rev-parse HEAD > "$FYP_STORAGE/results/pi05-base-check/conversion-commit.txt"
sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
  --cpus-per-task=8 --time=00:30:00 --job-name=fyp-pi05-convert \
  --output="$FYP_STORAGE/results/pi05-base-check/convert-%j.log" \
  evaluations/fyp/convert_pi05_base_job.sh
