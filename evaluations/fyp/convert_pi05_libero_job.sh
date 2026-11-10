#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
source "$FYP_RUNTIME/envs/rlinf-pi05-check/bin/activate"
export TMPDIR=/tmp JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
cd "$HOME/RLinf"
test -n "${SLURM_JOB_ID:-}"
test -z "$(git status --porcelain)"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['FYP_STORAGE']) / 'models/pi05_libero_jax'
assert (root / 'download-complete.json').is_file()
items = json.loads((root / 'download-manifest.json').read_text())
for item in items:
    path = root / item['name'].removeprefix('checkpoints/pi05_libero/')
    if not path.is_file() or path.stat().st_size != int(item['size']):
        raise ValueError(f'Incomplete checkpoint file: {path}')
print(f'Verified {len(items)} downloaded objects')
PY
output="$FYP_STORAGE/models/pi05_libero_openpi"
test ! -e "$output" || { echo "Conversion output already exists; inspect before retrying"; exit 1; }
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
  --input-model "$FYP_STORAGE/models/pi05_libero_jax" \
  --input-norm-stats "$FYP_STORAGE/models/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json" \
  --output-model "$output" \
  --output-norm-stats "$output/assets/physical-intelligence/libero/norm_stats.json" \
  --action-dim 32 --action-horizon 10 --max-token-len 200
python - <<'PY'
import json, os
from pathlib import Path
from safetensors import safe_open
root = Path(os.environ['FYP_STORAGE']) / 'models/pi05_libero_openpi'
with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as weights:
    keys = list(weights.keys())
    if not keys:
        raise ValueError('Converted checkpoint has no tensors')
report = {'stage': 'conversion_complete', 'source': 'gs://openpi-assets/checkpoints/pi05_libero', 'tensor_count': len(keys), 'normalization': 'original LIBERO statistics, copied unchanged', 'action_horizon': 10, 'discrete_state_input': False, 'policy_loaded': False, 'libero_episodes': 0, 'slurm_job_id': os.environ['SLURM_JOB_ID']}
(root / 'fyp-conversion.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
PY
