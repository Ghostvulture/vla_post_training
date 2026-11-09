#!/usr/bin/env bash
# Add preprocessing dependencies only to the separate Pi0.5 check environment.
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
test -z "$(git status --porcelain)"
test -z "$(squeue -h -u "$USER")"
export TMPDIR=/tmp UV_CACHE_DIR="$FYP_RUNTIME/cache/uv" JAX_PLATFORMS=cpu
pi_python="$FYP_RUNTIME/envs/rlinf-pi05-check/bin/python"
uv pip install --python "$pi_python" --no-deps 'rlinf-openpi==0.1.1'
uv pip install --python "$pi_python" \
  'numpy==1.26.4' 'jax==0.5.3' 'jaxlib==0.5.3' 'orbax-checkpoint==0.11.13' \
  'flax==0.10.2' 'chex==0.1.90' 'optax==0.2.4' 'equinox==0.11.12' \
  'augmax==0.3.4' 'beartype==0.19.0' 'jaxtyping==0.2.36' \
  'ml-collections==1.0.0' 'numpydantic==1.6.6' 'tyro==0.9.5' \
  'openpi-client' 'polars' 'treescope' 'gcsfs' 'tqdm-loggable'
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$pi_python" - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

root = Path(os.environ['FYP_STORAGE']) / 'models/pi05_libero_calibration'
root.mkdir(parents=True, exist_ok=True)
name = 'checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json'
generation = '1757354341826612'
url = f'https://storage.googleapis.com/openpi-assets/{name}?generation={generation}'
with urllib.request.urlopen(url, timeout=60) as response:
    raw = response.read()
if len(raw) != 1914:
    raise ValueError(f'Expected 1914 bytes of official stats, got {len(raw)}')
stats = json.loads(raw)
for key, dim in [('state', 8), ('actions', 7)]:
    for field in ('mean', 'std', 'q01', 'q99'):
        if len(stats['norm_stats'][key][field]) != dim:
            raise ValueError(f'Unexpected normalization dimension: {key}/{field}')
path = root / 'norm_stats.json'
if path.exists() and path.read_bytes() != raw:
    raise ValueError('Refusing to replace different existing calibration statistics')
path.write_bytes(raw)
(root / 'provenance.json').write_text(json.dumps({
    'source': 'gs://openpi-assets/' + name,
    'generation': generation,
    'protocol': 'pi05_base weights with public LIBERO interface calibration; no weight training',
    'downloaded_finetuned_weights': False,
    'downloaded_demonstrations': False,
}, indent=2))
print('Saved public LIBERO calibration statistics', flush=True)
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from openpi.models.tokenizer import PaligemmaTokenizer
get_openpi_config('pi05_libero')
PaligemmaTokenizer(200)
print('Preprocessing imports and tokenizer initialization passed', flush=True)
PY
"$pi_python" -m pip freeze > "$FYP_STORAGE/results/pi05-base-check/calibration-packages.txt"
