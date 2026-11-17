#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
runtime="${FYP_RUNTIME:?}"
storage="${FYP_STORAGE:?}"
work="$runtime/robodojo-official"
checkout="$work/RoboDojo"
checkpoint="$storage/robodojo/ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000"
archive="$storage/results/robodojo-xvla-screen-$SLURM_JOB_ID"
mkdir -p "$work" "$archive"
exec > >(tee "$archive/console.log") 2>&1
export CONDA_ENVS_PATH="$work/envs" CONDA_PKGS_DIRS="$work/conda-pkgs"
export PIP_CACHE_DIR="$runtime/cache/pip" HF_HOME="$runtime/cache/robodojo-hf"
export TMPDIR="$work/tmp" TORCH_EXTENSIONS_DIR="$work/torch-extensions"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 TERM=xterm-256color
export PIP_DEFAULT_TIMEOUT=100 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
export OMNI_KIT_ACCEPT_EULA=YES ISAACLAB_RL_FRAMEWORK=none
mkdir -p "$TMPDIR" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
test -s "$checkpoint/model.safetensors"
test -d "$storage/robodojo/Assets/Robots"
gpu="${CUDA_VISIBLE_DEVICES:?missing Slurm GPU assignment}"
[[ "$gpu" != *,* ]]
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ ! -d "$checkout/.git" ]]; then
    git clone --depth 1 https://github.com/RoboDojo-Benchmark/RoboDojo.git "$checkout"
fi
cd "$checkout"
if [[ "$(git rev-parse HEAD)" != f57e1633ffed59761bc8e568cfe7530c4b33381c ]]; then
    git fetch --depth 1 origin f57e1633ffed59761bc8e568cfe7530c4b33381c
    git checkout --detach f57e1633ffed59761bc8e568cfe7530c4b33381c
fi
git submodule update --init --depth 1 third_party/curobo third_party/IsaacLab
if [[ ! -e XPolicyLab/.git ]]; then git submodule update --init --depth 1 XPolicyLab; fi
if [[ "$(git -C XPolicyLab rev-parse HEAD)" != 6baccd57ff958570d0db1588370289ee36c8bb70 ]]; then
    git -C XPolicyLab fetch --depth 1 origin 6baccd57ff958570d0db1588370289ee36c8bb70
    git -C XPolicyLab checkout --detach 6baccd57ff958570d0db1588370289ee36c8bb70
fi
git rev-parse HEAD > "$archive/robodojo-commit.txt"
git submodule status > "$archive/component-revisions.txt"
if [[ ! -e Assets ]]; then ln -s "$storage/robodojo/Assets" Assets; fi
test "$(realpath Assets)" = "$storage/robodojo/Assets"
python_boot="$runtime/envs/rlinf-evo1/bin/python"
"$python_boot" utils/update_embodiment_config_path.py
processor="$work/X-VLA-Pt-processor"
"$python_boot" - "$processor" <<'PY'
from pathlib import Path
import json
import sys
import requests
root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
revision = "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b"
for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "vocab.json"]:
    if (root / name).is_file():
        with (root / name).open() as stream:
            json.load(stream)
        continue
    r = requests.get(f"https://huggingface.co/2toINF/X-VLA-Pt/resolve/{revision}/{name}", timeout=(15, 90))
    r.raise_for_status()
    r.json()
    (root / name).write_bytes(r.content)
PY
# Keep downloaded model and asset snapshots separate from this adapter view.
bundle="$work/ckpt-100000"
mkdir -p "$bundle"
for name in model.safetensors config.json state.json; do
    [[ -e "$bundle/$name" ]] || ln -s "$checkpoint/$name" "$bundle/$name"
done
[[ -e "$bundle/processor" ]] || ln -s "$processor" "$bundle/processor"
adapter_dir="$checkout/XPolicyLab/policy/X_VLA/checkpoints"
mkdir -p "$adapter_dir"
model_id=RoboDojo-sim-arx_x5-ee-0
[[ -e "$adapter_dir/$model_id" ]] || ln -s "$bundle" "$adapter_dir/$model_id"
test "$(realpath "$adapter_dir/$model_id")" = "$bundle"
if [[ ! -x "$CONDA_ENVS_PATH/RoboDojo/bin/python" ]]; then
    conda create -y -n RoboDojo --override-channels -c conda-forge python=3.11 pip cmake ninja cuda-nvcc=12.8 cuda-cudart-dev=12.8
fi
conda activate RoboDojo
export CUDA_HOME="$CONDA_PREFIX"
if [[ ! -f "$work/simulator-ready" ]]; then
    python -m pip install -r scripts/requirements.txt
    python -m pip install opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0
    bash scripts/install.sh --from isaacsim
    python -c 'import torch; assert torch.cuda.is_available(); import isaacsim; from curobo.motion_planner import MotionPlanner'
    touch "$work/simulator-ready"
fi
python -m pip freeze > "$archive/simulator-packages.txt"
policy_env=XVLA-np126
if [[ ! -x "$CONDA_ENVS_PATH/$policy_env/bin/python" ]]; then
    conda create -y -n "$policy_env" --override-channels -c conda-forge python=3.10 pip 'av=14.4.0=py310h94b1f55_0' 'numpy=1.26.3'
fi
conda activate "$policy_env"
if [[ ! -f "$CONDA_PREFIX/.policy-ready" ]]; then
    echo 'numpy==1.26.3' > "$archive/policy-constraints.txt"
    export PIP_CONSTRAINT="$archive/policy-constraints.txt"
    python -c 'import av; assert av.__version__ == "14.4.0"; print("FFmpeg libraries:", av.library_versions)'
    python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install 'transformers==4.51.3' -r XPolicyLab/policy/X_VLA/xvla/requirements.txt
    python -m pip install -e XPolicyLab 'numpy==1.26.3' 'opencv-python-headless==4.11.0.86'
    python - <<'PY'
from importlib.metadata import distributions
import av
import numpy
import scipy.special
import torch
from XPolicyLab.policy.X_VLA.model import XVLAProcessor
import peft

versions = [dist.version for dist in distributions() if dist.metadata["Name"].lower() == "numpy"]
assert versions == ["1.26.3"], versions
assert isinstance(scipy.special.loggamma, numpy.ufunc)
assert torch.cuda.is_available()
print("Policy preflight passed", versions, av.__version__, peft.__version__)
PY
    touch "$CONDA_PREFIX/.policy-ready"
    unset PIP_CONSTRAINT
fi
python -m pip freeze > "$archive/policy-packages.txt"
export EVAL_ENV_TYPE=sim
args=(--policy-dir XPolicyLab/policy/X_VLA --ckpt "$model_id" --policy-env "$policy_env" --eval-env RoboDojo --action-type ee --env-cfg arx_x5 --policy-gpu "$gpu" --env-gpu "$gpu")
touch "$archive/evaluation-start"
save_results() {
    find "$checkout" -type f -name '_result.json' -newer "$archive/evaluation-start" -exec cp --parents -t "$archive" {} +
}
trap save_results EXIT
bash scripts/robodojo.sh smoke "${args[@]}" --only stack_bowls --seed 0 --eval-num 1 --fail-fast
for task in stack_bowls stack_blocks push_T; do
    bash scripts/robodojo.sh eval "${args[@]}" --task "$task" --seed 1 --eval-num 10
done
