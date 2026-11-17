#!/bin/bash
set -eo pipefail
test "${SLURMD_NODENAME:?}" = "${FYP_COMPUTE_NODE:?}"
test "$SLURMD_NODENAME" != 4090node1
gpu="${CUDA_VISIBLE_DEVICES:?}"
[[ "$gpu" != *,* ]]
runtime="${FYP_RUNTIME:?}"
storage="${FYP_STORAGE:?}"
work="$runtime/robodojo-official"
dojo="$work/RoboDojo"
parent="$storage/results/robodojo-xvla-rl-20010"
archive="$storage/results/robodojo-xvla-toggle-$SLURM_JOB_ID"
mkdir "$archive"
exec > >(tee "$archive/console.log") 2>&1
model_name=FypXvlaRL20010
ref=RoboDojo-sim-arx_x5-ee-0
export CONDA_ENVS_PATH="$work/envs" EVAL_ENV_TYPE=sim OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate XVLA-np126
export PATH="$HOME/miniforge3/condabin:$PATH"
cd "$dojo"
export PYTHONPATH="$dojo:$dojo/XPolicyLab:${PYTHONPATH:-}"
test "$(git rev-parse HEAD)" = "$(cat "$parent/robodojo-commit.txt")"
test "$(git -C XPolicyLab rev-parse HEAD)" = "$(cat "$parent/xpolicy-commit.txt")"
cmp "XPolicyLab/policy/$model_name/deploy.py" XPolicyLab/policy/X_VLA/deploy.py
unset FYP_ROLLOUT_DIR FYP_RL_ADAPTER
export FYP_POLICY_SEED=1701
python - "$parent/shared" "$archive" <<'PY'
from pathlib import Path
import sys, shutil, json
from safetensors.numpy import load_file, save_file
import numpy as np
src, dst = map(Path, sys.argv[1:])
weights = load_file(str(src / "adapter_model.safetensors"))
groups = {
    "action": {k for k in weights if ".transformer.blocks." in k and k.endswith(".attn.qkv.lora_B.weight")},
    "language": {k for k in weights if ".vlm.language_model.model.encoder.layers." in k and k.endswith((".q_proj.lora_B.weight", ".v_proj.lora_B.weight"))},
}
all_b = {k for k in weights if k.endswith(".lora_B.weight")}
assert len(weights) == 96 and len(all_b) == 48
assert all(len(keys) == 24 for keys in groups.values())
assert all_b == groups["action"] | groups["language"]
assert all(np.isfinite(w).all() for w in weights.values())
assert all(np.any(weights[k]) for k in all_b)
for arm, off in [("action_only", groups["language"]), ("language_only", groups["action"]), ("both_off", all_b)]:
    out = dst / arm
    out.mkdir()
    shutil.copyfile(src / "adapter_config.json", out / "adapter_config.json")
    modified = {k: np.zeros_like(v) if k in off else v for k, v in weights.items()}
    file = str(out / "adapter_model.safetensors")
    save_file(modified, file, metadata={"format": "pt"})
    readback = load_file(file)
    assert set(readback) == set(weights)
    for k, w in weights.items():
        assert (not np.any(readback[k])) if k in off else np.array_equal(readback[k], w)
    print(arm, len(off), "LoRA B matrices zeroed", flush=True)
(dst / "toggle-groups.json").write_text(json.dumps({k: sorted(v) for k, v in groups.items()}, indent=2))
PY
for arm in both_off action_only language_only both_on; do
    export FYP_RL_ADAPTER="$archive/$arm"
    if [[ "$arm" == both_on ]]; then export FYP_RL_ADAPTER="$parent/shared"; fi
    for task in stack_blocks stack_bowls; do
        label="$arm-$task"
        export ROBODOJO_RUN_ID="fyp-$SLURM_JOB_ID-$label"
        bash scripts/robodojo.sh eval --policy-dir "XPolicyLab/policy/$model_name" --task "$task" --ckpt "$ref" --seed 2 --eval-num 10 --env-cfg arx_x5 --action-type ee --policy-env XVLA-np126 --eval-env RoboDojo --policy-gpu "$gpu" --env-gpu "$gpu"
        f="eval_result/RoboDojo/$task/$model_name/arx_x5/2_ckpt_name=${ref},action_type=ee/$ROBODOJO_RUN_ID/_result.json"
        python - "$f" <<'PY'
import json, sys
with open(sys.argv[1]) as src:
    result = json.load(src)
rows = list(result["details"].values())
assert len(rows) == result["eval_time"] == 10
assert {r["layout_id"] for r in rows} == set(range(10))
assert abs(sum(r["success"] for r in rows)/10 - result["success_rate"]) < 1e-9
print("Verified", sys.argv[1], result["success_rate"], flush=True)
PY
        cp "$f" "$archive/$label.json"
    done
done
