#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
export TMPDIR=/tmp
cd "$HOME/RLinf"
stage=${1:?Specify layout-pilot, plain, diverse, or eval}
root="$FYP_STORAGE/results/sg6-comparison-v1"
if [[ "$stage" == eval ]]; then
  for model in base plain diverse; do
    checkpoint="$FYP_STORAGE/models/sg6-comparison-v1-$model"
    [[ "$model" != base ]] || checkpoint="$FYP_STORAGE/models/Evo1_LIBERO"
    bash "$FYP_RUNTIME/fyp-eval-job.sh" --model-path "$checkpoint" \
      --tasks SG6 A1 A2 B2N B2P G1 SG7 \
      --variant-root "$FYP_STORAGE/variants/sg6-expanded-20261109" \
      --episodes 20 --trial-start 30 --num-envs 2 --budget-minutes 55 \
      --output-dir "$root/eval-$model"
  done
  exit 0
fi
case "$stage" in
  layout-pilot) steps=1; save=1 ;;
  plain|diverse) steps=20; save=10 ;;
  *) echo "Unknown stage: $stage"; exit 2 ;;
esac
extra=()
if [[ "$stage" != plain ]]; then
  extra+=("+env.train.layout_bank_path=$FYP_STORAGE/variants/sg6-layouts-v2/bank.pt")
fi
bash evaluations/fyp/train.sh "$root/$stage" \
  "runner.logger.experiment_name=sg6-grpo" \
  "runner.max_epochs=$steps" "runner.max_steps=$steps" "runner.save_interval=$save" \
  +env.train.log_interaction_steps=true +actor.optim.error_on_nonfinite_grad=true \
  "${extra[@]}"
audit_args=()
if [[ "$stage" == diverse ]]; then
  audit_args+=(--compare "$root/plain/training-audit.json")
fi
python evaluations/fyp/audit_training.py --run "$root/$stage" --iterations "$steps" "${audit_args[@]}"
python evaluations/fyp/export_evo1.py --base "$FYP_STORAGE/models/Evo1_LIBERO" \
  --weights "$root/$stage/sg6-grpo/checkpoints/global_step_$steps/actor/model_state_dict/full_weights.pt" \
  --output "$FYP_STORAGE/models/sg6-comparison-v1-$stage"
