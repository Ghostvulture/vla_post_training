# FYP: Evaluate the Evo-1 Baseline

[中文](README.zh-CN.md)

This guide covers connecting from your computer, activating the installed Evo-1 environment, submitting LIBERO evaluation through Slurm, and inspecting results. Edit code and run CPU checks locally; load models, simulate, and evaluate on the server. The `Evo1_LIBERO` checkpoint was trained for LIBERO. This project did not run additional SFT or RL before evaluation.

## Prepare the Environment After Login

Connect from a terminal on your computer. Replace the example with your SSH username and address, or use your configured SSH alias:

```bash
ssh your-user@your-server
```

Run the following on the server. The personal configuration outside the repository records your server's storage, runtime, and node, and activates the installed virtual environment. Run this block in every new server terminal:

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
mkdir -p "$FYP_STORAGE/results"
FYP_SLURM=(
  --partition=GPU
  --nodelist="$FYP_NODE"
  --gres=gpu:4090:1
  --cpus-per-task=8
  --time=01:40:00
  --output="$FYP_STORAGE/results/slurm-eval-%j.log"
)
command -v python
squeue -u "$USER"
```

`FYP_STORAGE` holds models and results; `FYP_RUNTIME` holds the environment and caches. Inspect their configured paths with `cat "$HOME/.config/fyp/evo1-env.sh"`. `FYP_SLURM` requests one 4090, eight CPUs, and a 100-minute limit. `squeue` shows your jobs. If submission reports `AssocMaxSubmitJobLimit`, wait for an existing job to finish before submitting another. Adapt the array to your cluster's allocation rules when they change.

The server-local launcher is `$FYP_RUNTIME/fyp-eval-job.sh`. It loads the same environment inside the job, reads the existing models offline, and calls this project's [evaluation entry point](evaluate.py). The job stores temporary files under the node's `/tmp` to keep Ray socket paths short; models, caches, and results remain in personal storage. Each `--budget-minutes` below limits one evaluation and must leave time for startup and cleanup within Slurm's `--time`.

## Submit Evaluation

### Smoke Test and Video

First check model loading, simulation, and video on two initial states from one task. Use a new output directory for each run:

```bash
FYP_RUN_ID="$(date +%Y%m%d-%H%M%S)"
sbatch "${FYP_SLURM[@]}" "$FYP_RUNTIME/fyp-eval-job.sh" \
  --tasks S1 --episodes 2 --trial-start 0 --video \
  --output-dir "$FYP_STORAGE/results/smoke-$FYP_RUN_ID" \
  --budget-minutes 60
```

`sbatch` returns a job ID. `--video` supports at most two episodes per task. After completion, find videos under `S1/official_logs/video/eval/` in the output directory. Zero success can be a valid result; investigate missing trajectories, crashes, or absent actions. A smoke test that repeats a formal task and initial state does not add an independent sample.

### Six-Task Screening

The launcher matches full task names against the installed LIBERO version. Labels are defined in [tasks.json](tasks.json).

| Label | Task | Suite |
|---|---|---|
| S1 | Place the black bowl from the table center on the plate | LIBERO-Spatial |
| S2 | Place the black bowl beside the cookie box on the plate | LIBERO-Spatial |
| O1 | Place alphabet soup in the basket | LIBERO-Object |
| O2 | Place cream cheese in the basket | LIBERO-Object |
| G1 | Open the middle drawer | LIBERO-Goal |
| G2 | Push the plate to the stove front | LIBERO-Goal |

Omit `--tasks` to run all six sequentially. This command uses ten initial states numbered 0–9 per task; each task calls the official RLinf evaluation entry point separately:

```bash
FYP_RUN_ID="$(date +%Y%m%d-%H%M%S)"
sbatch "${FYP_SLURM[@]}" "$FYP_RUNTIME/fyp-eval-job.sh" \
  --episodes 10 --trial-start 0 \
  --output-dir "$FYP_STORAGE/results/screen-0-9-$FYP_RUN_ID" \
  --budget-minutes 90
```

To restrict a run, add `--tasks S1 S2` and use a fresh output directory. Evaluation retains the [Evo-1 configuration](../libero/libero_spatial_evo1_eval.yaml): 448×448 images, 14-step action chunks, 350 simulation steps per episode, and the `libero_robot` normalization key. Record changes to the model, code version, or evaluation settings and reassess comparability.

The additional screening labels are `SG1 SG3 SG6 SG7 SS4 SS7`; their task goals are in [tasks.json](tasks.json). To screen them, use `--tasks SG1 SG3 SG6 SG7 SS4 SS7 --episodes 5 --trial-start 0`. To add five different states, use a new output directory and `--trial-start 5`. Omitting `--tasks` still runs only the six original baseline tasks in the table.

If time permits, wait for the first batch to finish and submit states 10–19:

```bash
FYP_RUN_ID="$(date +%Y%m%d-%H%M%S)"
sbatch "${FYP_SLURM[@]}" "$FYP_RUNTIME/fyp-eval-job.sh" \
  --episodes 10 --trial-start 10 \
  --output-dir "$FYP_STORAGE/results/screen-10-19-$FYP_RUN_ID" \
  --budget-minutes 90
```

Changing the output directory alone does not select new initial states; `--trial-start` controls their numbers. Merge batches only when the checkpoint, code version, and evaluation protocol match and their state numbers do not overlap. The launcher refuses to overwrite an existing output directory.

## Inspect Jobs, Logs, and Results

Replace the example job ID with the number returned by `sbatch`. `squeue` shows the current status, `tail` follows the Slurm log, and `sacct` shows the final status. `Ctrl+C` exits the log view without stopping the job.

```bash
FYP_JOB_ID=12345
squeue -j "$FYP_JOB_ID"
tail -f "$FYP_STORAGE/results/slurm-eval-$FYP_JOB_ID.log"
sacct -j "$FYP_JOB_ID" --format=JobID,State,ExitCode,Elapsed
```

Each output directory contains `run.json`, `summary.csv`, and each task's `episodes.csv`, `summary.json`, `console.log`, and `gpu-memory.csv`. Check `status`, completed counts, and initial-state numbers before interpreting a rate. Environment errors, timeouts, and unfinished trajectories must not be counted as ordinary task failures.

Inspect the two batch directories used in this project and merge them after both finish:

```bash
cat "$FYP_STORAGE/results/evo1-baseline-batch1/summary.csv"
cat "$FYP_STORAGE/results/evo1-baseline-batch2/summary.csv"
python evaluations/fyp/evaluate.py --report \
  "$FYP_STORAGE/results/evo1-baseline-batch1" \
  "$FYP_STORAGE/results/evo1-baseline-batch2"
```

For other batches, replace the paths with their actual output directories. The report reads saved files without loading a model or starting simulation; it checks provenance and protocol and rejects repeated initial states. Per-episode `success` means the goal was met at least once within 350 steps. `success_at_end` means it remained satisfied at the final step. Ten to twenty fixed initial states per task support screening, not claims about the true success rate or small differences.

To stop your own job, first confirm its ID with `squeue -u "$USER"`, then run `scancel "$FYP_JOB_ID"`. Inspect the saved `run.json` and episode records afterwards and mark unfinished batches as incomplete.

## Synchronize Code and Check Locally

The local project is the independent `RLinf` repository. After editing, inspect changes and stage only the files you intend to commit, for example:

```bash
cd /home/xyz/code/works/VLA_test/RLinf
git status --short
git add evaluations/fyp/README.zh-CN.md evaluations/fyp/README.md
git commit -s -m "docs(fyp): update evaluation guide"
git push origin main
```

Keep server code and models unchanged while a job is running. Once all jobs finish, inspect the server checkout and fast-forward it:

```bash
cd "$HOME/RLinf"
squeue -u "$USER"
git status --short
git pull --ff-only
```

Git transfers code. Models, results, personal paths, and SSH keys stay outside the repository. Run local CPU checks with:

```bash
python3 -m pytest -q \
  tests/unit_tests/test_fyp_evaluation.py \
  tests/unit_tests/test_fyp_report.py \
  tests/unit_tests/test_libero_env.py
```

These checks do not load the model or run simulation. Once the environment is installed, login does not require rerunning `requirements/install.sh` or downloading models again. Later RL work requires selecting tasks and a training budget from baseline results; the evaluation script does not start training.

## Move or Restore Server Configuration

The current server already has its personal configuration and launcher. On another server, set `FYP_STORAGE`, `FYP_RUNTIME`, `FYP_NODE`, cache paths, and the installed `rlinf-evo1` environment outside the repository. Replace the example paths and node in this template with the new server's values, then save it as `~/.config/fyp/evo1-env.sh`:

```bash
export FYP_STORAGE='/path/to/fyp'
export FYP_RUNTIME='/path/to/fyp-runtime'
export FYP_NODE='your-gpu-node'
export HF_HOME="$FYP_RUNTIME/cache/huggingface"
export TORCH_HOME="$FYP_RUNTIME/cache/torch"
export TORCH_EXTENSIONS_DIR="$FYP_RUNTIME/cache/torch-extensions"
export UV_CACHE_DIR="$FYP_RUNTIME/cache/uv"
export PIP_CACHE_DIR="$FYP_RUNTIME/cache/pip"
export TMPDIR="$FYP_RUNTIME/tmp"
source "$FYP_RUNTIME/envs/rlinf-evo1/bin/activate"
```

For a first installation on a new server, set the template's path variables, confirm that an administrator has installed system dependencies, and run the official installer. An existing environment does not need to be reinstalled:

```bash
cd "$HOME/RLinf"
mkdir -p "$FYP_RUNTIME/tmp" "$FYP_STORAGE/models" "$FYP_STORAGE/results"
export PATH="$HOME/.local/bin:$PATH"
TMPDIR="$FYP_RUNTIME/tmp" UV_PYTHON_INSTALL_DIR="$FYP_RUNTIME/uv-python" \
  bash requirements/install.sh embodied --model evo1 --env libero \
  --no-root --use-mirror --venv "$FYP_RUNTIME/envs/rlinf-evo1"
source "$HOME/.config/fyp/evo1-env.sh"
```

`--no-root` does not install missing system dependencies. To download missing weights, run the following in your server's personal storage; do not replace model files while a job is running:

```bash
hf download MINT-SJTU/Evo1_LIBERO --local-dir "$FYP_STORAGE/models/Evo1_LIBERO"
hf download OpenGVLab/InternVL3-1B --local-dir "$FYP_STORAGE/models/InternVL3-1B"
```

Confirm that the Evo-1 fork is at `$FYP_RUNTIME/envs/rlinf-evo1/Evo-1`. If the server-local job script is missing, source your configuration and recreate it with:

```bash
cat > "$FYP_RUNTIME/fyp-eval-job.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.config/fyp/evo1-env.sh"
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$HOME/RLinf"
python evaluations/fyp/evaluate.py \
  --model-path "$FYP_STORAGE/models/Evo1_LIBERO" \
  --evo-repo "$FYP_RUNTIME/envs/rlinf-evo1/Evo-1" \
  --vlm-path "$FYP_STORAGE/models/InternVL3-1B" \
  --num-envs 1 --gpu 0 "$@"
SCRIPT
chmod 700 "$FYP_RUNTIME/fyp-eval-job.sh"
```

After moving, check model paths, LIBERO simulator assets, node, and GPU allocation before running the two-episode smoke test. Install environments and download models and assets on the server, following the storage nodes' `user_data` / `public_data` rules.

## Near-task transfer: screening before training

The pilot is capped at 48 training GPU-hours (at most two GPUs) and 8 evaluation GPU-hours (one GPU). Six SG6 scene variants change the target: B1B/B1A/B1T replace cream cheese with butter/alphabet soup/tomato sauce; B2P uses the existing plate; B2W/B2B replace the bowl with a wooden tray/basket. Replacement changes geometry. States come from seeded environment resets, not the original task's identical physical states. Generation records initial-success and contact checks; invalid variants cannot be scored. Visual variants and training layout extensions follow only if screening passes.

Run generation within an allocated Slurm GPU job; the output directory must not exist. It does not load a VLA. Then use the single-GPU sbatch evaluation launcher above with the following additional arguments:

```bash
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-transfer-v1"
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-transfer-v1" --validate-goals
```

```bash
--tasks B1B B1A B1T B2P B2W B2B \
--variant-root "$FYP_STORAGE/variants/sg6-transfer-v1" \
--episodes 10 --trial-start 0
```

### Screening outcome on 2026-11-09

No RL training was started. The only valid receptacle candidate, B2P (cream cheese onto the existing plate in the same scene), achieved success_once in 1/10 development trials and success_at_end in 0/10, below the predefined 2–8/10 gate. The wooden tray B2W had substantial contact penetration. Basket B2B displaced the wine bottle during initialization despite passing the final settled contact check. Validation now checks penetration and planar displacement during settling, plus a feasible success-predicate witness. Three object variants passed environment checks but were not scored once the receptacle side could no longer yield an eligible pair.

Scores are in `$FYP_STORAGE/results/sg6-transfer-container-v2`; the gate report is `$FYP_STORAGE/results/sg6-transfer-gate-v2.json`. `sg6-transfer-screen-v1` was cancelled and is not scored. `sg6-transfer-container-video-v2` replays initializations for diagnosis and adds no screening samples.

The environment now accepts `env.train.train_trial_ids`, disallowing evaluation mode or a fixed reset ID with that option. Layout randomization, training launch configurations, checkpoint recovery, and the three-policy comparison have not been executed or accepted. No unverified training commands are provided. Revise task pairing before proceeding; this result does not establish absence of RL transfer.

## Expanded Original-Model Screening

Screen two candidates in each of four categories plus SG6, with 10 development initializations per condition. A1/A2 change only the wall to dark-blue/dark-green, keep the light-gray floor, and reuse original task states. B1B/B1A replace the object with butter/alphabet soup; B2N/B2P replace the bowl with a white bowl/the existing plate. G1/SG7 open a drawer/turn on the stove. B conditions use generated states, not identical physical states across conditions. Preserve the official On predicate; it is not a strict containment test.

The original model is MINT-SJTU/Evo1_LIBERO with the OpenGVLab/InternVL3-1B backbone, already adapted to LIBERO. This project has performed no SFT or RL. Complete all valid candidates even at 0/10 or 10/10; a failed training-selection gate no longer stops the other categories. Report invalid scenes separately from policy failures.

Generate and validate the four B variants within a Slurm GPU job, then pass the second block to the evaluation launcher using the sbatch workflow above. The output directory must not exist; preserve earlier screening results.

```bash
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-expanded-20261109" \
  --keys B1B B1A B2N B2P
python evaluations/fyp/prepare_variants.py \
  --output-dir "$FYP_STORAGE/variants/sg6-expanded-20261109" --validate-goals
```

```bash
--tasks SG6 A1 A2 B1B B1A B2N B2P G1 SG7 \
--variant-root "$FYP_STORAGE/variants/sg6-expanded-20261109" \
--episodes 10 --trial-start 0 --num-envs 2
```

### Nine-Condition Screening Results (2026-11-09)

All 90 trajectories completed without environment errors; no RL ran. Both columns count successes out of 10: at any point and at the end.

| Condition | Success Once | Success at End |
|---|---:|---:|
| SG6 Original task | 7 | 5 |
| A1 Dark-blue wall | 7 | 3 |
| A2 Dark-green wall | 9 | 5 |
| B1B Butter | 0 | 0 |
| B1A Alphabet soup | 0 | 0 |
| B2N White bowl | 5 | 1 |
| B2P Existing plate | 2 | 2 |
| G1 Open drawer | 10 | 10 |
| SG7 Turn on stove | 10 | 10 |

Evaluation used commit `1eb77c9e`; results are at `$FYP_STORAGE/results/sg6-expanded-20261109`. Slurm jobs 19464/19465 completed, totaling about 0.450 single-GPU hours for scene checks and evaluation. B2R, the ramekin, failed the goal-witness check and was not scored. B2P reused validated assets from the earlier directory, retaining their original generation and validation records.

The white bowl and plate leave room for improvement, but neither object-replacement candidate succeeded; a moderately difficult object pair is still missing. Only 1/10 white-bowl trials retained success at the end, requiring video diagnosis. A and the source task share initializations; B uses generated states, and the plate also changes target position. Cross-condition differences cannot be attributed to a single factor. Ten trials are screening evidence; 0/10 and 10/10 do not establish true probabilities of zero or one. Repeated initializations across batches are not additional samples. This screening does not launch training.

## Single-Task RL Pilot

The user authorized the next step. Train on SG6; use white bowl B2N and plate B2P as the primary near-task tests, with A1/A2 and G1/SG7 as appearance and different-skill controls. This narrows the question to container transfer, not object transfer. The training cap remains 48 GPU hours including retries, with at most 8 GPU hours for pilot troubleshooting; evaluation has a separate 8 GPU-hour cap.

`fyp_sg6_grpo_evo1` retains the official GRPO settings and trains only the action head. Initialization IDs 0–19 are training, 20–29 validation, and 30–49 final test. Two GPUs separate the actor from rollout/env, with 8 environments, group size 8, and micro-batch size 1. The default runs two iterations and saves each checkpoint. B/C do not enter training or test-based checkpoint selection. Layout diversification and matched-interaction comparison follow only after the pilot passes validation.

Run the pilot below with a fresh output directory. The 45-minute limit bounds the two-GPU job; it is not a completion estimate. Inspect reward-group metrics, gradients, and checkpoint saving. A successful launch alone does not establish a valid update.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
sbatch --partition=GPU --nodelist="$FYP_NODE" --gres=gpu:4090:2 \
  --cpus-per-task=16 --time=00:45:00 --job-name=fyp-grpo-pilot \
  --output="$FYP_STORAGE/results/grpo-pilot-%j.log" \
  --wrap='bash -lc '"'"'source "$HOME/.config/fyp/evo1-env.sh"; cd "$HOME/RLinf"; bash evaluations/fyp/train.sh "$FYP_STORAGE/results/sg6-grpo-pilot-v1"'"'"''
```

Training checkpoints are under `sg6-grpo-pilot/checkpoints/global_step_N` inside the output directory. To test one resumed iteration, give `train.sh` a new output directory and append `runner.resume_dir=/full/path/global_step_2 runner.max_epochs=3 runner.max_steps=3`. Resume uses the training checkpoint, not the downloaded model directory.

Run the export below on the server. It rejects nonfinite weights, changes outside the action head, or an unchanged action head. A successful export still requires loading through official evaluation.

```bash
python evaluations/fyp/export_evo1.py \
  --base "$FYP_STORAGE/models/Evo1_LIBERO" \
  --weights "$FYP_STORAGE/results/sg6-grpo-pilot-v1/sg6-grpo-pilot/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt" \
  --output "$FYP_STORAGE/models/sg6-grpo-pilot-step2"
```

### Pilot Validation Results (2026-11-09)

Two training iterations and one resumed iteration completed with mixed-reward groups and gradient norms of 5.14, 8.23, and 6.04, without OOM or NaN. The step-2 export changed 97 action-head tensors and no frozen tensors. Resume logs confirm applied weight version 2 and a shared training initialization across all 8 trajectories in the group. Official evaluation completed two development trajectories each for SG6/B2N/B2P with video: success once was 1/2, 1/2, 0/2; success at end was 1/2, 0/2, 0/2. These are loading checks, not evidence of transfer gains.

Slurm jobs 19470/19472/19473 completed, using 0.4233 GPU hours on the training side and 0.0594 on evaluation. Results are under `$FYP_STORAGE/results`: `sg6-grpo-pilot-v1`, `sg6-grpo-resume-v1`, and `sg6-grpo-pilot-step2-smoke`. The matched-method comparison, layout perturbations, and full held-out evaluation have not run. Formal training must restart both conditions from the same original model; neither may start from a pilot checkpoint.

Within the single-GPU evaluation Slurm job described above, reproduce the loading check with the command below and a fresh output directory:

```bash
bash "$FYP_RUNTIME/fyp-eval-job.sh" \
  --model-path "$FYP_STORAGE/models/sg6-grpo-pilot-step2" \
  --tasks SG6 B2N B2P \
  --variant-root "$FYP_STORAGE/variants/sg6-expanded-20261109" \
  --episodes 2 --trial-start 0 --num-envs 2 --video \
  --budget-minutes 20 \
  --output-dir "$FYP_STORAGE/results/sg6-grpo-pilot-step2-smoke"
```

## Background Comparison

The user authorized continued execution with wake-run. Each method receives 20 iterations and 56,000 simulator action steps (8 environments × 350 steps × 20 iterations, including post-success interaction). Save at iterations 10 and 20; use iteration 20 for the primary comparison. Both restart from the original model with seed 42 and run sequentially on at most two GPUs, with a three-hour Slurm cap per condition. Retries remain inside the 48 GPU-hour training cap. Final evaluation covers initialization IDs 30–49 for SG6/A1/A2/B2N/B2P/G1/SG7 under the original and both RL models: 420 planned trajectories within the 8 GPU-hour evaluation cap.

The first stage only generates and checks layouts. Generate four layouts for each training initialization 0–19, moving only the cheese and black bowl by at most 1 cm per axis within their original BDDL regions. Keep rotations, mass, friction, and other objects unchanged. Check non-successful starts, finite states, penetration, and settling motion. This finite bank does not cover arbitrary continuous perturbations. Only validated layouts enter training, with a separate RNG preserving the source-initialization sampling sequence across methods and one shared layout within each group.

After logging into the server and activating the personal environment, submit this first stage below. `--wait` waits for Slurm completion so local wake-run can resume this chat when the process exits. Do not submit a duplicate manually.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
sbatch --wait --parsable \
  --output="$FYP_STORAGE/results/layout-prepare-%j.log" \
  evaluations/fyp/prepare_layouts_job.sh
```

The first layout job (Slurm 19475) rejected an official source position just outside the BDDL sampling region before physical validation. Sampling now requires a nonempty intersection between the source ±1 cm and the original region; new positions must remain inside the original region. Preserve the failed v1 record and write new output to `sg6-layouts-v2`.

### Automated Execution and Validation

`sg6-layouts-v2` passed all 80 layout checks (four per training state 0–19). Training uses a separate layout RNG and shares one layout across each GRPO group. Evaluation and partial-group resets cannot use the layout bank.

`run_comparison.sh` submits four stages sequentially: one layout integration iteration, 20 ordinary RL iterations, 20 diversified RL iterations, and complete held-out evaluation of all three models. Each stage waits through `sbatch --wait`; a failure stops subsequent stages. Local wake-run waits for the driver process to exit and notifies this chat. Maximum requested allocation is 13 training GPU hours plus prior usage, and 3 evaluation GPU hours for this comparison, within the total caps. Do not rerun an existing comparison directory; inspect a failure before deciding how to resume.

Training validation requires finite metrics, at least one mixed-reward group and nonzero gradient, and exact per-iteration simulator action counts. Both methods must share the source-initialization group sequence. Do not confuse post-termination loss-mask counts with physical interaction counts. Export requires changed action-head weights and unchanged frozen weights. Simulator action counts include post-success actions but exclude initialization settling.

Use the commands below to launch or inspect records; launch only once. Exports go to `$FYP_STORAGE/models/sg6-comparison-v1-{plain,diverse}`; metrics, commands, checkpoints, and evaluations go to `$FYP_STORAGE/results/sg6-comparison-v1`. Final evaluation uses the fixed iteration-20 checkpoints, never held-out scores to select a checkpoint.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
bash evaluations/fyp/run_comparison.sh
```

```bash
squeue -u "$USER"
tail -n 30 "$FYP_STORAGE/results/sg6-comparison-v1/plain/console.log"
cat "$FYP_STORAGE/results/sg6-comparison-v1/plain/training-audit.json"
```

## Pi0.5 Base Feasibility Check (In Progress)

Use the official `gs://openpi-assets/checkpoints/pi05_base` checkpoint. Download, convert, and verify the control interface before evaluating a few development initializations. This stage runs neither SFT nor RL. The official release distinguishes base and LIBERO-adapted checkpoints; that distinction alone does not prove the full training history excludes LIBERO.

Run the following on the server. The downloader records official object generations, resumes partial files, and checks byte counts. Generations identify source versions; they are not a separate SHA256 repository registry.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
python evaluations/fyp/download_pi05_base.py \
  --output "$FYP_STORAGE/models/pi05_base_jax"
```

A completed download does not validate policy behavior or success rates. The source weights use JAX; conversion will reuse RLinf’s `jax_to_openpi` tool. Included assets cover Franka/DROID and other embodiments, but not LIBERO. Establish state/action semantics before evaluation instead of silently borrowing LIBERO-finetuned normalization. Keep the existing Evo-1 environment separate.

The official download is complete (29 files, 12.44 GB). If the server cannot resolve Google Storage, forward the existing local proxy for the SSH session with `-R 127.0.0.1:17890:127.0.0.1:7890` and set `HTTPS_PROXY=http://127.0.0.1:17890` and `HTTP_PROXY=http://127.0.0.1:17890` only for the remote command. Keep TLS verification enabled; the tunnel closes with SSH.

Next, create the conversion environment and submit a 30-minute Slurm job with eight CPUs and no GPU:

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
bash evaluations/fyp/prepare_pi05_base.sh
```

The `rlinf-pi05-check` environment reads existing Evo-1 dependencies through a `.pth` file and installs additions into its own directory. It is not a fully independent dependency copy; preserve the source environment for reproducibility. Conversion retains a 50-step action horizon, 32 internal action dimensions, and 200 tokens. Original Franka normalization is copied unchanged, not treated as LIBERO normalization. Existing conversion output is never overwritten; inspect failure logs before retrying. This stage does not yet validate policy inference or environment control.

Conversion completed in Slurm job 19488. Next, check weight loading and two synthetic inference passes on one GPU:

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
  --gres=gpu:4090:1 --cpus-per-task=8 --time=00:30:00 \
  --job-name=fyp-pi05-infer \
  --output="$FYP_STORAGE/results/pi05-base-check/inference-%j.log" \
  evaluations/fyp/check_pi05_inference_job.sh
```

The JSON report records finite outputs, latency and peak GPU memory. Synthetic images and token IDs do not validate language processing, LIBERO control or task success; this check runs zero environment episodes.

Slurm job 19492 passed the synthetic inference check using commit `31005946`: 3,353,433,872 parameters, output shape `[1, 50, 32]`, all finite, and zero maximum difference between two passes with fixed noise. The first and second passes took 0.717 and 0.310 seconds. Peak allocated/reserved GPU memory was 6.649/6.805 GiB on one RTX 4090. These measurements exclude simulation and prompt tokenization and do not establish RL training memory requirements. No LIBERO episodes have run for this checkpoint.

The next evaluation requires an explicit normalization protocol. The downloaded base assets have no LIBERO statistics; the copied Franka statistics are not validated for LIBERO. Using public LIBERO statistics would be target-domain interface calibration, even with unchanged base weights, and must be reported separately from a claim of no LIBERO data exposure. Calibration alone would not prove correct control semantics.

The user-approved next protocol is unchanged base weights with public LIBERO interface calibration. Prepare preprocessing dependencies and only the official 1.9 KB statistics file (no fine-tuned weights or demonstrations):

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
bash evaluations/fyp/prepare_pi05_calibration.sh
```

The first calibrated rollout checks SG6 development states 0 and 1, records videos, and uses a 350-step limit. The base-specific preprocessing config retains discrete state tokens and a 50-step prediction horizon, with 10 denoising steps and 5 executed actions per call. Pydantic is pinned to 2.10.6 for numpydantic 1.6.6 compatibility. Run preparation before allocating a GPU; inspect completed trajectories rather than treating command success as task success.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
bash evaluations/fyp/prepare_pi05_calibration.sh
sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
  --gres=gpu:4090:1 --cpus-per-task=8 --time=00:30:00 \
  --job-name=fyp-pi05-libero \
  --output="$FYP_STORAGE/results/pi05-base-check/libero-%j.log" \
  evaluations/fyp/check_pi05_libero_job.sh
```

Slurm job 19494 completed two distinct SG6 development trajectories (IDs 0 and 1): success_once 0/2 and success_at_end 0/2, each reaching 350 steps without an environment error in the log. Logged episode durations were 38.794 and 38.244 seconds; total single-GPU allocation was 165 seconds. Sampled video frames show arm motion without observed target pickup or placement. This verifies an executable pipeline, not correct domain/control alignment or useful zero-shot task ability. Two failures do not estimate overall performance. No training was started. Full logs and videos are under `$FYP_STORAGE/results/pi05-base-check/libero-19494`.

Screen three additional tasks, five development states (0–4) each: S1 bowl to plate, O2 cream cheese to basket, G1 open middle drawer. All 15 episodes save videos. Tasks run sequentially on one GPU, retain the calibrated-base protocol and 350-step limit, and write separate per-task directories. This is screening, not training or an Evo comparison under matched protocols.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
  --gres=gpu:4090:1 --cpus-per-task=8 --time=00:45:00 \
  --job-name=fyp-pi05-screen \
  --output="$FYP_STORAGE/results/pi05-base-check/screen-%j.log" \
  evaluations/fyp/screen_pi05_libero_job.sh
```

Expanded screening completed on 2026-11-10 (Slurm 19560, Git 1b69cea4): S1, O2 and G1 each completed development states 0–4, with success_once and success_at_end both 0/5 for every task. All 15 episodes reached 350 steps; five videos were saved per task. One GPU (physical index 2) ran for 12m57s. Job 19557 failed during model loading from GPU contention and completed no trajectories; exclude it from task failure rates. Together with SG6 (0/2), no success was observed in 17 attempts across four tasks. This is not a full LIBERO benchmark or a conclusion about other checkpoints. A LIBERO-adapted positive control is recommended next; no control run or training has started. Results: `$FYP_STORAGE/results/pi05-base-check/libero-19560-{S1,O2,G1}`.

The next positive control uses official `gs://openpi-assets/checkpoints/pi05_libero`, stored separately from base weights. The official model config uses action horizon 10 and `discrete_state_input=False`; do not reuse the base shape blindly. Only inference is planned, on S1/O2/G1 with five development states each. Download the adapted weights first; conversion and evaluation follow after integrity checks.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
python evaluations/fyp/download_pi05_base.py --checkpoint pi05_libero \
  --output "$FYP_STORAGE/models/pi05_libero_jax"
```

The official adapted checkpoint download contains 16 objects. Convert it in a separate CPU Slurm job, retaining its original LIBERO statistics. Evaluation must explicitly use horizon 10 and disable discrete state input; the converter metadata alone does not configure RLinf inference.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
  --cpus-per-task=8 --time=00:30:00 --job-name=fyp-pi05-control-convert \
  --output="$FYP_STORAGE/results/pi05-base-check/control-convert-%j.log" \
  evaluations/fyp/convert_pi05_libero_job.sh
```

Run the adapted positive control on S1/O2/G1, five development states each. This loads the official pi05_libero converted weights and their own statistics, with a 10-step horizon and no discrete state tokens. Results use `control-<job>-<task>` directories and must not be merged with base screening. Both conditions execute five actions per call with ten denoising steps and a 350-step limit. This is a pipeline diagnostic, not an equal-checkpoint comparison.

```bash
source "$HOME/.config/fyp/evo1-env.sh"
cd "$HOME/RLinf"
sbatch --wait --parsable --partition=GPU --nodelist="$FYP_NODE" \
  --gres=gpu:4090:1 --cpus-per-task=8 --time=00:45:00 \
  --job-name=fyp-pi05-control \
  --output="$FYP_STORAGE/results/pi05-base-check/control-%j.log" \
  evaluations/fyp/screen_pi05_libero_job.sh control
```
