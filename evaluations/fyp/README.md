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
