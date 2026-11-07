# FYP: Evaluate the Evo-1 Baseline

[中文](README.zh-CN.md)

This guide takes you from editing and checking evaluation code locally to running a six-task Evo-1 baseline on a server through Git. Start with a two-episode server smoke test to check model loading, rendering, and result recording, then evaluate ten episodes per task.

The project remote is `https://github.com/Ghostvulture/vla_post_training.git`. Server access, paths and runtime validation still need to be verified. These commands describe the workflow; they do not represent measured success rates. Replace the example `/srv/fyp` directory with a writable location on your server.

## Divide Work Between Your Computer and the Server

Use your computer to edit code and the server to run the model and simulator. Local checks cover arguments, generated commands, log processing, and initial-state selection. They do not establish that the model loads or completes a task.

| Work | Your Computer | Server |
|---|---|---|
| Edit code, configuration, and documentation | Edit with Codex | Receive through Git |
| CPU unit tests and `--dry-run` | Run | Optionally repeat after updates |
| Download models and simulator assets | Skip | Download and retain |
| Model inference, MuJoCo rendering, smoke tests | Skip | Run |
| Baseline evaluation and later RL training | Skip | Run |
| Read CSV, JSON, and logs | Optionally copy small files for analysis | Retain original records |

This stage reuses the [official evaluation launcher](../run_eval.sh) and [Evo-1 evaluation configuration](../libero/libero_spatial_evo1_eval.yaml). Evaluate the released checkpoint without SFT or demonstration downloads. The installer downloads the simulator assets needed to run LIBERO; those are different from SFT demonstration trajectories.

## Check the Code Locally

Run CPU checks from your local repository root. Use an existing test environment; install missing test dependencies such as `pytest` according to that environment's requirements. Keep the server installation commands on the server.

```bash
cd /home/xyz/code/works/VLA_test/RLinf
python3 -m pytest -q \
  tests/unit_tests/test_fyp_evaluation.py \
  tests/unit_tests/test_fyp_report.py \
  tests/unit_tests/test_libero_env.py
```

These tests cover evaluation configuration, reporting, and LIBERO initial-state selection. They do not load Evo-1, execute simulations, or produce success rates.

Generate the planned commands using server paths to inspect the task and episode settings:

```bash
python3 evaluations/fyp/evaluate.py \
  --model-path /srv/fyp/models/Evo1_LIBERO \
  --evo-repo /srv/fyp/RLinf/.venv/Evo-1 \
  --vlm-path /srv/fyp/models/InternVL3-1B \
  --output-dir /srv/fyp/results/evo1-baseline-batch1 \
  --episodes 10 --trial-start 0 --num-envs 1 --gpu 0 \
  --budget-minutes 1200 --dry-run
```

`--dry-run` only prints commands. It neither imports model or simulator modules nor writes the output directory. Task IDs are resolved by exact full-name matching against the server's installed LIBERO tasks. Local output contains placeholders, not final runnable task IDs; run the Python entry point below so it can resolve them.

## Synchronize Code Through Git

The local checkout uses the dedicated `main` branch, with `upstream` pointing to official RLinf. Add your own remote repository as `origin`; keep project pushes on that remote. Review and commit the changes to synchronize first, using your Git identity and `git commit -s`.

Publish the branch from your computer:

```bash
FYP_REMOTE_URL='https://github.com/Ghostvulture/vla_post_training.git'
git remote add origin "$FYP_REMOTE_URL"
git push -u origin main
```

Replace `FYP_REMOTE_URL` first. `git push` transfers committed code, not uncommitted changes, models, or results. If `origin` already exists, inspect it with `git remote -v` instead of adding it again.

After configuring SSH, connect from your computer:

```bash
FYP_SSH_HOST='your-ssh-alias'
ssh "$FYP_SSH_HOST"
```

Replace `FYP_SSH_HOST` with a configured SSH alias or `username@server-address`. Manage SSH keys through your local SSH configuration, outside the repository. Run the installation and evaluation commands below in the server terminal.

Clone the project on the server for the first time:

```bash
mkdir -p /srv/fyp
cd /srv/fyp
FYP_REMOTE_URL='https://github.com/Ghostvulture/vla_post_training.git'
git clone --branch main "$FYP_REMOTE_URL" RLinf
cd /srv/fyp/RLinf
```

The server needs read access to this remote repository. For later updates, test and commit locally, then run `git push`. After the current server job ends, run `git pull --ff-only` in the server checkout before launching another batch. Keep a running evaluation's code unchanged. Execution requires a clean server checkout; uncommitted changes prevent launch.

## Install the Environment and Models on the Server

Check the allocated NVIDIA GPU, available GPU memory, and disk space before installing the Evo-1/LIBERO environment for this RLinf version.

```bash
cd /srv/fyp/RLinf
nvidia-smi
df -h /srv/fyp
bash requirements/install.sh embodied --model evo1 --env libero --no-root
source .venv/bin/activate
```

The [official installer](../../requirements/install.sh) supports this combination. `--no-root` skips system dependency installation and applies only when an administrator has already installed those dependencies. It does not supply missing system libraries. Resolve missing-library or compiler errors on the server before retrying. Add `--use-mirror` according to the official instructions if your network needs mirrors. Prefer repository, model, and output paths without spaces.

The installer puts the RLinf Evo-1 fork under `.venv/Evo-1`, or uses `EVO1_PATH` when set. Check that location before passing it to evaluation:

```bash
FYP_EVO_REPO="${EVO1_PATH:-$PWD/.venv/Evo-1}"
test -d "$FYP_EVO_REPO"
realpath "$FYP_EVO_REPO"
```

Adjust `FYP_EVO_REPO` to the installed location if you used a custom `--venv`. This path follows the installer; it does not establish that installation on your server has succeeded.

Download [Evo1_LIBERO](https://huggingface.co/MINT-SJTU/Evo1_LIBERO) and its base VLM, [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B):

```bash
mkdir -p /srv/fyp/models /srv/fyp/results
hf download MINT-SJTU/Evo1_LIBERO \
  --local-dir /srv/fyp/models/Evo1_LIBERO
hf download OpenGVLab/InternVL3-1B \
  --local-dir /srv/fyp/models/InternVL3-1B
```

The installed `huggingface_hub` package supplies `hf`. The Evo checkpoint directory should contain `config.json`, `norm_stats.json`, and `mp_rank_00_model_states.pt`; store the base VLM separately. Pass both as server-local paths to avoid downloads during model loading. Keep models and simulator assets on the server, outside Git.

## Run a Small Server Smoke Test

Run two S1 episodes with video first. Define shared arguments so subsequent commands use the same model paths and GPU settings:

```bash
cd /srv/fyp/RLinf
source .venv/bin/activate
FYP_EVO_REPO="${EVO1_PATH:-$PWD/.venv/Evo-1}"
FYP_EVAL_ARGS=(
  --model-path /srv/fyp/models/Evo1_LIBERO
  --evo-repo "$FYP_EVO_REPO"
  --vlm-path /srv/fyp/models/InternVL3-1B
  --num-envs 1
  --gpu 0
)
python evaluations/fyp/evaluate.py "${FYP_EVAL_ARGS[@]}" \
  --tasks S1 --episodes 2 --trial-start 0 --video \
  --output-dir /srv/fyp/results/evo1-smoke \
  --budget-minutes 60
```

This checks weight loading, action execution, 448×448 rendering, episode records, and video. Zero success can still be a valid measurement; a crash, missing episodes, or absent actions needs investigation first. `--video` permits at most two episodes per task. The baseline leaves video disabled.

Read `wall_seconds` in the task summary and use `wall_seconds / 2` to estimate the observed time per episode. Ten episodes across six tasks also involve six task startups, and task lengths can differ; reserve startup time and margin. This is a budget estimate, not a performance promise.

`--budget-minutes` limits the whole batch for one evaluation command and defaults to 1200 minutes. Installation, downloads, the smoke test, and earlier batches are not automatically charged to the next batch. The project targets 24 hours including setup; subtract time already spent before passing the remaining budget.

`--gpu` is an index within the devices visible to RLinf. For example, use `--gpu 0` when the server sets `CUDA_VISIBLE_DEVICES=3`; memory sampling maps this to physical GPU 3. `gpu-memory.csv` samples whole-device usage approximately every five seconds, including other processes, so it is not an exact peak for this model alone. Each task uses a separate local Ray instance and temporary directory; the launcher does not run machine-wide `ray stop`.

## Evaluate the Six-Task Baseline on the Server

After the smoke test passes, evaluate ten episodes on each fixed task. [tasks.json](tasks.json) defines the tasks. Before execution, their full names must match the server's LIBERO task table exactly; investigate version or task-table mismatches instead of guessing IDs.

| Label | Task Description | Suite |
|---|---|---|
| S1 | Place the black bowl from the table center on the plate | LIBERO-Spatial |
| S2 | Place the black bowl beside the cookie box on the plate | LIBERO-Spatial |
| O1 | Place alphabet soup in the basket | LIBERO-Object |
| O2 | Place cream cheese in the basket | LIBERO-Object |
| G1 | Open the middle drawer | LIBERO-Goal |
| G2 | Push the plate to the stove front | LIBERO-Goal |

These labels identify baseline evaluation tasks; they do not preselect later A/B/C training and transfer groups. Omitting `--tasks` selects all six. Pass a subset such as `--tasks S1 S2` to restrict a run.

```bash
python evaluations/fyp/evaluate.py "${FYP_EVAL_ARGS[@]}" \
  --episodes 10 --trial-start 0 \
  --output-dir /srv/fyp/results/evo1-baseline-batch1 \
  --budget-minutes 1200
```

Replace `1200` with the actual remaining minutes. Evaluation retains the official Evo configuration's 350 simulation steps, 448×448 images, and fourteen-step action chunks. Keep those settings consistent across comparisons. If the budget only supports a smaller run, use `--episodes 5` with a new `evo1-baseline-fallback` directory, and report five episodes per task.

To expand to twenty episodes per task, use a new directory and the next initial-state range:

```bash
python evaluations/fyp/evaluate.py "${FYP_EVAL_ARGS[@]}" \
  --episodes 10 --trial-start 10 \
  --output-dir /srv/fyp/results/evo1-baseline-batch2 \
  --budget-minutes 1200
```

The first batch uses trials 0–9 and the second uses 10–19. `--trial-start` selects the initial-state offset; changing the output directory alone does not create new samples. Use a fresh output directory: the script rejects existing run directories. Recalculate the remaining budget before each extension, and retain identical code versions, model files, model paths, and evaluation protocols. Do not replace weights in place and combine the resulting runs.

## Inspect and Combine Results

Each run directory retains run information, episode records, and summaries for later comparisons with post-trained models under the same conditions.

```text
evo1-baseline-batch1/
├── run.json
├── summary.csv
├── S1/
│   ├── console.log
│   ├── command.json
│   ├── gpu-memory.csv
│   ├── episodes.csv
│   ├── summary.json
│   └── official_logs/
└── ...
```

When enabled, videos are stored within the official log directory. Check completed episode counts and error status before interpreting success rates. A timeout, crash, or incomplete run is not a completed ten-episode experiment. Ten episodes per task support an initial diagnosis, not claims about small differences or cross-task transfer.

Combine two saved batches with:

```bash
python evaluations/fyp/evaluate.py --report \
  /srv/fyp/results/evo1-baseline-batch1 \
  /srv/fyp/results/evo1-baseline-batch2
```

`--report` reads saved results and prints a combined summary without loading a model or launching simulation. It checks provenance and protocol consistency and rejects duplicate or overlapping trials. The two batches contain at most twenty distinct trials per task. The smoke test's S1 trials 0–1 overlap with the first batch, so exclude its directory from the baseline report. You can also copy `run.json` and each task's `summary.json` and `episodes.csv` to your computer, preserving the directory structure, and run the same reporting command locally.

## Plan RL After the Baseline

Use the six-task baseline, failure videos, and observed execution costs to choose training and held-out evaluation tasks before setting a server training budget. Retain the baseline scores, task definitions, and trial ranges for capability-retention comparisons. Later, update the model only on the selected training tasks and evaluate the other tasks with the same protocol.

Use the [official Evo-1 GRPO configuration](../../examples/embodiment/config/libero_spatial_grpo_evo1.yaml) as the starting point for RL on A, freezing the vision-language backbone and updating the action head. Verify rewards, actual parameter updates, weight synchronization and checkpoint recovery before fixing a budget and repeating with three training seeds. Confirm the checkpoint, task groups, available GPUs, stopping conditions, and evaluation frequency before training. This entry point evaluates the baseline; it does not select A/B/C groups or launch training automatically.
