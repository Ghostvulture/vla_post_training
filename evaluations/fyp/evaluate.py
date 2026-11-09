# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch the official Evo-1 evaluator once per selected LIBERO task."""

import argparse
import importlib.metadata
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("tasks.json")
LOG = logging.getLogger("fyp-evaluation")
PROTOCOL = {
    "config": "libero_spatial_evo1_eval",
    "image_size": 448,
    "max_episode_steps": 350,
    "num_action_chunks": 14,
    "seed": 0,
    "success_metric": "success_once",
    "arm_key": "libero_robot",
    "libero_type": "standard",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate the small screening budget before importing server packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        nargs="+",
        type=Path,
        help="Merge saved run directories; no GPU needed",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["S1", "S2", "O1", "O2", "G1", "G2"]
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Distinct initialization states per task",
    )
    parser.add_argument(
        "--variant-root", type=Path, help="Server-generated immutable task variants"
    )
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument(
        "--gpu", type=int, default=0, help="Single RLinf resource rank on the server"
    )
    parser.add_argument("--budget-minutes", type=float, default=1200)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--evo-repo", type=Path)
    parser.add_argument("--vlm-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--video", action="store_true", help="Smoke tests only: at most 2 episodes/task"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without asset checks or execution",
    )
    args = parser.parse_args(argv)
    if args.report:
        return args
    for name in ("model_path", "evo_repo", "vlm_path", "output_dir"):
        if getattr(args, name) is None:
            parser.error(
                f"--{name.replace('_', '-')} is required unless --report is used"
            )
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not 1 <= args.episodes <= 20 or args.trial_start < 0:
        parser.error("Expected 1..20 episodes and a nonnegative trial start")
    if args.episodes % args.num_envs:
        parser.error(
            "episodes must be divisible by num-envs to avoid padding or extra trials"
        )
    if args.gpu < 0 or not 0 < args.budget_minutes <= 1440:
        parser.error(
            "Expected a nonnegative GPU rank and a budget in (0, 1440] minutes"
        )
    if args.video and args.episodes > 2:
        parser.error("Use --video only for a smoke test of at most 2 episodes/task")
    keys = {task["key"] for task in json.loads(MANIFEST.read_text())}
    if len(set(args.tasks)) != len(args.tasks) or set(args.tasks) - keys:
        parser.error(f"Tasks must be distinct members of {sorted(keys)}")
    return args


def build_command(
    args: argparse.Namespace, task: dict, trial_ids: list[int]
) -> list[str]:
    """Keep the official model, action, camera and horizon configuration."""
    output = args.output_dir / task["key"] / "official_logs"
    command = [
        "bash",
        str(ROOT / "evaluations/run_eval.sh"),
        "libero",
        PROTOCOL["config"],
        "~cluster.component_placement",
        f"+cluster.component_placement={{env:{args.gpu},rollout:{args.gpu}}}",
        f"env.eval.task_suite_name={task['suite']}",
        f"+env.eval.task_id_filter=[{task['task_id']}]",
        f"+env.eval.eval_trial_ids={json.dumps(trial_ids, separators=(',', ':'))}",
        f"env.eval.total_num_envs={args.num_envs}",
        f"env.eval.rollout_epoch={len(trial_ids) // args.num_envs}",
        "env.eval.use_fixed_reset_state_ids=true",
        "env.eval.use_ordered_reset_state_ids=true",
        "env.eval.specific_reset_id=null",
        "env.eval.seed=0",
        f"env.eval.video_cfg.save_video={str(args.video).lower()}",
        f"rollout.model.model_path={json.dumps(str(args.model_path))}",
        f"rollout.model.evo1.repo_path={json.dumps(str(args.evo_repo))}",
        f"+rollout.model.evo1.vlm_name={json.dumps(str(args.vlm_path))}",
        f"runner.logger.log_path={json.dumps(str(output))}",
        f"runner.logger.experiment_name=fyp_{task['key']}",
    ]
    if task.get("scene_properties"):
        props = task["scene_properties"]
        command.append(
            "+env.eval.init_params.scene_properties={"
            + ",".join(f"{key}:{value}" for key, value in props.items())
            + "}"
        )
    if task.get("variant"):
        if args.variant_root is None:
            raise ValueError("Variant tasks require --variant-root")
        root = args.variant_root.resolve() / task["variant"]
        command.extend(
            [
                f"+env.eval.task_bddl_override={json.dumps(str(root / 'task.bddl'))}",
                f"+env.eval.task_init_states_override={json.dumps(str(root / 'states.pt'))}",
            ]
        )
    return command


def git_revision(path: Path) -> str:
    """Record a Git commit; do not create a parallel checksum registry."""
    checkout = Path(
        subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    if path != ROOT and checkout == ROOT:
        raise ValueError(f"Expected a separate Evo-1 Git checkout at {path}")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    )
    if dirty:
        raise ValueError(
            f"Commit changes before running; Git checkout is dirty: {checkout}"
        )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def check_assets(args: argparse.Namespace) -> None:
    """Check local server assets without loading model weights."""
    for name in ("config.json", "norm_stats.json", "mp_rank_00_model_states.pt"):
        if not (args.model_path / name).is_file():
            raise ValueError(f"Missing Evo1_LIBERO file: {args.model_path / name}")
    stats = json.loads((args.model_path / "norm_stats.json").read_text())
    if "libero_robot" not in stats:
        raise ValueError("norm_stats.json must contain the official libero_robot key")
    if not any(
        (path / "scripts/Evo1.py").is_file()
        for path in (args.evo_repo, args.evo_repo / "Evo_1")
    ):
        raise ValueError(f"Expected an installed Evo-1 checkout at {args.evo_repo}")
    if not (args.vlm_path / "config.json").is_file():
        raise ValueError(
            f"Expected the complete InternVL3-1B directory at {args.vlm_path}"
        )
    if not any(args.vlm_path.glob("*.safetensors")) and not any(
        args.vlm_path.glob("*.bin")
    ):
        raise ValueError(f"No InternVL3-1B weight files found at {args.vlm_path}")


def resolve_tasks(tasks: list[dict], trial_ids: list[int]) -> tuple[list[dict], Path]:
    """Resolve IDs by exact task name in the server's installed benchmark."""
    import libero.libero as libero_core
    from libero.libero import benchmark

    # Match LiberoEnv's installed-package paths before reading initial states.
    installed_root = str(Path(libero_core.__file__).resolve().parent)
    if libero_core.get_libero_path("benchmark_root") != installed_root:
        libero_core.set_libero_default_path(installed_root)
    registry = benchmark.get_benchmark_dict()
    resolved = []
    for task in tasks:
        suite = registry[task["suite"]]()
        matches = [
            i for i in range(suite.n_tasks) if suite.get_task(i).name == task["name"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one match for {task['name']}, found {matches}")
        task_id = matches[0]
        count = len(suite.get_task_init_states(task_id))
        if max(trial_ids) >= count:
            raise ValueError(
                f"{task['name']} has {count} initial states, requested {trial_ids}"
            )
        resolved.append({**task, "task_id": task_id, "available_trials": count})
    return resolved, Path(benchmark.__file__).resolve().parent


def physical_gpu(gpu: int) -> int:
    """Map a RLinf rank through the same visible-device list as its scheduler."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return gpu
    try:
        devices = [int(value.strip()) for value in visible.split(",")]
        if any(value < 0 for value in devices) or gpu >= len(devices):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"GPU rank {gpu} is not valid for CUDA_VISIBLE_DEVICES={visible!r}"
        ) from exc
    return devices[gpu]


def gpu_sample(gpu: int) -> str:
    """Sample device memory, including other users' allocations on this device."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def stop_process(process: subprocess.Popen) -> None:
    """Stop this launcher's process group without stopping a shared Ray cluster."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=15)
        return
    until = time.monotonic() + 5
    while time.monotonic() < until:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=15)


def run_task(
    command: list[str], directory: Path, deadline: float, gpu: int
) -> tuple[int | None, float, str | None]:
    """Run the official script and preserve output even when its process fails."""
    environment = dict(
        os.environ,
        RAY_DEDUP_LOGS="0",
        LIBERO_TYPE="standard",
        ROBOT_PLATFORM="LIBERO",
        PYTHONUNBUFFERED="1",
    )
    # An inherited RAY_ADDRESS could attach to another user's Ray cluster.
    environment["RAY_ADDRESS"] = "local"
    environment["RAY_TMPDIR"] = tempfile.mkdtemp(prefix="fyp-ray-")
    environment.pop("EVO1_REPO_PATH", None)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    write_json(
        directory / "runtime.json",
        {
            key: environment[key]
            for key in ("RAY_ADDRESS", "RAY_TMPDIR", "RAY_DEDUP_LOGS", "LIBERO_TYPE")
        },
    )
    started = time.monotonic()
    error = None
    returncode = None
    with (
        (directory / "console.log").open("w") as output,
        (directory / "gpu-memory.csv").open("w") as gpu_log,
    ):
        gpu_log.write(
            "timestamp,gpu_uuid,gpu_name,memory_used_mib,memory_total_mib,utilization_percent\n"
        )
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    error = "Evaluation time budget exhausted"
                    stop_process(process)
                    break
                gpu_log.write(gpu_sample(gpu) + "\n")
                gpu_log.flush()
                try:
                    process.wait(timeout=min(5, max(0.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            returncode = process.returncode
        except (OSError, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                stop_process(process)
                returncode = process.returncode
    return returncode, time.monotonic() - started, error


def write_json(path: Path, value: dict) -> None:
    """Write a readable record next to raw results."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Keep dry-run and reporting usable on a CPU-only development computer."""
    args = parse_args(argv)
    from evaluations.fyp.report import inspect_task, summarize_runs, write_summary

    if args.report:
        rows = summarize_runs(args.report)
        LOG.info("%s", json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    manifest = json.loads(MANIFEST.read_text())
    tasks = [
        next(task for task in manifest if task["key"] == key) for key in args.tasks
    ]
    trials = list(range(args.trial_start, args.trial_start + args.episodes))
    if args.dry_run:
        for task in tasks:
            command = build_command(
                args, {**task, "task_id": f"RESOLVE_{task['key']}_BY_NAME"}, trials
            )
            LOG.info("%s\n%s", task["name"], shlex.join(command))
        LOG.info(
            "Dry-run only; task IDs are placeholders until resolved on the server."
        )
        return 0
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    if args.output_dir.exists():
        raise ValueError(
            f"Output directory already exists; choose a fresh path: {args.output_dir}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty:
        raise ValueError(
            "Commit local changes, push, and pull on the server before running; checkout is dirty"
        )
    check_assets(args)
    sampled_gpu = physical_gpu(args.gpu)
    tasks, libero_path = resolve_tasks(tasks, trials)
    if any(task.get("variant") for task in tasks):
        if args.variant_root is None:
            raise ValueError("Variant tasks require --variant-root")
        for task in tasks:
            if task.get("variant"):
                folder = args.variant_root.resolve() / task["variant"]
                validation = json.loads((folder / "validation.json").read_text())
                if not all(
                    validation.get(key)
                    for key in ("valid", "goal_witness_success", "settling_valid")
                ):
                    raise ValueError(f"Variant has not passed validation: {folder}")
                task["variant_root"] = str(args.variant_root.resolve())
                task["variant_validation"] = validation
    args.output_dir.mkdir(parents=True)
    metadata = {
        "git_commit": git_revision(ROOT),
        "model_path": str(args.model_path),
        "vlm_path": str(args.vlm_path),
        "evo_repo": str(args.evo_repo),
        "evo_git_commit": git_revision(args.evo_repo),
        "libero_version": importlib.metadata.version("rlinf-libero"),
        "libero_package_path": str(libero_path),
        "protocol": {
            **PROTOCOL,
            "num_envs": args.num_envs,
            "variant_root": str(args.variant_root.resolve())
            if args.variant_root
            else None,
        },
        "tasks": tasks,
        "trial_ids": trials,
        "budget_minutes": args.budget_minutes,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "gpu_resource_rank": args.gpu,
        "gpu_sample_device": sampled_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "ray", "hydra-core")
        },
        "status": "running",
    }
    write_json(args.output_dir / "run.json", metadata)
    summaries = []
    for task in tasks:
        if time.monotonic() >= deadline:
            metadata["error"] = "Time budget exhausted before starting remaining tasks"
            break
        directory = args.output_dir / task["key"]
        directory.mkdir()
        command = build_command(args, task, trials)
        write_json(directory / "command.json", {"argv": command, "task": task})
        LOG.info(
            "Running %s (%s), trials=%s; log: %s",
            task["key"],
            task["name"],
            trials,
            directory / "console.log",
        )
        code, elapsed, error = run_task(command, directory, deadline, sampled_gpu)
        summary = inspect_task(directory, task, trials, code, elapsed, error)
        summaries.append(summary)
        write_summary(args.output_dir / "summary.csv", summaries)
        LOG.info(
            "%s: status=%s, completed=%s, successes=%s, wall=%.1fs",
            task["key"],
            summary["status"],
            summary["num_completed"],
            summary["successes"],
            elapsed,
        )
        if summary["status"] != "complete":
            break
    completed = [
        summary["task"]["key"]
        for summary in summaries
        if summary["status"] == "complete"
    ]
    metadata.update(
        status="complete" if len(completed) == len(tasks) else "incomplete",
        completed_tasks=completed,
        wall_seconds=time.monotonic() - started,
    )
    write_json(args.output_dir / "run.json", metadata)
    return 0 if metadata["status"] == "complete" else 1


if __name__ == "__main__":
    # The CLI must work before Ray or the embodied dependencies are installed.
    sys.path.insert(0, str(ROOT))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        LOG.error("Evaluation could not complete: %s", exc)
        sys.exit(1)
