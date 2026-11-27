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

"""Compare local base/PPO weights using matched native RoboTwin episodes."""

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import types

if __package__:
    from .robotwin_qualify import SOURCE_COMMIT, bind_allocated_gpu
else:
    from robotwin_qualify import SOURCE_COMMIT, bind_allocated_gpu

LOG = logging.getLogger(__name__)


def write_json(path: Path, value: dict) -> None:
    """Write machine-readable results without accepting nonfinite numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize_pairs(base: list, ppo: list, seeds: list, same_scene) -> dict:
    """Reject missing/error trajectories and unmatched scenes before scoring."""
    for rows in (base, ppo):
        if [row["reset_seed"] for row in rows] != seeds:
            raise ValueError("Expected exactly the requested ordered reset seeds")
        if any(row["status"] != "completed" for row in rows):
            raise ValueError("Incomplete/error trajectories cannot enter success rates")
    pairs = []
    for left, right in zip(base, ppo):
        if left["policy_seed"] != right["policy_seed"]:
            raise ValueError("Paired policy noise seeds differ")
        if not same_scene(left["initial_scene"], right["initial_scene"]):
            raise ValueError(f"Initial scenes differ for seed {left['reset_seed']}")
        pairs.append({"reset_seed": left["reset_seed"], "base": left["success"],
                      "ppo": right["success"], "scene_match": True})
    return {
        "status": "completed", "episodes_per_model": len(seeds), "pairs": pairs,
        "base_successes": sum(p["base"] for p in pairs),
        "ppo_successes": sum(p["ppo"] for p in pairs),
        "failure_to_success": sum(not p["base"] and p["ppo"] for p in pairs),
        "success_to_failure": sum(p["base"] and not p["ppo"] for p in pairs),
        "scope": "Development cohort; not a recovered original evaluation manifest",
    }


def policy_class(source: Path):
    """Import real policy modules while excluding distributed registration."""
    for name in ("rlinf", "rlinf.models", "rlinf.models.embodiment", "rlinf.models.embodiment.xvla"):
        module = types.ModuleType(name)
        module.__path__ = [str(source.joinpath(*name.split(".")))]
        sys.modules[name] = module
    from rlinf.models.embodiment.xvla.policy import XVLAPolicy

    return XVLAPolicy


def run_variant(args) -> None:
    """Run one model in a fresh process and retain per-episode evidence."""
    import imageio.v2 as imageio
    import numpy as np
    from omegaconf import OmegaConf
    import torch

    out = args.output / args.variant
    out.mkdir(exist_ok=False)
    state = {"status": "running", "variant": args.variant,
             "job_id": os.environ["SLURM_JOB_ID"], "episodes": []}
    env = None
    try:
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
        state.update(bind_allocated_gpu(torch))
        sys.path.insert(0, str(args.source))
        from tools.xvla.robotwin import job_runtime, protocol, runner
        from tools.xvla.robotwin.bind_renderer import bind_renderer

        if job_runtime.commit(args.source) != SOURCE_COMMIT:
            raise RuntimeError("Source differs from inspected policy/adapter revision")
        state["source_commit"] = SOURCE_COMMIT
        state["robotwin_commit"] = job_runtime.commit(args.robotwin)
        if state["robotwin_commit"] != protocol.ROBOTWIN_COMMIT:
            raise RuntimeError("RoboTwin source revision differs")
        state["versions"] = {p: importlib.metadata.version(p) for p in
                             ("torch", "transformers", "numpy", "sapien", "mplib", "nvidia-curobo", "setuptools")}
        state["model_base"] = str(args.base)
        state["ppo_weights"] = str(args.ppo / "full_weights.pt") if args.variant == "ppo" else None
        state["execution"] = "Standalone real policy and native env adapter; no distributed runner"
        state["video"] = "Initial observation plus each executed anchor; 10 fps playback, not real-time physics"
        write_json(out / "run.json", state)

        cfg = OmegaConf.load(args.ppo / "online-config.yaml")
        if cfg.env.eval.task_name != "move_stapler_pad" or cfg.env.eval.condition != "easy":
            raise ValueError("Expected the supplied move_stapler_pad/easy configuration")
        if cfg.env.eval.max_episode_steps != 300:
            raise ValueError("Expected a 300-anchor budget")
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
        model_cfg.model_path = str(args.base)
        env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
        env_cfg.robotwin_root = str(args.robotwin)
        env_cfg.reset_seeds = args.seeds
        env_cfg.reset_seed_mode = "cycle"
        OmegaConf.save(OmegaConf.create({"model": model_cfg, "env": env_cfg}), out / "resolved-config.yaml")
        torch.manual_seed(20261127)
        model = policy_class(args.source).from_checkpoint(model_cfg)
        if args.variant == "ppo":
            weights = torch.load(args.ppo / "full_weights.pt", map_location="cpu", weights_only=True, mmap=True)
            loaded = model.load_state_dict(weights, strict=True)
            state["strict_load"] = {"missing": loaded.missing_keys, "unexpected": loaded.unexpected_keys}
            del weights
        model = model.cuda().eval().requires_grad_(False)
        if any(not torch.isfinite(p).all().item() for p in model.state_dict().values() if p.is_floating_point()):
            raise RuntimeError("Model contains nonfinite weights")
        state["model_loaded"] = True
        write_json(out / "run.json", state)
        bind_renderer(pin_legacy_renderer=True)
        spec = importlib.util.spec_from_file_location(
            "fyp_native_robotwin", args.source / "rlinf/envs/sim/robotwin/xvla_env.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = module.XVLARoboTwinEnv(env_cfg, 1, 0, 1, None)

        for seed in args.seeds:
            started = time.perf_counter()
            row = {"status": "running", "variant": args.variant,
                   "reset_seed": seed, "policy_seed": 20261127 + seed}
            episode = out / f"seed-{seed}"
            episode.mkdir()
            writer = imageio.get_writer(episode / "head.mp4", fps=10, codec="libx264", quality=7)
            frames = 0
            task = env._tasks[0]
            original_observe = task.observe

            def observe():
                nonlocal frames
                raw = original_observe()
                images, _ = protocol.observation_values(raw)
                writer.append_data(images[0])
                frames += 1
                return raw

            task.observe = observe
            try:
                obs, reset_info = env.reset(env_seeds=[seed])
                if reset_info["reset_seeds"].tolist() != [seed]:
                    raise RuntimeError("Reset returned a different seed")
                row["initial_scene"] = task.scene_signature()
                if args.variant == "ppo":
                    reference = json.loads((args.output / "base" / f"seed-{seed}" / "episode.json").read_text())
                    if not runner.same_scene(reference["initial_scene"], row["initial_scene"]):
                        raise RuntimeError("Initial scene differs from the base model episode")
                noise_rng = torch.Generator(device="cuda").manual_seed(row["policy_seed"])
                action_chunks = []
                chunk_records = []
                for query in range(10):
                    noise = torch.randn(1, 30, 20, generator=noise_rng, device="cuda")
                    with torch.inference_mode():
                        actions, _ = model.predict_action_batch(obs, mode="eval", compute_values=False, initial_noise=noise)
                    action_chunks.append(actions[0].copy())
                    observations, rewards, terminations, truncations, infos = env.chunk_step(actions)
                    obs, info = observations[0], infos[0]
                    chunk_records.append({"query": query + 1,
                                          "executed_anchors": int(info["executed_mask"].sum()),
                                          "reward": float(rewards.sum())})
                    LOG.info("%s seed=%s query=%s steps=%s success=%s", args.variant, seed,
                             query + 1, env.elapsed_steps.tolist(), info["success"].tolist())
                    if bool((terminations[:, -1] | truncations[:, -1]).all()):
                        break
                else:
                    raise RuntimeError("300-anchor budget finished without a terminal/truncation flag")
                success = bool(info["success"][0])
                row.update(status="completed", success=success,
                           termination_reason="success" if success else ("step_limit" if bool(truncations[0, -1]) else "actor_pose_failure"),
                           metrics={key: value.tolist() for key, value in info["episode"].items()},
                           chunks=chunk_records, seconds=time.perf_counter() - started)
                np.savez_compressed(episode / "actions.npz", chunks=np.stack(action_chunks))
            except BaseException as error:
                row.update(status="error", error=repr(error), seconds=time.perf_counter() - started)
                raise
            finally:
                task.observe = original_observe
                writer.close()
                row["video_frames"] = frames
                write_json(episode / "episode.json", row)
                state["episodes"].append(row)
                write_json(out / "run.json", state)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="error", error=repr(error))
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except BaseException as error:
            state.update(status="error", cleanup_error=repr(error))
            raise
        finally:
            write_json(out / "run.json", state)


def main() -> None:
    """Run both variants in separate processes and validate paired results."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "robotwin", "base", "ppo", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2053, 2054])
    parser.add_argument("--variant", choices=("base", "ppo"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not os.environ.get("SLURM_JOB_ID") or os.uname().nodename == "4090node1":
        raise RuntimeError("Run inside a GPU Slurm allocation on a compute node")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(range(2053, 2063)):
        raise ValueError("Choose unique seeds from the supplied development evaluation pool 2053–2062")
    for name in ("source", "robotwin", "base", "ppo", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.variant:
        run_variant(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "reset_seeds": args.seeds, "job_id": os.environ["SLURM_JOB_ID"],
             "project_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip()}
    write_json(args.output / "summary.json", state)
    try:
        for variant in ("base", "ppo"):
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--variant", variant]
            with (args.output / f"{variant}.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode:
                tail = (args.output / f"{variant}.log").read_text(errors="replace").splitlines()[-80:]
                LOG.error("%s worker failed:\n%s", variant, "\n".join(tail))
                raise subprocess.CalledProcessError(process.returncode, command)
        sys.path.insert(0, str(args.source))
        from tools.xvla.robotwin.runner import same_scene

        rows = [json.loads((args.output / name / "run.json").read_text()) for name in ("base", "ppo")]
        if any(row["status"] != "completed" for row in rows):
            raise RuntimeError("One model process did not complete cleanly")
        state.update(summarize_pairs(rows[0]["episodes"], rows[1]["episodes"], args.seeds, same_scene))
        LOG.info("PAIRED_COMPARISON_COMPLETED %s", json.dumps(state))
    except BaseException as error:
        state.update(status="error", error=repr(error))
        raise
    finally:
        write_json(args.output / "summary.json", state)


if __name__ == "__main__":
    main()
