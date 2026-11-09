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

"""Exercise the official evaluator's configuration without loading a policy."""

import json
import sys
import time
from pathlib import Path

import pytest

from evaluations.fyp.evaluate import (
    MANIFEST,
    ROOT,
    build_command,
    check_assets,
    main,
    parse_args,
    physical_gpu,
    run_task,
)


def arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--model-path",
        str(tmp_path / "model"),
        "--evo-repo",
        str(tmp_path / "evo"),
        "--vlm-path",
        str(tmp_path / "vlm"),
        "--output-dir",
        str(tmp_path / "results"),
        *extra,
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--episodes", "5", "--num-envs", "2"],
        ["--episodes", "0"],
        ["--trial-start", "-1"],
        ["--video"],
        ["--tasks", "S1", "S1"],
        ["--tasks", "unknown"],
        ["--budget-minutes", "nan"],
        ["--budget-minutes", "1441"],
    ],
)
def test_invalid_budgets_fail_before_assets(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(arguments(tmp_path, *extra))


def test_dry_run_needs_no_assets_and_writes_nothing(tmp_path, caplog):
    caplog.set_level("INFO")
    assert main(arguments(tmp_path, "--dry-run")) == 0
    assert not list(tmp_path.iterdir())
    assert caplog.text.count("RESOLVE_") == 6


def test_commands_compose_with_official_hydra_config(tmp_path, monkeypatch):
    hydra = pytest.importorskip("hydra")
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    args = parse_args(
        arguments(
            tmp_path,
            "--gpu",
            "2",
            "--num-envs",
            "2",
            "--variant-root",
            str(tmp_path / "variants"),
        )
    )
    tasks = json.loads(MANIFEST.read_text())
    for task in tasks:
        command = build_command(args, {**task, "task_id": 3}, list(range(10, 20)))
        assert command[:4] == [
            "bash",
            str(ROOT / "evaluations/run_eval.sh"),
            "libero",
            "libero_spatial_evo1_eval",
        ]
        with hydra.initialize_config_dir(
            config_dir=str(ROOT / "evaluations/libero"), version_base=None
        ):
            cfg = hydra.compose(
                config_name="libero_spatial_evo1_eval", overrides=command[4:]
            )
        assert dict(cfg.cluster.component_placement) == {"env": 2, "rollout": 2}
        if task.get("scene_properties"):
            assert (
                dict(cfg.env.eval.init_params.scene_properties)
                == task["scene_properties"]
            )
        assert cfg.env.eval.task_suite_name == task["suite"]
        assert cfg.env.eval.task_id_filter == [3]
        assert cfg.env.eval.eval_trial_ids == list(range(10, 20))
        assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 10
        assert cfg.env.eval.max_episode_steps == 350
        assert cfg.env.eval.max_steps_per_rollout_epoch == 350
        assert cfg.env.eval.init_params.camera_heights == 448
        assert cfg.rollout.model.num_action_chunks == 14
        assert cfg.rollout.model.evo1.arm_key == "libero_robot"
        assert not cfg.env.eval.video_cfg.save_video
        assert cfg.runner.only_eval
        assert cfg.rollout.model.evo1.vlm_name == str(args.vlm_path)


def test_missing_checkpoint_fails_before_launch(tmp_path):
    args = parse_args(arguments(tmp_path))
    with pytest.raises(ValueError, match="Missing Evo1_LIBERO file"):
        check_assets(args)


def test_gpu_sampling_uses_physical_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert physical_gpu(0) == 3
    assert physical_gpu(1) == 1
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        physical_gpu(2)


def test_failed_child_preserves_log_and_return_code(tmp_path):
    code, elapsed, error = run_task(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('fixture failure'); sys.exit(7)",
        ],
        tmp_path,
        time.monotonic() + 30,
        0,
    )
    assert code == 7
    assert error is None
    assert elapsed > 0
    assert "fixture failure" in (tmp_path / "console.log").read_text()


def test_expired_budget_stops_only_own_child(tmp_path):
    code, _, error = run_task(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        time.monotonic() - 1,
        0,
    )
    assert code != 0
    assert error == "Evaluation time budget exhausted"


def test_variants_require_explicit_asset_root(tmp_path):
    args = parse_args(arguments(tmp_path, "--tasks", "B1B"))
    task = {"key": "B1B", "suite": "libero_goal", "task_id": 6, "variant": "B1B"}
    with pytest.raises(ValueError, match="variant-root"):
        build_command(args, task, [0])
    args.variant_root = tmp_path / "variants"
    command = build_command(args, task, [0])
    assert any("B1B/task.bddl" in item for item in command)
    assert any("B1B/states.pt" in item for item in command)


def test_variant_changes_preserve_unrelated_scene_objects():
    from evaluations.fyp.prepare_variants import variant_bddl

    source = "(:language Put the cream cheese on the bowl)\ncream_cheese_1 - cream_cheese\nplate_1 - plate\nwine_bottle_1 - wine_bottle\n(:goal (And (On cream_cheese_1 akita_black_bowl_1)))"
    changed = variant_bddl(source, "B1B")
    assert "butter_1 - butter" in changed
    assert "(On butter_1 akita_black_bowl_1)" in changed
    assert "wine_bottle_1 - wine_bottle" in changed
    changed = variant_bddl(source, "B2P")
    assert "(On cream_cheese_1 plate_1)" in changed
    assert "cream_cheese_1 - cream_cheese" in changed
    for key, target in [("B2N", "white_bowl"), ("B2R", "glazed_rim_porcelain_ramekin")]:
        changed = variant_bddl(source, key)
        assert f"(On cream_cheese_1 {target}_1)" in changed
        assert "wine_bottle_1 - wine_bottle" in changed
    with pytest.raises(ValueError):
        variant_bddl(source, "unknown")


def test_training_pilot_is_single_task_and_excludes_heldout_states(monkeypatch):
    hydra = pytest.importorskip("hydra")
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    with hydra.initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base=None
    ):
        cfg = hydra.compose(config_name="fyp_sg6_grpo_evo1")
    assert dict(cfg.cluster.component_placement) == {"actor": 0, "env,rollout": 1}
    assert cfg.env.train.task_suite_name == "libero_goal"
    assert cfg.env.train.task_id_filter == [6]
    assert cfg.env.train.train_trial_ids == list(range(20))
    assert cfg.env.eval.eval_trial_ids == list(range(20, 30))
    assert not set(cfg.env.train.train_trial_ids) & set(cfg.env.eval.eval_trial_ids)
    assert cfg.actor.model.evo1.rl_trainable_scope == "action_head"
    assert cfg.actor.micro_batch_size == 1
    assert cfg.env.train.total_num_envs == cfg.algorithm.group_size == 8
    assert cfg.env.train.max_episode_steps == 350
    assert cfg.runner.max_steps == 2


def test_export_requires_known_wrapper_prefix():
    from evaluations.fyp.export_evo1 import unwrap_weights

    assert unwrap_weights({"evo1.action_head.weight": 1}) == {"action_head.weight": 1}
    for invalid in [{}, {"action_head.weight": 1}, {"evo1.x": 1, "other.x": 2}]:
        with pytest.raises(ValueError, match="evo1"):
            unwrap_weights(invalid)


def test_layout_perturbations_stay_in_region_and_training_pool():
    import numpy as np

    from evaluations.fyp.prepare_layouts import REGIONS, SOURCE_TRIALS, sample_xy

    assert SOURCE_TRIALS == list(range(20))
    generator = np.random.default_rng(42)
    for bounds in REGIONS.values():
        original = np.array(bounds[:2]) + 0.001
        for _ in range(100):
            sampled = sample_xy(original, bounds, generator)
            assert np.all(sampled >= np.array(bounds[:2]))
            assert np.all(sampled <= np.array(bounds[2:]))
            assert np.all(np.abs(sampled - original) <= 0.01)
        with pytest.raises(ValueError, match="outside"):
            sample_xy([99, 99], bounds, generator)


def test_layout_sampling_accepts_settled_source_just_outside_region():
    import numpy as np

    from evaluations.fyp.prepare_layouts import REGIONS, sample_xy

    original = np.array([-0.03059484, 0.14390628])
    bounds = REGIONS["cream_cheese_1"]
    sampled = sample_xy(original, bounds, np.random.default_rng(42))
    assert np.all(sampled >= bounds[:2])
    assert np.all(sampled <= bounds[2:])
    assert np.all(np.abs(sampled - original) <= 0.01)


def test_training_audit_rejects_missing_budget_and_ineffective_updates():
    from evaluations.fyp.audit_training import validate_records

    tags = [
        "train/actor/grad_norm",
        "train/actor/approx_kl",
        "rollout/group_mixed_fraction",
        "rollout/advantages_max",
    ]
    metrics = {
        tag: [{"step": 0, "value": 1.0}, {"step": 1, "value": 1.0}] for tag in tags
    }
    validate_records(metrics, [2800, 5600], 2)
    with pytest.raises(ValueError, match="interaction"):
        validate_records(metrics, [2800], 2)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_records(metrics, [2800], 1)
    metrics[tags[0]][0]["value"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_records(metrics, [2800, 5600], 2)
    metrics[tags[0]] = [{"step": 0, "value": 0.0}, {"step": 1, "value": 0.0}]
    with pytest.raises(ValueError, match="nonzero"):
        validate_records(metrics, [2800, 5600], 2)


def test_pi05_download_rejects_outside_prefix():
    from evaluations.fyp.download_pi05_base import relative_name

    assert relative_name("checkpoints/pi05_base/params/_METADATA") == Path(
        "params/_METADATA"
    )
    for name in ("checkpoints/pi05_libero/model", "checkpoints/pi05_base/../model"):
        with pytest.raises(ValueError):
            relative_name(name)


def test_pi05_download_resumes_exact_generation(tmp_path, monkeypatch):
    import io

    from evaluations.fyp import download_pi05_base as downloader

    target = tmp_path / "params" / "weights"
    target.parent.mkdir()
    target.with_name("weights.part").write_bytes(b"abc")
    requests = []

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

    def open_request(request, timeout):
        requests.append(request)
        return Response(b"def")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_request)
    item = {
        "name": "checkpoints/pi05_base/params/weights",
        "size": "6",
        "generation": "123",
    }
    downloader.download(item, tmp_path)
    assert target.read_bytes() == b"abcdef"
    assert "generation=123" in requests[0].full_url
    assert requests[0].headers["Range"] == "bytes=3-"
    downloader.download(item, tmp_path)
    assert len(requests) == 1


def test_pi05_base_calibration_keeps_base_shape(tmp_path, monkeypatch):
    hydra = pytest.importorskip("hydra")
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    monkeypatch.setenv("FYP_STORAGE", str(tmp_path))
    with hydra.initialize_config_dir(
        config_dir=str(ROOT / "evaluations/libero"), version_base=None
    ):
        cfg = hydra.compose(config_name="libero_pi05_base_calibrated_eval")
    assert cfg.rollout.model.openpi.action_horizon == 50
    assert cfg.rollout.model.openpi.discrete_state_input is True
    assert cfg.rollout.model.openpi.task == "eval"
    assert cfg.rollout.model.num_action_chunks == 5
    assert cfg.rollout.model.num_steps == 10
    assert cfg.rollout.model.openpi_data.norm_stats_path.endswith(
        "pi05_libero_calibration/norm_stats.json"
    )
    assert list(cfg.env.eval.eval_trial_ids) == [0, 1]
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 2
