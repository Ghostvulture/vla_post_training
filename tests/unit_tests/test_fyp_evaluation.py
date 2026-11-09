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
