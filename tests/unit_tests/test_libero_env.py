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

"""LIBERO reset selection and episode accounting without a simulator."""

import importlib
import logging
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def libero_utils(monkeypatch):
    # Only the external benchmark import is needed to load the real helpers.
    benchmark = ModuleType("libero.libero.benchmark")
    benchmark.Benchmark = type("Benchmark", (), {})
    inner = ModuleType("libero.libero")
    inner.__path__ = []
    inner.benchmark = benchmark
    outer = ModuleType("libero")
    outer.__path__ = []
    outer.libero = inner
    monkeypatch.setenv("LIBERO_TYPE", "standard")
    for module in (outer, inner, benchmark):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    module_name = "rlinf.envs.sim.libero.utils"
    already_imported = module_name in sys.modules
    utils = importlib.import_module(module_name)
    yield utils
    if not already_imported:
        sys.modules.pop(module_name, None)
        package = sys.modules["rlinf.envs.sim.libero"]
        if getattr(package, "utils", None) is utils:
            delattr(package, "utils")


def test_trial_batches_select_disjoint_states_before_rank_distribution(libero_utils):
    bins = [50, 40, 30]
    batches = []
    for start in (0, 10):
        ids = libero_utils.build_eval_trial_reset_state_ids(
            bins, [2], OmegaConf.create(list(range(start, start + 10)))
        )
        np.testing.assert_array_equal(ids, np.arange(90 + start, 100 + start))
        distributed = libero_utils.distribute_reset_state_ids_round_robin(ids, 2)
        np.testing.assert_array_equal(distributed.T.flatten(), ids)
        batches.append(set(distributed.flatten()))
    assert not batches[0] & batches[1]
    assert len(batches[0] | batches[1]) == 20


def test_trial_selection_preserves_tasks_and_requested_trial_order(libero_utils):
    ids = libero_utils.build_eval_trial_reset_state_ids([5, 8, 3], [0, 2], [2, 0])
    np.testing.assert_array_equal(ids, [2, 0, 15, 13])


@pytest.mark.parametrize(
    "trial_ids",
    [[], [0, 0], [True], [np.bool_(False)], [-1], [1.0], ["1"], 1],
)
def test_invalid_trial_selections_are_rejected(libero_utils, trial_ids):
    with pytest.raises(ValueError, match="eval_trial_ids"):
        libero_utils.build_eval_trial_reset_state_ids([5], [0], trial_ids)


def test_trial_bounds_are_checked_for_every_selected_task(libero_utils):
    with pytest.raises(ValueError, match=r"task_id 1: expected \[0, 1\]"):
        libero_utils.build_eval_trial_reset_state_ids([5, 2], [0, 1], [2])
    np.testing.assert_array_equal(
        libero_utils.build_eval_trial_reset_state_ids([5, 2], [0], [2]), [2]
    )


def test_empty_task_selection_is_rejected(libero_utils):
    with pytest.raises(ValueError, match="at least one selected task"):
        libero_utils.build_eval_trial_reset_state_ids([5], [], [0])


def test_full_eval_pool_retains_interleaved_coverage(libero_utils):
    ids = libero_utils.build_interleaved_eval_reset_state_ids([3, 2], np.array([3, 5]))
    np.testing.assert_array_equal(ids, [0, 3, 1, 4, 2])
    np.testing.assert_array_equal(
        libero_utils.distribute_reset_state_ids_round_robin(ids, 2),
        [[0, 1, 2], [3, 4, -1]],
    )


@pytest.mark.parametrize("array_type", [np.asarray, torch.tensor])
def test_episode_logs_and_metrics_count_each_trial_once(
    libero_utils, caplog, array_type
):
    logger = logging.getLogger("test.libero.eval")
    caplog.set_level(logging.INFO, logger=logger.name)
    seen = set()
    stats = {}
    args = (
        np.array([0, 1]),
        {
            "episode": {
                "success_once": array_type([True, False]),
                "episode_len": array_type([350, 350]),
                "success_at_end": array_type([False, False]),
            }
        },
        np.array([2, 2]),
        np.array([10, 11]),
        2,
        seen,
        stats,
    )
    counted = libero_utils.record_completed_episode_task_stats(
        *args, logger=logger, elapsed_seconds=np.array([12.345678, 15.0])
    )
    np.testing.assert_array_equal(counted, [True, True])
    assert stats == {2: {"success": 1, "total": 2}}
    assert seen == {(2, 10), (2, 11)}
    assert caplog.messages == [
        "[libero eval] task_id=2, trial_id=10, success=True, "
        "episode_len=350, elapsed_seconds=12.345678, success_at_end=False",
        "[libero eval] task_id=2, trial_id=11, success=False, "
        "episode_len=350, elapsed_seconds=15.000000, success_at_end=False",
    ]
    caplog.clear()
    counted = libero_utils.record_completed_episode_task_stats(
        *args, logger=logger, elapsed_seconds=np.array([20.0, 20.0])
    )
    np.testing.assert_array_equal(counted, [False, False])
    assert stats == {2: {"success": 1, "total": 2}}
    assert len(caplog.messages) == 2
    assert all("duplicate episode skipped" in message for message in caplog.messages)


@pytest.mark.parametrize("elapsed_seconds", [None, np.array([np.nan])])
def test_optional_episode_fields_are_explicitly_unavailable(
    libero_utils, caplog, elapsed_seconds
):
    logger = logging.getLogger("test.libero.eval")
    caplog.set_level(logging.INFO, logger=logger.name)
    libero_utils.record_completed_episode_task_stats(
        np.array([0]),
        {"episode": {"success_once": np.array([False])}},
        np.array([0]),
        np.array([0]),
        1,
        set(),
        {},
        logger=logger,
        elapsed_seconds=elapsed_seconds,
    )
    assert caplog.messages == [
        "[libero eval] task_id=0, trial_id=0, success=False, "
        "episode_len=null, elapsed_seconds=null"
    ]
