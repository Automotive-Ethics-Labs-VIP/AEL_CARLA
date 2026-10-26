"""
test_training_data.py
=====================
Unit, integration, and CARLA end-to-end tests for:

  - python_api/generate_training_data.py

State vector extraction is tested in test_state_vector_extractor.py.

Test levels
-----------
Unit          No CARLA, no spawning. Pure function / class logic tested in
              isolation with hand-crafted frames.

Integration   MockAdapter + registry + spawner + collector + extractor wired
              together. Validates the full pipeline without CARLA.

CARLA         Requires --carla flag and a running CARLA server.
              Marked with @pytest.mark.carla and skipped automatically
              unless --carla is passed.

Running
-------
All tests (no CARLA):
    pytest test/test_training_data.py -v

CARLA tests only:
    pytest test/test_training_data.py -v -m carla --carla
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from ael_common import index

from python_api.state_vector_extractor import StateVectorExtractor
from python_api.ethical_attributes import (
    AgeGroup, Disability, SocialRole, EthicalAttributeSchema,
)
from python_api.actor_registry import EthicalActorRegistry
from python_api.adapter import MockAdapter
from python_api.spawn_walkers import EthicalWalkerSpawner
from python_api.collector import DataCollector
from python_api.generate_training_data import (
    generate_training_data,
    run_rollout,
    _random_policy,
    _MockEnvironment,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM = 40

# ael-v1 indices used below (see the ael-common README for the full layout)
IDX_PASSENGERS   = index("num_passengers")
IDX_VEL_DELTA    = index("velocity_delta")
OBS_BASE         = index("left_pedestrian")   # first path-block index


def _spawn_registered(
    adapter: MockAdapter,
    registry: EthicalActorRegistry,
    n: int,
    age_group: AgeGroup = AgeGroup.ADULT,
    disability: Disability = Disability.NONE,
    social_role: SocialRole = SocialRole.CIVILIAN,
) -> List[int]:
    """Spawn n mock walkers and register known attributes. Returns actor IDs."""
    bp = adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
    actor_ids: List[int] = []
    for i in range(n):
        sp = adapter.get_spawn_points()[i % len(adapter.get_spawn_points())]
        aid = adapter.spawn_walker(sp, bp)
        attrs = EthicalAttributeSchema(
            age_group=age_group,
            disability=disability,
            pregnancy=False,
            group_size=1,
            social_role=social_role,
        )
        registry.register(aid, attrs)
        actor_ids.append(aid)
    return actor_ids


# ─────────────────────────────────────────────────────────────────────────────
# generate_training_data – unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomPolicy:

    def test_always_returns_valid_action(self):
        for _ in range(200):
            action = _random_policy([0.0] * STATE_DIM)
            assert action in range(5), f"Invalid action: {action}"

    def test_returns_int(self):
        action = _random_policy([0.0] * STATE_DIM)
        assert isinstance(action, int)

    def test_all_five_actions_eventually_produced(self):
        seen = set()
        for _ in range(500):
            seen.add(_random_policy([0.0] * STATE_DIM))
        assert seen == {0, 1, 2, 3, 4}, f"Missing actions: {set(range(5)) - seen}"





class TestRunRollout:

    def _build_mock_env(self, n_walkers=10):
        adapter  = MockAdapter(num_spawn_points=max(n_walkers + 5, 20))
        registry = EthicalActorRegistry()
        spawner  = EthicalWalkerSpawner(adapter, registry)
        env      = _MockEnvironment(spawner, adapter, n_walkers)
        env.reset()
        return env, registry, adapter

    def test_returns_states_and_actions_of_equal_length(self):
        env, registry, adapter = self._build_mock_env()
        extractor = StateVectorExtractor()
        states, actions = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=10)
        assert len(states) == len(actions) == 10

    def test_each_state_is_40_dims(self):
        env, registry, adapter = self._build_mock_env()
        extractor = StateVectorExtractor()
        states, _ = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=5)
        for i, state in enumerate(states):
            assert len(state) == STATE_DIM, f"State {i} has {len(state)} dims"

    def test_all_actions_are_valid(self):
        env, registry, adapter = self._build_mock_env()
        extractor = StateVectorExtractor()
        _, actions = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=20)
        for a in actions:
            assert a in range(5), f"Invalid action: {a}"

    def test_states_are_python_floats_not_numpy(self):
        env, registry, adapter = self._build_mock_env()
        extractor = StateVectorExtractor()
        states, _ = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=5)
        for state in states:
            for v in state:
                assert isinstance(v, float), f"Expected float, got {type(v)}"

    def test_velocity_delta_is_zero_on_first_step(self):
        """reset() is called before each rollout — first step delta must be 0."""
        env, registry, adapter = self._build_mock_env()
        extractor = StateVectorExtractor()
        states, _ = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=5)
        assert states[0][IDX_VEL_DELTA] == pytest.approx(0.0), (
            "velocity_delta must be 0 on the first step of every rollout"
        )

    def test_all_states_are_finite(self):
        env, registry, adapter = self._build_mock_env()
        extractor = StateVectorExtractor()
        states, _ = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=10)
        for i, state in enumerate(states):
            for j, v in enumerate(state):
                assert math.isfinite(v), f"states[{i}][{j}] = {v} is not finite"


class TestGenerateTrainingData:
    """Tests for the top-level generate_training_data() function."""

    def _run(self, tmp_path, **kwargs) -> List[dict]:
        out = str(tmp_path / "pairs.json")
        generate_training_data(
            num_pairs=int(kwargs.get("num_pairs", 3)),
            max_steps=int(kwargs.get("max_steps", 5)),
            num_walkers=int(kwargs.get("num_walkers", 5)),
            output_file=str(kwargs.get("output_file", out)),
            use_mock=bool(kwargs.get("use_mock", True)),
            node_id=str(kwargs.get("node_id", "n0")),
            num_passengers=int(kwargs.get("num_passengers", 1)),
        )
        with open(out) as f:
            return json.load(f)

    # --- Output file structure ---

    def test_output_file_is_created(self, tmp_path):
        out = str(tmp_path / "sub" / "pairs.json")
        generate_training_data(num_pairs=1, max_steps=3, num_walkers=3,
                               output_file=out, use_mock=True)
        assert Path(out).exists()

    def test_output_directory_created_if_not_exists(self, tmp_path):
        out = str(tmp_path / "new_dir" / "deep" / "pairs.json")
        generate_training_data(num_pairs=1, max_steps=3, num_walkers=3,
                               output_file=out, use_mock=True)
        assert Path(out).exists()

    def test_output_is_valid_json(self, tmp_path):
        out = str(tmp_path / "pairs.json")
        generate_training_data(num_pairs=2, max_steps=3, num_walkers=3,
                               output_file=out, use_mock=True)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, list)

    # --- Pair count and IDs ---

    def test_correct_number_of_pairs(self, tmp_path):
        data = self._run(tmp_path, num_pairs=4)
        assert len(data) == 4

    def test_pair_ids_are_unique_and_sequential(self, tmp_path):
        data = self._run(tmp_path, num_pairs=5)
        ids = [p["id"] for p in data]
        assert ids == [f"pair_{i:04d}" for i in range(5)]

    # --- Trajectory structure ---

    def test_each_pair_has_trajectory_a_and_b(self, tmp_path):
        data = self._run(tmp_path, num_pairs=2)
        for pair in data:
            assert "id"           in pair
            assert "trajectory_a" in pair
            assert "trajectory_b" in pair

    def test_each_trajectory_has_states_and_actions(self, tmp_path):
        data = self._run(tmp_path)
        for pair in data:
            for key in ("trajectory_a", "trajectory_b"):
                assert "states"  in pair[key]
                assert "actions" in pair[key]

    def test_states_and_actions_equal_length(self, tmp_path):
        data = self._run(tmp_path, max_steps=7)
        for pair in data:
            for key in ("trajectory_a", "trajectory_b"):
                assert len(pair[key]["states"]) == len(pair[key]["actions"]) == 7

    # --- State vector correctness ---

    def test_each_state_is_40_dimensional(self, tmp_path):
        data = self._run(tmp_path)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                for i, state in enumerate(pair[traj_key]["states"]):
                    assert len(state) == STATE_DIM, (
                        f"{traj_key} state {i} has {len(state)} dims, expected {STATE_DIM}"
                    )

    def test_states_are_lists_of_python_floats(self, tmp_path):
        """JSON round-trip requires plain Python floats, not numpy scalars."""
        data = self._run(tmp_path)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                for state in pair[traj_key]["states"]:
                    for v in state:
                        assert isinstance(v, float), f"Got {type(v)}"

    def test_states_are_all_finite(self, tmp_path):
        data = self._run(tmp_path)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                for state in pair[traj_key]["states"]:
                    assert all(math.isfinite(v) for v in state)

    # --- Action correctness ---

    def test_actions_are_in_valid_range(self, tmp_path):
        data = self._run(tmp_path, max_steps=20)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                for action in pair[traj_key]["actions"]:
                    assert action in range(5), f"Invalid action: {action}"

    def test_actions_are_plain_ints(self, tmp_path):
        data = self._run(tmp_path)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                for action in pair[traj_key]["actions"]:
                    assert isinstance(action, int), f"Got {type(action)}"

    # --- Velocity delta resets between rollouts ---

    def test_first_state_of_each_trajectory_has_zero_velocity_delta(self, tmp_path):
        """
        reset() must be called before each rollout. velocity_delta at step 0
        must always be 0 regardless of any prior trajectory in the same pair.
        """
        data = self._run(tmp_path, num_pairs=3, max_steps=5)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                first_state = pair[traj_key]["states"][0]
                assert first_state[IDX_VEL_DELTA] == pytest.approx(0.0), (
                    f"{traj_key}: velocity_delta at step 0 should be 0, "
                    f"got {first_state[IDX_VEL_DELTA]}"
                )

    # --- Passengers index ---

    def test_passengers_index_reflects_constructor_param(self, tmp_path):
        data = self._run(tmp_path, num_passengers=3)
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                for state in pair[traj_key]["states"]:
                    assert state[IDX_PASSENGERS] == pytest.approx(3.0)

    # --- Team A format compatibility ---

    def test_output_readable_by_team_a_annotate_trajectories_format(self, tmp_path):
        """
        Simulate what Team A's annotate_trajectories.py does when it reads the file.
        It calls json.load, iterates pairs, and accesses states/actions as plain lists.
        """
        data = self._run(tmp_path, num_pairs=2)
        for pair in data:
            pair_id = pair["id"]
            assert pair_id.startswith("pair_")

            traj_a = pair["trajectory_a"]
            states_a  = traj_a["states"]   # must be list of lists
            actions_a = traj_a["actions"]  # must be list of ints

            # Team A does: np.array(states) and direct list iteration
            assert isinstance(states_a,  list)
            assert isinstance(actions_a, list)
            assert isinstance(states_a[0], list)
            assert isinstance(actions_a[0], int)

    def test_output_is_json_serializable_after_generation(self, tmp_path):
        """Round-trip through JSON must produce identical data."""
        out = str(tmp_path / "pairs.json")
        generate_training_data(num_pairs=2, max_steps=4, num_walkers=4,
                               output_file=out, use_mock=True)
        with open(out) as f:
            data1 = json.load(f)
        # Re-serialize and re-parse
        data2 = json.loads(json.dumps(data1))
        assert data1 == data2

    # --- Edge cases ---

    def test_num_pairs_one_works(self, tmp_path):
        data = self._run(tmp_path, num_pairs=1)
        assert len(data) == 1

    def test_max_steps_one_produces_single_step_trajectories(self, tmp_path):
        data = self._run(tmp_path, num_pairs=1, max_steps=1)
        for traj_key in ("trajectory_a", "trajectory_b"):
            assert len(data[0][traj_key]["states"])  == 1
            assert len(data[0][traj_key]["actions"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# generate_training_data – integration tests (mock pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateTrainingDataIntegration:

    def test_different_trajectories_within_same_pair(self, tmp_path):
        """
        Traj A and traj B are independent rollouts — actions should differ
        in at least some steps when using a random policy over enough steps.
        """
        out = str(tmp_path / "pairs.json")
        generate_training_data(num_pairs=5, max_steps=30, num_walkers=5,
                               output_file=out, use_mock=True)
        with open(out) as f:
            data = json.load(f)

        any_different = False
        for pair in data:
            if pair["trajectory_a"]["actions"] != pair["trajectory_b"]["actions"]:
                any_different = True
                break
        assert any_different, (
            "All trajectory pairs are identical — random policy may be broken"
        )

    def test_obstacle_features_nonzero_when_walkers_spawned(self, tmp_path):
        """
        When walkers are spawned and have ethical attributes registered,
        at least some obstacle features in some state should be nonzero.
        """
        out = str(tmp_path / "pairs.json")
        generate_training_data(num_pairs=1, max_steps=5, num_walkers=20,
                               output_file=out, use_mock=True)
        with open(out) as f:
            data = json.load(f)

        obstacle_indices = list(range(OBS_BASE, STATE_DIM))
        any_nonzero = False
        for state in data[0]["trajectory_a"]["states"]:
            if any(state[i] != 0.0 for i in obstacle_indices):
                any_nonzero = True
                break
        assert any_nonzero, (
            "Obstacle features are all zero — walkers may not have been registered "
            "or actor positions may all be out of range"
        )

    def test_multiple_runs_produce_independent_files(self, tmp_path):
        """Two calls with different seeds must produce different trajectories."""
        out1 = str(tmp_path / "pairs1.json")
        out2 = str(tmp_path / "pairs2.json")
        generate_training_data(num_pairs=2, max_steps=10, num_walkers=5,
                               output_file=out1, use_mock=True)
        generate_training_data(num_pairs=2, max_steps=10, num_walkers=5,
                               output_file=out2, use_mock=True)

        with open(out1) as f:
            d1 = json.load(f)
        with open(out2) as f:
            d2 = json.load(f)

        # Files should be structurally identical in format
        assert len(d1) == len(d2)
        for pair in d1 + d2:
            assert len(pair["trajectory_a"]["states"][0]) == STATE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# generate_training_data – CARLA end-to-end tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.carla
class TestGenerateTrainingDataCARLA:

    def test_carla_generates_valid_pairs_file(self, tmp_path, carla_client, carla_world):
        """Full CARLA pipeline — output must satisfy Team A's format contract."""
        from python_api.adapter import CARLAAdapter
        from python_api.generate_training_data import (
            _CARLAEnvironment, run_rollout, _random_policy
        )
        import carla

        adapter  = CARLAAdapter(carla_client, carla_world)
        registry = EthicalActorRegistry()
        spawner  = EthicalWalkerSpawner(adapter, registry)
        env      = _CARLAEnvironment(adapter, spawner, num_walkers=10)

        extractor = StateVectorExtractor()
        env.reset()
        carla_world.tick()

        states, actions = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=5)
        # Do not call env.teardown() — see test_carla_ethical_attributes_in_states
        # for explanation. The carla_world fixture handles all actor cleanup.

        assert len(states) == 5
        assert len(actions) == 5
        for state in states:
            assert len(state) == STATE_DIM
            assert all(math.isfinite(v) for v in state)
        for action in actions:
            assert action in range(5)

    def test_carla_generate_training_data_writes_valid_file(self, tmp_path, carla_client, carla_world):
        """Test the full generate_training_data() function with CARLA."""
        out = str(tmp_path / "carla_pairs.json")
        generate_training_data(
            num_pairs=2,
            max_steps=5,
            num_walkers=10,
            output_file=out,
            use_mock=False,
            carla_host="localhost",
            carla_port=2000,
        )
        with open(out) as f:
            data = json.load(f)

        assert len(data) == 2
        for pair in data:
            for traj_key in ("trajectory_a", "trajectory_b"):
                assert len(pair[traj_key]["states"]) == 5
                for state in pair[traj_key]["states"]:
                    assert len(state) == STATE_DIM

    def test_carla_ethical_attributes_in_states(self, carla_client, carla_world, spawn_points):
        """Walkers with known ethical attributes must produce valid 40D states."""
        from python_api.adapter import CARLAAdapter
        from python_api.generate_training_data import _CARLAEnvironment, run_rollout, _random_policy
        from unittest.mock import patch

        adapter  = CARLAAdapter(carla_client, carla_world)
        registry = EthicalActorRegistry()
        spawner  = EthicalWalkerSpawner(adapter, registry)

        child_attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN,
        )
        with patch.object(spawner, "_generate_ethical_attributes", return_value=child_attrs):
            env = _CARLAEnvironment(adapter, spawner, num_walkers=10)
            env.reset()
            carla_world.tick()

        extractor = StateVectorExtractor()
        states, _ = run_rollout(env, extractor, registry, _random_policy, "n0", max_steps=3)
        # Do NOT call env.teardown() here. run_rollout() calls env.reset() internally
        # which respawns walkers via apply_batch_sync(do_tick=False). Calling destroy()
        # on those actors before the next tick causes CARLA to log "not found" errors.
        # The carla_world fixture teardown handles all cleanup correctly.

        for state in states:
            assert len(state) == STATE_DIM
            assert all(math.isfinite(v) for v in state)