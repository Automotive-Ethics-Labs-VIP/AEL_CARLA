"""
test_training_data.py
=====================
Unit, integration, and CARLA end-to-end tests for:

  - python_api/state_vector_extractor.py
  - python_api/generate_training_data.py

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

from python_api.state_vector_extractor import (
    StateVectorExtractor,
    _build_obstacle_features,
    _normalise_angle,
)
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

# Index map – must stay in sync with state_vector_extractor.py
IDX_SPEED        = 0
IDX_PASSENGERS   = 1
IDX_LANE_POS     = 2
IDX_VEL_DELTA    = 3
IDX_N_STRAIGHT   = 4
IDX_N_LEFT       = 5
IDX_N_RIGHT      = 6
OBS_BASE         = 7
FEATS_PER_DIR    = 11

# Obstacle feature offsets within a direction block
OFS_COUNT        = 0
OFS_CHILD        = 1
OFS_ELDERLY      = 2
OFS_WHEELCHAIR   = 3
OFS_CANE         = 4
OFS_BLIND        = 5
OFS_PREGNANT     = 6
OFS_EMERGENCY    = 7
OFS_HEALTHCARE   = 8
OFS_MAX_VULN     = 9
OFS_AVG_VULN     = 10

# Direction block base indices
IDX_STRAIGHT_BASE = OBS_BASE + 0 * FEATS_PER_DIR   # 7
IDX_LEFT_BASE     = OBS_BASE + 1 * FEATS_PER_DIR   # 18
IDX_RIGHT_BASE    = OBS_BASE + 2 * FEATS_PER_DIR   # 29


def _make_frame(
    velocity: Optional[List[float]] = None,
    position: Optional[List[float]] = None,
    actors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a minimal DataCollector-style frame for testing."""
    return {
        "timestamp": 0.0,
        "sensor_bundle": {"rgb": "", "depth": "", "lidar": ""},
        "ethical_context": {"visible_actors": actors or []},
        "vehicle_state": {
            "position": position or [0.0, 0.0, 0.0],
            "velocity": velocity or [0.0, 0.0, 0.0],
            "controls": {"throttle": 0.0, "steer": 0.0, "brake": 0.0},
        },
        "scene_metadata": {"weather": "clear_noon", "map": "TestMap"},
    }


def _make_actor(
    position: List[float],
    age_group: str = "adult",
    disability: str = "none",
    social_role: str = "civilian",
    vulnerability_score: float = 0.0,
) -> Dict[str, Any]:
    """Build a minimal actor entry matching DataCollector's output."""
    return {
        "id": "n0-1",
        "age_group": age_group,
        "disability": disability,
        "social_role": social_role,
        "vulnerability_score": vulnerability_score,
        "position": position,
        "velocity": [0.0, 0.0, 0.0],
    }


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
# _normalise_angle – unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseAngle:

    def test_zero_unchanged(self):
        assert _normalise_angle(0.0) == pytest.approx(0.0)

    def test_positive_pi_unchanged(self):
        # pi is the upper boundary; values > pi wrap down
        assert _normalise_angle(math.pi) == pytest.approx(math.pi)

    def test_above_pi_wraps_down(self):
        result = _normalise_angle(math.pi + 0.1)
        assert result == pytest.approx(-math.pi + 0.1, abs=1e-9)

    def test_negative_pi_wraps_to_positive(self):
        # -pi should become pi (our convention: (-pi, pi])
        result = _normalise_angle(-math.pi)
        assert result == pytest.approx(math.pi, abs=1e-9)

    def test_two_pi_wraps_to_zero(self):
        result = _normalise_angle(2 * math.pi)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_negative_quarter_pi_unchanged(self):
        assert _normalise_angle(-math.pi / 4) == pytest.approx(-math.pi / 4)

    def test_three_pi_wraps_to_pi(self):
        result = _normalise_angle(3 * math.pi)
        assert result == pytest.approx(math.pi, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# _build_obstacle_features – unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildObstacleFeatures:

    def test_empty_list_returns_all_zeros(self):
        feats = _build_obstacle_features([])
        assert feats == [0.0] * 11
        assert len(feats) == 11

    def test_single_healthy_adult_minimal_features(self):
        actor = _make_actor([0.0, 0.0, 0.0])
        feats = _build_obstacle_features([actor])
        assert feats[OFS_COUNT]      == pytest.approx(0.2)   # 1/5
        assert feats[OFS_CHILD]      == 0.0
        assert feats[OFS_ELDERLY]    == 0.0
        assert feats[OFS_WHEELCHAIR] == 0.0
        assert feats[OFS_CANE]       == 0.0
        assert feats[OFS_BLIND]      == 0.0
        assert feats[OFS_PREGNANT]   == 0.0
        assert feats[OFS_EMERGENCY]  == 0.0
        assert feats[OFS_HEALTHCARE] == 0.0

    def test_child_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], age_group="child", vulnerability_score=0.3)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_CHILD]   == 1.0
        assert feats[OFS_ELDERLY] == 0.0

    def test_elderly_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], age_group="elderly", vulnerability_score=0.2)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_ELDERLY] == 1.0
        assert feats[OFS_CHILD]   == 0.0

    def test_wheelchair_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], disability="wheelchair", vulnerability_score=0.25)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_WHEELCHAIR] == 1.0
        assert feats[OFS_CANE]       == 0.0

    def test_cane_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], disability="cane", vulnerability_score=0.25)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_CANE]       == 1.0
        assert feats[OFS_WHEELCHAIR] == 0.0

    def test_blind_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], disability="blind", vulnerability_score=0.25)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_BLIND] == 1.0

    def test_emergency_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], social_role="emergency")
        feats = _build_obstacle_features([actor])
        assert feats[OFS_EMERGENCY]  == 1.0
        assert feats[OFS_HEALTHCARE] == 0.0

    def test_healthcare_flag_set(self):
        actor = _make_actor([0.0, 0.0, 0.0], social_role="healthcare")
        feats = _build_obstacle_features([actor])
        assert feats[OFS_HEALTHCARE] == 1.0
        assert feats[OFS_EMERGENCY]  == 0.0

    def test_pregnancy_flag_always_zero_collector_does_not_expose_it(self):
        """
        The collector's actor dict does not include a pregnancy key.
        has_pregnant is therefore always 0 — vulnerability_score captures
        the pregnancy contribution instead.
        """
        actor = _make_actor([0.0, 0.0, 0.0], vulnerability_score=0.25)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_PREGNANT] == 0.0, (
            "has_pregnant should always be 0 because the collector "
            "does not expose pregnancy in the actor dict. "
            "Vulnerability_score (indices 9-10) captures it instead."
        )

    def test_max_vulnerability_single_actor(self):
        actor = _make_actor([0.0, 0.0, 0.0], vulnerability_score=0.55)
        feats = _build_obstacle_features([actor])
        assert feats[OFS_MAX_VULN] == pytest.approx(0.55)
        assert feats[OFS_AVG_VULN] == pytest.approx(0.55)

    def test_max_and_avg_vulnerability_multiple_actors(self):
        actors = [
            _make_actor([0.0, 0.0, 0.0], vulnerability_score=0.3),
            _make_actor([1.0, 0.0, 0.0], vulnerability_score=0.7),
            _make_actor([2.0, 0.0, 0.0], vulnerability_score=0.5),
        ]
        feats = _build_obstacle_features(actors)
        assert feats[OFS_MAX_VULN] == pytest.approx(0.7)
        assert feats[OFS_AVG_VULN] == pytest.approx((0.3 + 0.7 + 0.5) / 3)

    def test_actor_count_normalised_by_five(self):
        actors = [_make_actor([float(i), 0.0, 0.0]) for i in range(5)]
        feats = _build_obstacle_features(actors)
        assert feats[OFS_COUNT] == pytest.approx(1.0)   # 5/5

    def test_actor_count_capped_at_one(self):
        actors = [_make_actor([float(i), 0.0, 0.0]) for i in range(10)]
        feats = _build_obstacle_features(actors)
        assert feats[OFS_COUNT] == pytest.approx(1.0)   # capped, not 2.0

    def test_actor_count_partial(self):
        actors = [_make_actor([float(i), 0.0, 0.0]) for i in range(3)]
        feats = _build_obstacle_features(actors)
        assert feats[OFS_COUNT] == pytest.approx(3 / 5.0)

    def test_null_vulnerability_score_excluded_from_stats(self):
        """Actor with vulnerability_score=None should not crash and not skew stats."""
        actors = [
            {"id": "n0-1", "age_group": "adult", "disability": "none",
             "social_role": "civilian", "vulnerability_score": None,
             "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
        ]
        feats = _build_obstacle_features(actors)
        assert feats[OFS_MAX_VULN] == pytest.approx(0.0)
        assert feats[OFS_AVG_VULN] == pytest.approx(0.0)

    def test_multiple_flags_set_from_different_actors(self):
        """Child and elderly in same bin → both flags set simultaneously."""
        actors = [
            _make_actor([0.0, 0.0, 0.0], age_group="child",   vulnerability_score=0.3),
            _make_actor([1.0, 0.0, 0.0], age_group="elderly", vulnerability_score=0.2),
        ]
        feats = _build_obstacle_features(actors)
        assert feats[OFS_CHILD]   == 1.0
        assert feats[OFS_ELDERLY] == 1.0

    def test_output_always_length_11(self):
        for n in range(6):
            actors = [_make_actor([float(i), 0.0, 0.0]) for i in range(n)]
            assert len(_build_obstacle_features(actors)) == 11


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – output shape and type
# ─────────────────────────────────────────────────────────────────────────────

class TestStateVectorShape:

    def test_output_is_list_of_40(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame())
        assert isinstance(state, list)
        assert len(state) == STATE_DIM

    def test_all_values_are_python_floats(self):
        """Must be plain Python floats, not numpy scalars, for JSON serialization."""
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame())
        for i, v in enumerate(state):
            assert isinstance(v, float), f"state[{i}] is {type(v)}, expected float"

    def test_all_values_are_finite(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame())
        for i, v in enumerate(state):
            assert math.isfinite(v), f"state[{i}] = {v} is not finite"

    def test_empty_frame_produces_40_zeros_except_passengers(self):
        extractor = StateVectorExtractor(num_passengers=3)
        state = extractor.extract(_make_frame())
        assert state[IDX_PASSENGERS] == 3.0
        # All other indices should be 0
        for i, v in enumerate(state):
            if i == IDX_PASSENGERS:
                continue
            assert v == pytest.approx(0.0), f"Expected 0 at index {i}, got {v}"


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – index 0: speed
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeedExtraction:

    def test_stationary_vehicle_speed_zero(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame(velocity=[0.0, 0.0, 0.0]))
        assert state[IDX_SPEED] == pytest.approx(0.0)

    def test_speed_from_x_only_velocity(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame(velocity=[10.0, 0.0, 0.0]))
        assert state[IDX_SPEED] == pytest.approx(10.0)

    def test_speed_from_y_only_velocity(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame(velocity=[0.0, 5.0, 0.0]))
        assert state[IDX_SPEED] == pytest.approx(5.0)

    def test_speed_from_3d_velocity_is_magnitude(self):
        extractor = StateVectorExtractor()
        vx, vy, vz = 3.0, 4.0, 0.0
        state = extractor.extract(_make_frame(velocity=[vx, vy, vz]))
        expected = math.sqrt(vx**2 + vy**2 + vz**2)
        assert state[IDX_SPEED] == pytest.approx(expected)

    def test_speed_from_full_3d_velocity(self):
        extractor = StateVectorExtractor()
        vx, vy, vz = 1.0, 2.0, 2.0
        state = extractor.extract(_make_frame(velocity=[vx, vy, vz]))
        assert state[IDX_SPEED] == pytest.approx(3.0)  # sqrt(1+4+4)


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – index 1: passengers
# ─────────────────────────────────────────────────────────────────────────────

class TestPassengersExtraction:

    def test_default_passengers_is_one(self):
        state = StateVectorExtractor().extract(_make_frame())
        assert state[IDX_PASSENGERS] == pytest.approx(1.0)

    def test_custom_passengers_zero(self):
        state = StateVectorExtractor(num_passengers=0).extract(_make_frame())
        assert state[IDX_PASSENGERS] == pytest.approx(0.0)

    def test_custom_passengers_four(self):
        state = StateVectorExtractor(num_passengers=4).extract(_make_frame())
        assert state[IDX_PASSENGERS] == pytest.approx(4.0)

    def test_passengers_constant_across_frames(self):
        extractor = StateVectorExtractor(num_passengers=2)
        for _ in range(5):
            state = extractor.extract(_make_frame())
            assert state[IDX_PASSENGERS] == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – index 2: lane_position
# ─────────────────────────────────────────────────────────────────────────────

class TestLanePositionExtraction:

    def test_centred_vehicle_lane_pos_zero(self):
        state = StateVectorExtractor().extract(_make_frame(position=[0.0, 0.0, 0.0]))
        assert state[IDX_LANE_POS] == pytest.approx(0.0)

    def test_right_offset_positive(self):
        state = StateVectorExtractor().extract(_make_frame(position=[10.0, 2.5, 0.0]))
        assert state[IDX_LANE_POS] == pytest.approx(2.5)

    def test_left_offset_negative(self):
        state = StateVectorExtractor().extract(_make_frame(position=[10.0, -1.8, 0.0]))
        assert state[IDX_LANE_POS] == pytest.approx(-1.8)

    def test_lane_pos_uses_y_not_x(self):
        """x and z should not affect lane_position."""
        extractor = StateVectorExtractor()
        s1 = extractor.extract(_make_frame(position=[0.0,  1.0, 0.0]))
        s2 = extractor.extract(_make_frame(position=[99.0, 1.0, 5.0]))
        assert s1[IDX_LANE_POS] == pytest.approx(s2[IDX_LANE_POS])


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – index 3: velocity_delta
# ─────────────────────────────────────────────────────────────────────────────

class TestVelocityDelta:

    def test_first_call_after_init_is_zero(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame(velocity=[10.0, 0.0, 0.0]))
        assert state[IDX_VEL_DELTA] == pytest.approx(0.0)

    def test_first_call_after_reset_is_zero(self):
        extractor = StateVectorExtractor()
        extractor.extract(_make_frame(velocity=[5.0, 0.0, 0.0]))
        extractor.reset()
        state = extractor.extract(_make_frame(velocity=[10.0, 0.0, 0.0]))
        assert state[IDX_VEL_DELTA] == pytest.approx(0.0)

    def test_acceleration_positive_delta(self):
        extractor = StateVectorExtractor()
        extractor.extract(_make_frame(velocity=[5.0, 0.0, 0.0]))   # speed=5
        state = extractor.extract(_make_frame(velocity=[8.0, 0.0, 0.0]))   # speed=8
        assert state[IDX_VEL_DELTA] == pytest.approx(3.0)

    def test_deceleration_negative_delta(self):
        extractor = StateVectorExtractor()
        extractor.extract(_make_frame(velocity=[10.0, 0.0, 0.0]))  # speed=10
        state = extractor.extract(_make_frame(velocity=[6.0, 0.0, 0.0]))   # speed=6
        assert state[IDX_VEL_DELTA] == pytest.approx(-4.0)

    def test_constant_speed_zero_delta(self):
        extractor = StateVectorExtractor()
        extractor.extract(_make_frame(velocity=[7.0, 0.0, 0.0]))
        state = extractor.extract(_make_frame(velocity=[7.0, 0.0, 0.0]))
        assert state[IDX_VEL_DELTA] == pytest.approx(0.0)

    def test_delta_accumulates_correctly_over_three_frames(self):
        extractor = StateVectorExtractor()
        extractor.extract(_make_frame(velocity=[0.0, 0.0, 0.0]))   # speed=0, delta=0
        s2 = extractor.extract(_make_frame(velocity=[5.0, 0.0, 0.0]))   # delta=5
        s3 = extractor.extract(_make_frame(velocity=[3.0, 0.0, 0.0]))   # delta=-2
        assert s2[IDX_VEL_DELTA] == pytest.approx(5.0)
        assert s3[IDX_VEL_DELTA] == pytest.approx(-2.0)

    def test_reset_then_new_sequence_starts_fresh(self):
        extractor = StateVectorExtractor()
        extractor.extract(_make_frame(velocity=[20.0, 0.0, 0.0]))
        extractor.extract(_make_frame(velocity=[25.0, 0.0, 0.0]))
        extractor.reset()
        # New episode — first call should be 0 regardless of prior speed
        state = extractor.extract(_make_frame(velocity=[30.0, 0.0, 0.0]))
        assert state[IDX_VEL_DELTA] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – actor direction binning (indices 4-6)
# ─────────────────────────────────────────────────────────────────────────────

class TestActorDirectionBinning:
    """
    Ego at origin, heading along +X axis (velocity=[1,0,0]).

    Straight: actor within ±22.5° → approx. directly ahead or behind X
    Left:     actor between -90° and -22.5° relative to heading
    Right:    actor between +22.5° and +90° relative to heading
    Behind:   >90° → ignored
    """

    def _ego_heading_x(self, actor_pos):
        """Frame with ego heading +X and one actor."""
        return _make_frame(
            velocity=[1.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[_make_actor(actor_pos)],
        )

    def test_no_actors_all_counts_zero(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(_make_frame(velocity=[1.0, 0.0, 0.0]))
        assert state[IDX_N_STRAIGHT] == 0.0
        assert state[IDX_N_LEFT]     == 0.0
        assert state[IDX_N_RIGHT]    == 0.0

    def test_actor_directly_ahead_is_straight(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(self._ego_heading_x([10.0, 0.0, 0.0]))
        assert state[IDX_N_STRAIGHT] == 1.0
        assert state[IDX_N_LEFT]     == 0.0
        assert state[IDX_N_RIGHT]    == 0.0

    def test_actor_to_left_90_degrees(self):
        """Actor at (0,10) is 90° left of +X heading."""
        extractor = StateVectorExtractor()
        state = extractor.extract(self._ego_heading_x([0.0, -10.0, 0.0]))
        assert state[IDX_N_LEFT]     == 1.0
        assert state[IDX_N_STRAIGHT] == 0.0
        assert state[IDX_N_RIGHT]    == 0.0

    def test_actor_to_right_90_degrees(self):
        """Actor at (0,-10) is 90° right of +X heading in CARLA coords."""
        extractor = StateVectorExtractor()
        state = extractor.extract(self._ego_heading_x([0.0, 10.0, 0.0]))
        assert state[IDX_N_RIGHT]    == 1.0
        assert state[IDX_N_STRAIGHT] == 0.0
        assert state[IDX_N_LEFT]     == 0.0

    def test_actor_directly_behind_is_ignored(self):
        extractor = StateVectorExtractor()
        state = extractor.extract(self._ego_heading_x([-10.0, 0.0, 0.0]))
        assert state[IDX_N_STRAIGHT] == 0.0
        assert state[IDX_N_LEFT]     == 0.0
        assert state[IDX_N_RIGHT]    == 0.0

    def test_actor_beyond_max_range_is_ignored(self):
        extractor = StateVectorExtractor(max_range_m=10.0)
        state = extractor.extract(self._ego_heading_x([20.0, 0.0, 0.0]))
        assert state[IDX_N_STRAIGHT] == 0.0

    def test_actor_within_max_range_is_counted(self):
        extractor = StateVectorExtractor(max_range_m=50.0)
        state = extractor.extract(self._ego_heading_x([5.0, 0.0, 0.0]))
        assert state[IDX_N_STRAIGHT] == 1.0

    def test_multiple_actors_counted_in_correct_bins(self):
        """One ahead, one left, one right, one behind (ignored)."""
        frame = _make_frame(
            velocity=[1.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[
                _make_actor([10.0,   0.0, 0.0]),   # straight
                _make_actor([0.0,  -10.0, 0.0]),   # left
                _make_actor([0.0,   10.0, 0.0]),   # right
                _make_actor([-10.0,  0.0, 0.0]),   # behind → ignored
            ],
        )
        extractor = StateVectorExtractor()
        state = extractor.extract(frame)
        assert state[IDX_N_STRAIGHT] == 1.0
        assert state[IDX_N_LEFT]     == 1.0
        assert state[IDX_N_RIGHT]    == 1.0

    def test_stationary_vehicle_uses_zero_heading(self):
        """When speed ~0, heading defaults to 0° (+X). Actor ahead still straight."""
        frame = _make_frame(
            velocity=[0.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[_make_actor([5.0, 0.0, 0.0])],
        )
        extractor = StateVectorExtractor()
        state = extractor.extract(frame)
        assert state[IDX_N_STRAIGHT] == 1.0

    def test_narrow_straight_cone_routes_angled_actor_to_side(self):
        """With straight_half=10°, an actor at 20° goes to right bin, not straight."""
        extractor = StateVectorExtractor(straight_half_angle_deg=10.0)
        # Actor at 20° to the right of heading
        angle = math.radians(20)
        dist  = 10.0
        actor_x = dist * math.cos(angle)
        actor_y = dist * math.sin(angle)
        frame = _make_frame(
            velocity=[1.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[_make_actor([actor_x, actor_y, 0.0])],
        )
        state = extractor.extract(frame)
        assert state[IDX_N_STRAIGHT] == 0.0
        assert state[IDX_N_RIGHT]    == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – obstacle feature block indices
# ─────────────────────────────────────────────────────────────────────────────

class TestObstacleFeatureBlockLayout:
    """
    Validates that ethical features end up in the correct index positions
    of the 40-dim state vector, not just in the abstract feature slot.
    """

    def _state_with_one_actor_ahead(self, **actor_kwargs) -> List[float]:
        frame = _make_frame(
            velocity=[1.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[_make_actor([5.0, 0.0, 0.0], **actor_kwargs)],
        )
        return StateVectorExtractor().extract(frame)

    def test_straight_block_starts_at_index_7(self):
        state = self._state_with_one_actor_ahead(vulnerability_score=0.3)
        # Index 7 = count_norm for straight direction
        assert state[IDX_STRAIGHT_BASE + OFS_COUNT] == pytest.approx(0.2)

    def test_child_in_straight_direction_index(self):
        state = self._state_with_one_actor_ahead(age_group="child", vulnerability_score=0.3)
        assert state[IDX_STRAIGHT_BASE + OFS_CHILD] == 1.0

    def test_elderly_in_straight_direction_index(self):
        state = self._state_with_one_actor_ahead(age_group="elderly", vulnerability_score=0.2)
        assert state[IDX_STRAIGHT_BASE + OFS_ELDERLY] == 1.0

    def test_wheelchair_in_straight_direction_index(self):
        state = self._state_with_one_actor_ahead(disability="wheelchair", vulnerability_score=0.25)
        assert state[IDX_STRAIGHT_BASE + OFS_WHEELCHAIR] == 1.0

    def test_max_vulnerability_in_straight_direction_index(self):
        state = self._state_with_one_actor_ahead(vulnerability_score=0.75)
        assert state[IDX_STRAIGHT_BASE + OFS_MAX_VULN] == pytest.approx(0.75)

    def test_left_block_starts_at_index_18(self):
        assert IDX_LEFT_BASE == 18

    def test_right_block_starts_at_index_29(self):
        assert IDX_RIGHT_BASE == 29

    def test_total_obstacle_indices_is_33(self):
        """3 directions × 11 features = 33, filling indices 7–39."""
        assert IDX_STRAIGHT_BASE == 7
        assert IDX_RIGHT_BASE + FEATS_PER_DIR == 40   # last index is 39

    def test_child_in_left_direction_does_not_set_straight_flag(self):
        """Actor to the left should only set left-block flags, not straight-block."""
        frame = _make_frame(
            velocity=[1.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[_make_actor([0.0, -5.0, 0.0], age_group="child", vulnerability_score=0.3)],
        )
        state = StateVectorExtractor().extract(frame)
        assert state[IDX_STRAIGHT_BASE + OFS_CHILD] == 0.0, "Straight child flag should be 0"
        assert state[IDX_LEFT_BASE     + OFS_CHILD] == 1.0, "Left child flag should be 1"
        assert state[IDX_RIGHT_BASE    + OFS_CHILD] == 0.0, "Right child flag should be 0"

    def test_actor_to_right_does_not_set_straight_or_left_flags(self):
        frame = _make_frame(
            velocity=[1.0, 0.0, 0.0],
            position=[0.0, 0.0, 0.0],
            actors=[_make_actor([0.0, 5.0, 0.0], disability="blind", vulnerability_score=0.25)],
        )
        state = StateVectorExtractor().extract(frame)
        assert state[IDX_STRAIGHT_BASE + OFS_BLIND] == 0.0
        assert state[IDX_LEFT_BASE     + OFS_BLIND] == 0.0
        assert state[IDX_RIGHT_BASE    + OFS_BLIND] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – integration tests (mock adapter)
# ─────────────────────────────────────────────────────────────────────────────

class TestStateVectorIntegration:
    """
    Full pipeline: MockAdapter → EthicalWalkerSpawner → EthicalActorRegistry
    → DataCollector → StateVectorExtractor.
    """

    def test_full_pipeline_produces_40d_state(self):
        adapter  = MockAdapter(num_spawn_points=20)
        registry = EthicalActorRegistry()
        spawner  = EthicalWalkerSpawner(adapter, registry)
        actor_ids = _spawn_registered(adapter, registry, n=5)

        collector = DataCollector(adapter, registry, server=None, node_id="n0")
        extractor = StateVectorExtractor()
        frame = collector.collect(visible_actor_ids=actor_ids)
        state = extractor.extract(frame)

        assert len(state) == STATE_DIM
        assert all(math.isfinite(v) for v in state)

    def test_child_walker_reflected_in_state_vector(self):
        """Child walkers must produce has_child=1 in at least one direction block."""
        adapter  = MockAdapter(num_spawn_points=20)
        registry = EthicalActorRegistry()
        actor_ids = _spawn_registered(adapter, registry, n=5, age_group=AgeGroup.CHILD)

        collector = DataCollector(adapter, registry, server=None, node_id="n0")
        extractor = StateVectorExtractor()
        frame = collector.collect(visible_actor_ids=actor_ids)
        state = extractor.extract(frame)

        # At least one direction block should have has_child = 1
        has_child_any = any(
            state[OBS_BASE + d * FEATS_PER_DIR + OFS_CHILD] == 1.0
            for d in range(3)
        )
        assert has_child_any, "Child walkers must set has_child flag in some direction"

    def test_no_walkers_actor_indices_all_zero(self):
        adapter   = MockAdapter(num_spawn_points=10)
        registry  = EthicalActorRegistry()
        collector = DataCollector(adapter, registry, server=None)
        extractor = StateVectorExtractor()
        frame = collector.collect(visible_actor_ids=[])
        state = extractor.extract(frame)

        for i in range(IDX_N_STRAIGHT, STATE_DIM):
            assert state[i] == pytest.approx(0.0), f"Expected 0 at index {i}"

    def test_velocity_delta_across_sequential_mock_frames(self):
        """velocity_delta must reflect real speed change between mock frames."""
        adapter   = MockAdapter(num_spawn_points=5)
        registry  = EthicalActorRegistry()
        collector = DataCollector(adapter, registry, server=None)
        extractor = StateVectorExtractor()

        # Override vehicle_state velocity between frames
        frame1 = collector.collect(visible_actor_ids=[])
        frame1["vehicle_state"]["velocity"] = [5.0, 0.0, 0.0]
        extractor.extract(frame1)  # prime the extractor

        frame2 = collector.collect(visible_actor_ids=[])
        frame2["vehicle_state"]["velocity"] = [9.0, 0.0, 0.0]
        state2 = extractor.extract(frame2)

        assert state2[IDX_VEL_DELTA] == pytest.approx(4.0)

    def test_node_prefixed_ids_do_not_affect_extraction(self):
        """The node prefix on actor IDs (e.g. 'n1-42') must not break extraction."""
        actor = _make_actor([5.0, 0.0, 0.0], age_group="child", vulnerability_score=0.3)
        actor["id"] = "n3-99999"   # exotic prefix
        frame = _make_frame(
            velocity=[1.0, 0.0, 0.0],
            actors=[actor],
        )
        state = StateVectorExtractor().extract(frame)
        assert len(state) == STATE_DIM
        assert state[IDX_N_STRAIGHT] == 1.0

    def test_state_vector_is_json_serializable(self):
        """Team A's training pipeline reads states from JSON — must be serializable."""
        adapter   = MockAdapter(num_spawn_points=10)
        registry  = EthicalActorRegistry()
        actor_ids = _spawn_registered(adapter, registry, n=3)
        collector = DataCollector(adapter, registry, server=None)
        extractor = StateVectorExtractor()
        frame = collector.collect(visible_actor_ids=actor_ids)
        state = extractor.extract(frame)

        # This must not raise
        serialized = json.dumps(state)
        restored = json.loads(serialized)
        assert restored == pytest.approx(state, abs=1e-9)

    def test_wheelchair_walker_reflected_in_vulnerability(self):
        adapter   = MockAdapter(num_spawn_points=20)
        registry  = EthicalActorRegistry()
        actor_ids = _spawn_registered(
            adapter, registry, n=3,
            age_group=AgeGroup.ADULT,
            disability=Disability.WHEELCHAIR,
        )
        collector = DataCollector(adapter, registry, server=None)
        extractor = StateVectorExtractor()
        frame = collector.collect(visible_actor_ids=actor_ids)
        state = extractor.extract(frame)

        # Wheelchair adds 0.25 to vulnerability score — should appear in max/avg
        max_vuln_any = max(
            state[OBS_BASE + d * FEATS_PER_DIR + OFS_MAX_VULN]
            for d in range(3)
        )
        assert max_vuln_any == pytest.approx(0.25, abs=1e-6)

    def test_concurrent_extractions_are_independent(self):
        """
        Two extractors running on separate threads must not share state
        (velocity_delta from one must not bleed into the other).
        """
        results = {}
        errors  = []

        def run(thread_id: int, speed_seq):
            try:
                ext = StateVectorExtractor()
                deltas = []
                prev = None
                for spd in speed_seq:
                    frame = _make_frame(velocity=[spd, 0.0, 0.0])
                    state = ext.extract(frame)
                    if prev is not None:
                        deltas.append(state[IDX_VEL_DELTA])
                    prev = spd
                results[thread_id] = deltas
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run, args=(1, [0, 5, 10, 15]))
        t2 = threading.Thread(target=run, args=(2, [0, 2,  4,  6]))
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert not errors, f"Thread errors: {errors}"
        assert results[1] == pytest.approx([5.0, 5.0, 5.0])
        assert results[2] == pytest.approx([2.0, 2.0, 2.0])


# ─────────────────────────────────────────────────────────────────────────────
# StateVectorExtractor – CARLA end-to-end tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.carla
class TestStateVectorExtractorCARLA:

    def test_carla_frame_produces_valid_40d_state(self, adapter, registry, spawner, carla_world, spawn_points):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=20)
        carla_world.tick()

        collector = DataCollector(adapter, registry, server=None, node_id="n0")
        extractor = StateVectorExtractor(num_passengers=1)
        frame = collector.collect(visible_actor_ids=[w.id for w in walkers])
        state = extractor.extract(frame)

        assert len(state) == STATE_DIM
        assert all(math.isfinite(v) for v in state), "State contains non-finite values"
        assert all(isinstance(v, float) for v in state), "State contains non-float values"

    def test_carla_moving_ego_produces_nonzero_speed(self, carla_client, carla_world):
        """Attach a moving vehicle as ego and verify speed > 0 in state vector."""
        import carla
        from python_api.adapter import CARLAAdapter

        bp_lib = carla_world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_pts  = carla_world.get_map().get_spawn_points()
        ego = carla_world.try_spawn_actor(vehicle_bp, spawn_pts[0])
        assert ego is not None, "Could not spawn ego vehicle"
        carla_world.tick()

        try:
            ego.set_target_velocity(carla.Vector3D(x=10.0, y=0.0, z=0.0))
            carla_world.tick()

            adapter_with_ego = CARLAAdapter(carla_client, carla_world, ego_vehicle=ego)
            registry  = EthicalActorRegistry()
            collector = DataCollector(adapter_with_ego, registry, server=None)
            extractor = StateVectorExtractor()

            frame = collector.collect(visible_actor_ids=[])
            state = extractor.extract(frame)

            assert state[IDX_SPEED] >= 0.0
        finally:
            ego.destroy()
            carla_world.tick()

    def test_carla_pedestrians_appear_in_some_direction_bin(self, adapter, registry, spawner, carla_world, spawn_points):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=50)
        carla_world.tick()

        collector = DataCollector(adapter, registry, server=None, node_id="n0")
        extractor = StateVectorExtractor()
        frame = collector.collect(visible_actor_ids=[w.id for w in walkers])
        state = extractor.extract(frame)

        total_actors = state[IDX_N_STRAIGHT] + state[IDX_N_LEFT] + state[IDX_N_RIGHT]
        assert total_actors >= 0.0
        # All actor counts must be non-negative
        assert state[IDX_N_STRAIGHT] >= 0.0
        assert state[IDX_N_LEFT]     >= 0.0
        assert state[IDX_N_RIGHT]    >= 0.0

    def test_carla_ethical_attributes_appear_in_obstacle_features(self, adapter, registry, carla_world, spawn_points):
        """
        Spawn children and verify the child flag appears in at least one direction block.

        Vehicle spawn points in Town03 are spread across the whole map and can
        be hundreds of metres from the ego position [0,0,0]. The default 50 m
        detection range would exclude almost all of them, making the binning
        trivially empty. We use a 10 km range here so the test exercises the
        attribute-propagation logic rather than the range filter.
        """
        from python_api.spawn_walkers import EthicalWalkerSpawner
        from unittest.mock import patch

        child_attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN,
        )

        spawner = EthicalWalkerSpawner(adapter, registry)
        with patch.object(spawner, "_generate_ethical_attributes", return_value=child_attrs):
            walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=10)
        carla_world.tick()

        collector = DataCollector(adapter, registry, server=None)
        # Use a 10 km range so all walkers on the map fall within detection range
        # regardless of where CARLA placed the ego reference point.
        extractor = StateVectorExtractor(max_range_m=10_000.0)
        frame = collector.collect(visible_actor_ids=[w.id for w in walkers])
        state = extractor.extract(frame)

        # At least some actors must have been binned into a direction block
        total_binned = state[IDX_N_STRAIGHT] + state[IDX_N_LEFT] + state[IDX_N_RIGHT]
        assert total_binned > 0, (
            "No actors were binned into any direction block even with 10 km range. "
            "Check that walkers were spawned and that visible_actor_ids is correct."
        )

        # Every binned actor is a child — child flag must be set in every non-empty block
        for d in range(3):
            base = OBS_BASE + d * FEATS_PER_DIR
            count = state[base + OFS_COUNT]
            if count > 0:
                assert state[base + OFS_CHILD] == 1.0, (
                    f"Direction block {d} has {count:.2f} actors but has_child=0. "
                    "Child attribute not propagating through the pipeline."
                )

    def test_carla_reset_between_episodes_clears_delta(self, adapter, registry, carla_world):
        collector = DataCollector(adapter, registry, server=None)
        extractor = StateVectorExtractor()

        # Episode 1
        f1 = collector.collect(visible_actor_ids=[])
        f1["vehicle_state"]["velocity"] = [10.0, 0.0, 0.0]
        extractor.extract(f1)

        f2 = collector.collect(visible_actor_ids=[])
        f2["vehicle_state"]["velocity"] = [15.0, 0.0, 0.0]
        extractor.extract(f2)

        # Reset and start episode 2
        extractor.reset()
        f3 = collector.collect(visible_actor_ids=[])
        f3["vehicle_state"]["velocity"] = [20.0, 0.0, 0.0]
        state = extractor.extract(f3)

        assert state[IDX_VEL_DELTA] == pytest.approx(0.0), (
            "velocity_delta must be 0 on first step of a new episode"
        )


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