"""
Tests for python_api/state_vector_extractor.py (ael-v1 output).

The golden test rebuilds every CATA-200 scenario as a frame of actors around an
ego vehicle at a random pose, and requires the extractor to return exactly the
lab's vector. CARLA tests need --carla and a running server.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional

import pytest
from ael_common import Path, check_vector, decode, index, load_cata_200
from ael_common.state_vector import DIRECTIONS

from python_api.actor_registry import EthicalActorRegistry
from python_api.adapter import MockAdapter
from python_api.collector import DataCollector
from python_api.ethical_attributes import AgeGroup, Disability, EthicalAttributeSchema, SocialRole
from python_api.state_vector_extractor import (
    StateVectorExtractor,
    _normalise_angle,
    classify_actor,
    classify_blueprint,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Where each path's actors go, relative to the ego: (metres forward, metres right).
_OFFSETS = {"straight": (10.0, 0.0), "left": (10.0, -6.0), "right": (10.0, 6.0)}

_BLUEPRINT = {
    "pedestrian": "walker.pedestrian.0001",
    "cyclist": "vehicle.bh.crossbike",
    "vehicle": "vehicle.tesla.model3",
    "truck": "vehicle.carlamotors.carlacola",
    "motorcycle": "vehicle.yamaha.yzf",
    "barrier": "static.prop.streetbarrier",
    "property": "static.prop.bench01",
}


def _frame(
    actors: Optional[List[Dict[str, Any]]] = None,
    velocity: Optional[List[float]] = None,
    position: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
) -> Dict[str, Any]:
    vehicle_state: Dict[str, Any] = {
        "position": position or [0.0, 0.0, 0.0],
        "velocity": velocity or [0.0, 0.0, 0.0],
        "controls": {"throttle": 0.0, "steer": 0.0, "brake": 0.0},
    }
    if rotation is not None:
        vehicle_state["rotation"] = rotation
    return {
        "timestamp": 0.0,
        "ethical_context": {"visible_actors": actors or []},
        "vehicle_state": vehicle_state,
        "scene_metadata": {"weather": "clear_noon", "map": "TestMap"},
    }


def _actor(kind: str, position: List[float], **fields: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"id": "n0-1", "type": _BLUEPRINT[kind], "position": position, "velocity": [0.0, 0.0, 0.0]}
    if kind == "pedestrian":
        entry.update(age_group="adult", disability="none", pregnancy=False, social_role="civilian")
    entry.update(fields)
    return entry


def _place(ego: List[float], yaw_deg: float, forward: float, right: float) -> List[float]:
    """World position at (forward, right) metres from the ego. CARLA is left-handed: y points right at yaw 0."""
    yaw = math.radians(yaw_deg)
    return [
        ego[0] + forward * math.cos(yaw) - right * math.sin(yaw),
        ego[1] + forward * math.sin(yaw) + right * math.cos(yaw),
        0.0,
    ]


def _actors_for_path(path: Path, casualties: int, passengers: int, position: List[float]) -> List[Dict[str, Any]]:
    """Actors that should extract back to exactly (path, casualties)."""
    actors: List[Dict[str, Any]] = []
    people = 0
    pedestrians: List[Dict[str, Any]] = []
    for kind in sorted(path.obstacles):
        if kind in ("vehicle", "truck"):
            actors.append(_actor(kind, position, occupants=1))
            people += 1
        elif kind == "pedestrian":
            pedestrians.append(_actor(kind, position))
            people += 1
        elif kind in ("cyclist", "motorcycle"):
            actors.append(_actor(kind, position))
            people += 1
        else:
            actors.append(_actor(kind, position))
    if "barrier" in path.obstacles:
        people += passengers

    # Child and elderly can't be the same person.
    if {"child", "elderly"} <= path.vulnerable:
        pedestrians.append(_actor("pedestrian", position))
        people += 1
    extra = casualties - people
    assert extra >= 0, f"cannot build {path} with {casualties} casualties"
    for _ in range(extra):
        if pedestrians:
            pedestrians.append(_actor("pedestrian", position))
        else:
            rider = next(a for a in actors if a["type"] in (_BLUEPRINT["cyclist"], _BLUEPRINT["motorcycle"], _BLUEPRINT["vehicle"], _BLUEPRINT["truck"]))
            if "occupants" in rider:
                rider["occupants"] += 1
            else:
                actors.append(dict(rider))

    groups = sorted(path.vulnerable)
    if "child" in groups:
        pedestrians[0]["age_group"] = "child"
    if "elderly" in groups:
        pedestrians[-1]["age_group"] = "elderly"
    if "pregnant" in groups:
        pedestrians[0]["pregnancy"] = True
    if "disabled" in groups:
        pedestrians[0]["disability"] = "wheelchair"
    return actors + pedestrians


def _frame_for_vector(vec: List[float], ego: List[float], yaw_deg: float) -> Dict[str, Any]:
    state = decode(vec)
    actors: List[Dict[str, Any]] = []
    for direction in DIRECTIONS:
        forward, right = _OFFSETS[direction]
        position = _place(ego, yaw_deg, forward, right)
        actors += _actors_for_path(state.paths[direction], state.casualties[direction], state.num_passengers, position)
    yaw = math.radians(yaw_deg)
    speed = state.velocity_ego
    return _frame(
        actors,
        velocity=[speed * math.cos(yaw), speed * math.sin(yaw), 0.0],
        position=ego,
        rotation=[0.0, yaw_deg, 0.0],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Golden: every CATA-200 scenario round-trips through a built scene
# ─────────────────────────────────────────────────────────────────────────────

class TestCata200Golden:

    def test_cata_s001_scene_extracts_to_lab_vector(self):
        frame = _frame(
            [
                _actor("pedestrian", [10.0, 0.0, 0.0]),
                _actor("pedestrian", [10.0, -6.0, 0.0], age_group="elderly"),
                _actor("pedestrian", [10.0, -6.0, 0.0], age_group="child"),
                _actor("barrier", [10.0, 6.0, 0.0]),
            ],
            velocity=[10.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
        )
        assert StateVectorExtractor(num_passengers=1).extract(frame) == load_cata_200()["CATA_S001"]

    def test_all_200_scenarios_round_trip_from_any_ego_pose(self):
        rng = random.Random(0)
        for sid, vec in sorted(load_cata_200().items()):
            ego = [rng.uniform(-500, 500), rng.uniform(-500, 500), 0.0]
            yaw = rng.uniform(-180, 180)
            frame = _frame_for_vector(vec, ego, yaw)
            extractor = StateVectorExtractor(num_passengers=int(vec[1]))
            got = extractor.extract(frame)
            assert got == pytest.approx(vec, abs=1e-9), sid


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle classification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassification:

    @pytest.mark.parametrize("type_id, expected", [
        ("walker.pedestrian.0001", "pedestrian"),
        ("walker", "pedestrian"),
        ("vehicle.diamondback.century", "cyclist"),
        ("vehicle.kawasaki.ninja", "motorcycle"),
        ("vehicle.carlamotors.firetruck", "truck"),
        ("vehicle.audi.tt", "vehicle"),
        ("static.prop.streetbarrier", "barrier"),
        ("static.prop.chainbarrier", "barrier"),
        ("static.prop.trafficcone01", "property"),
        ("traffic.traffic_light", "property"),
        ("sensor.camera.rgb", None),
        ("controller.ai.walker", None),
    ])
    def test_blueprints(self, type_id, expected):
        assert classify_blueprint(type_id) == expected

    def test_explicit_obstacle_type_wins(self):
        assert classify_actor({"obstacle_type": "barrier", "type": "static.prop.bench01"}) == "barrier"

    def test_invalid_explicit_type_raises(self):
        with pytest.raises(ValueError):
            classify_actor({"id": "x", "obstacle_type": "wall"})

    def test_base_type_beats_blueprint_name(self):
        assert classify_actor({"base_type": "bicycle", "type": "vehicle.unknown.model"}) == "cyclist"
        assert classify_actor({"base_type": "bus", "type": "vehicle.unknown.model"}) == "truck"

    def test_legacy_entry_with_attributes_is_pedestrian(self):
        assert classify_actor({"age_group": "adult"}) == "pedestrian"

    def test_unknown_entry_is_ignored(self):
        assert classify_actor({"position": [1, 0, 0]}) is None
        state = StateVectorExtractor().extract(_frame([{"position": [10.0, 0.0, 0.0]}]))
        assert state[index("casualties_if_straight")] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Path contents and casualties
# ─────────────────────────────────────────────────────────────────────────────

def _straight(*actors: Dict[str, Any], **kwargs: Any):
    return StateVectorExtractor(**kwargs).extract_state(_frame(list(actors), rotation=[0.0, 0.0, 0.0]))


class TestPathsAndCasualties:

    def test_vulnerable_flags(self):
        p = [10.0, 0.0, 0.0]
        state = _straight(
            _actor("pedestrian", p, age_group="child"),
            _actor("pedestrian", p, age_group="elderly", pregnancy=True),
            _actor("pedestrian", p, disability="cane"),
        )
        assert state.paths["straight"] == Path(["pedestrian"], ["child", "elderly", "pregnant", "disabled"])
        assert state.casualties["straight"] == 3

    @pytest.mark.parametrize("disability", ["wheelchair", "cane", "blind"])
    def test_every_disability_counts_as_disabled(self, disability):
        state = _straight(_actor("pedestrian", [10.0, 0.0, 0.0], disability=disability))
        assert state.paths["straight"].vulnerable == frozenset(["disabled"])

    def test_teen_and_adult_are_not_vulnerable(self):
        state = _straight(_actor("pedestrian", [10.0, 0.0, 0.0], age_group="teen"))
        assert state.paths["straight"].vulnerable == frozenset()

    def test_vulnerable_attributes_on_non_pedestrians_are_ignored(self):
        state = _straight(_actor("cyclist", [10.0, 0.0, 0.0], age_group="child"))
        assert state.paths["straight"] == Path(["cyclist"])

    def test_vehicle_occupants(self):
        p = [10.0, 0.0, 0.0]
        assert _straight(_actor("vehicle", p)).casualties["straight"] == 1
        assert _straight(_actor("vehicle", p), default_vehicle_occupants=3).casualties["straight"] == 3
        assert _straight(_actor("truck", p, occupants=2)).casualties["straight"] == 2
        assert _straight(_actor("vehicle", p, occupants=None)).casualties["straight"] == 1

    def test_barrier_kills_ego_passengers_once(self):
        p = [10.0, 0.0, 0.0]
        state = _straight(_actor("barrier", p), _actor("barrier", p), num_passengers=3)
        assert state.casualties["straight"] == 3

    def test_barrier_and_pedestrian_add_up(self):
        p = [10.0, 0.0, 0.0]
        state = _straight(_actor("barrier", p), _actor("pedestrian", p), num_passengers=2)
        assert state.casualties["straight"] == 3

    def test_property_kills_nobody(self):
        state = _straight(_actor("property", [10.0, 0.0, 0.0]))
        assert state.paths["straight"] == Path(["property"])
        assert state.casualties["straight"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Direction binning
# ─────────────────────────────────────────────────────────────────────────────

def _direction_of(position: List[float], rotation=(0.0, 0.0, 0.0), velocity=None, **kwargs: Any) -> Optional[str]:
    frame = _frame([_actor("pedestrian", position)], velocity=velocity, rotation=list(rotation) if rotation else None)
    state = StateVectorExtractor(**kwargs).extract_state(frame)
    hit = [d for d in DIRECTIONS if state.casualties[d]]
    return hit[0] if hit else None


class TestDirections:

    @pytest.mark.parametrize("position, expected", [
        ([10.0, 0.0, 0.0], "straight"),
        ([10.0, 4.0, 0.0], "straight"),    # 21.8°
        ([10.0, -4.0, 0.0], "straight"),
        ([10.0, 5.0, 0.0], "right"),       # 26.6°, positive y is right in CARLA
        ([10.0, -5.0, 0.0], "left"),
        ([0.0, 10.0, 0.0], "right"),       # exactly 90°
        ([-10.0, 0.0, 0.0], None),         # behind
        ([-1.0, 10.0, 0.0], None),         # just behind the side edge
        ([60.0, 0.0, 0.0], None),          # out of range
    ])
    def test_bins_relative_to_heading(self, position, expected):
        assert _direction_of(position) == expected

    def test_yaw_rotates_the_bins(self):
        # Facing +y (yaw 90) in CARLA's left-handed frame: -x is to the right, +x to the left.
        assert _direction_of([0.0, 10.0, 0.0], rotation=(0.0, 90.0, 0.0)) == "straight"
        assert _direction_of([-10.0, 10.0, 0.0], rotation=(0.0, 90.0, 0.0)) == "right"
        assert _direction_of([10.0, 10.0, 0.0], rotation=(0.0, 90.0, 0.0)) == "left"

    def test_yaw_beats_velocity_direction(self):
        # Sliding sideways: the car faces +x but moves along +y.
        assert _direction_of([10.0, 0.0, 0.0], rotation=(0.0, 0.0, 0.0), velocity=[0.0, 5.0, 0.0]) == "straight"

    def test_velocity_heading_used_without_rotation(self):
        assert _direction_of([0.0, 10.0, 0.0], rotation=None, velocity=[0.0, 5.0, 0.0]) == "straight"

    def test_stationary_without_rotation_faces_plus_x(self):
        assert _direction_of([10.0, 0.0, 0.0], rotation=None) == "straight"

    def test_range_is_configurable(self):
        assert _direction_of([60.0, 0.0, 0.0], max_range_m=100.0) == "straight"

    def test_narrow_cone(self):
        assert _direction_of([10.0, 2.0, 0.0], straight_half_angle_deg=5.0) == "right"

    @pytest.mark.parametrize("angle, expected", [
        (0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi), (3 * math.pi, math.pi), (2 * math.pi, 0.0),
    ])
    def test_normalise_angle(self, angle, expected):
        assert _normalise_angle(angle) == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# Ego fields and output contract
# ─────────────────────────────────────────────────────────────────────────────

class TestEgoAndOutput:

    def test_speed_is_3d_magnitude(self):
        state = StateVectorExtractor().extract(_frame(velocity=[3.0, 4.0, 12.0]))
        assert state[index("velocity_ego")] == pytest.approx(13.0)

    def test_passengers(self):
        assert StateVectorExtractor(num_passengers=4).extract(_frame())[index("num_passengers")] == 4.0

    def test_lane_position_is_zero_until_map_support(self):
        state = StateVectorExtractor().extract(_frame(position=[3.0, 7.5, 0.0]))
        assert state[index("lane_position")] == 0.0

    def test_velocity_delta_and_reset(self):
        extractor = StateVectorExtractor()
        assert extractor.extract(_frame(velocity=[5.0, 0.0, 0.0]))[3] == 0.0
        assert extractor.extract(_frame(velocity=[8.0, 0.0, 0.0]))[3] == pytest.approx(3.0)
        assert extractor.extract(_frame(velocity=[6.0, 0.0, 0.0]))[3] == pytest.approx(-2.0)
        extractor.reset()
        assert extractor.extract(_frame(velocity=[20.0, 0.0, 0.0]))[3] == 0.0

    def test_empty_frame(self):
        state = StateVectorExtractor().extract({})
        assert state == [0.0, 1.0] + [0.0] * 38

    def test_output_is_valid_ael_v1_and_json_safe(self):
        frame = _frame_for_vector(load_cata_200()["CATA_S005"], [0.0, 0.0, 0.0], 0.0)
        state = StateVectorExtractor(num_passengers=3).extract(frame)
        check_vector(state)
        assert all(type(v) is float for v in state)
        assert json.loads(json.dumps(state)) == state


# ─────────────────────────────────────────────────────────────────────────────
# Collector → extractor (mock adapter)
# ─────────────────────────────────────────────────────────────────────────────

def _register_walker(adapter: MockAdapter, registry: EthicalActorRegistry, **attrs: Any) -> int:
    bp = adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
    aid = adapter.spawn_walker(adapter.get_spawn_points()[0], bp)
    fields = dict(age_group=AgeGroup.ADULT, disability=Disability.NONE, pregnancy=False,
                  group_size=1, social_role=SocialRole.CIVILIAN)
    fields.update(attrs)
    registry.register(aid, EthicalAttributeSchema(**fields))
    return aid


class TestCollectorIntegration:

    def test_collector_entries_carry_type_and_pregnancy(self):
        adapter, registry = MockAdapter(num_spawn_points=5), EthicalActorRegistry()
        aid = _register_walker(adapter, registry, pregnancy=True)
        frame = DataCollector(adapter, registry, server=None, node_id="n0").collect(visible_actor_ids=[aid])
        entry = frame["ethical_context"]["visible_actors"][0]
        assert entry["type"] == "walker"
        assert entry["pregnancy"] is True
        assert "base_type" not in entry

    def test_registered_walkers_reach_the_vector(self):
        adapter, registry = MockAdapter(num_spawn_points=5), EthicalActorRegistry()
        ids = [
            _register_walker(adapter, registry, age_group=AgeGroup.CHILD),
            _register_walker(adapter, registry, pregnancy=True),
            _register_walker(adapter, registry, disability=Disability.BLIND),
        ]
        frame = DataCollector(adapter, registry, server=None).collect(visible_actor_ids=ids)
        state = StateVectorExtractor().extract_state(frame)
        # The mock puts walkers and ego at the origin, which bins as straight.
        assert state.paths["straight"] == Path(["pedestrian"], ["child", "pregnant", "disabled"])
        assert state.casualties["straight"] == 3

    def test_mock_snapshot_has_rotation(self):
        snapshot = MockAdapter().get_snapshot()
        assert snapshot["vehicle_state"]["rotation"] == [0.0, 0.0, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# CARLA end-to-end (requires --carla and a running server)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.carla
class TestStateVectorExtractorCARLA:

    def test_carla_frame_is_valid_ael_v1(self, adapter, registry, spawner, carla_world, spawn_points):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=20)
        carla_world.tick()
        frame = DataCollector(adapter, registry, server=None, node_id="n0").collect(
            visible_actor_ids=[w.id for w in walkers]
        )
        check_vector(StateVectorExtractor().extract(frame))
        assert all(e["type"].startswith("walker.") for e in frame["ethical_context"]["visible_actors"])

    def test_carla_ego_snapshot_has_rotation_and_speed(self, carla_client, carla_world):
        import carla
        from python_api.adapter import CARLAAdapter

        bp = carla_world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = carla_world.try_spawn_actor(bp, carla_world.get_map().get_spawn_points()[0])
        assert ego is not None, "Could not spawn ego vehicle"
        try:
            ego.set_target_velocity(carla.Vector3D(x=10.0, y=0.0, z=0.0))
            carla_world.tick()
            adapter = CARLAAdapter(carla_client, carla_world, ego_vehicle=ego)
            frame = DataCollector(adapter, EthicalActorRegistry(), server=None).collect(visible_actor_ids=[])
            assert len(frame["vehicle_state"]["rotation"]) == 3
            state = StateVectorExtractor().extract(frame)
            assert state[index("velocity_ego")] >= 0.0
        finally:
            ego.destroy()
            carla_world.tick()

    def test_carla_children_set_child_flag(self, adapter, registry, carla_world, spawn_points):
        from unittest.mock import patch

        from python_api.spawn_walkers import EthicalWalkerSpawner

        child = EthicalAttributeSchema(age_group=AgeGroup.CHILD, disability=Disability.NONE,
                                       pregnancy=False, group_size=1, social_role=SocialRole.CIVILIAN)
        spawner = EthicalWalkerSpawner(adapter, registry)
        with patch.object(spawner, "_generate_ethical_attributes", return_value=child):
            walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=10)
        carla_world.tick()
        frame = DataCollector(adapter, registry, server=None).collect(visible_actor_ids=[w.id for w in walkers])
        # Spawn points are spread over the map, so widen the range to bin them all.
        state = StateVectorExtractor(max_range_m=10_000.0).extract_state(frame)
        occupied = [d for d in DIRECTIONS if state.paths[d].obstacles]
        assert occupied, "No walkers were binned into any path"
        for d in occupied:
            assert "child" in state.paths[d].vulnerable
