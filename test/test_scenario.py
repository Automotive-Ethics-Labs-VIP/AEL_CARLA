"""Tests for python_api/scenario.py: the scenario file format, its checks, and the CARLA builder wiring."""

from __future__ import annotations

import copy
import json
import os
import sys
import types
from typing import Any, Dict, List

import pytest
from ael_common import decode, encode, load_cata_200
from ael_common.state_vector import DIRECTIONS

from python_api.actor_registry import EthicalActorRegistry
from python_api.scenario import (
    CarlaScenarioBuilder,
    ScenarioError,
    annotate_frame,
    check_scenario,
    declared_state,
    load_scenario,
    main,
    parse_scenario,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "scenarios", "CATA_S001.json")

with open(EXAMPLE, encoding="utf-8") as _f:
    EXAMPLE_RAW = json.load(_f)

_SIDE = {"straight": 0.0, "left": -5.0, "right": 5.0}


def _raw(**changes: Any) -> Dict[str, Any]:
    raw = copy.deepcopy(EXAMPLE_RAW)
    raw.update(changes)
    return raw


def _problems(raw: Any) -> List[str]:
    scenario, problems = parse_scenario(raw)
    return problems if scenario is None else check_scenario(scenario)


def scenario_from_vector(sid: str, vec: List[float]) -> Dict[str, Any]:
    """A scenario file whose declared state is exactly the given ael-v1 vector."""
    state = decode(vec)
    paths: Dict[str, List[Dict[str, Any]]] = {}
    for direction in DIRECTIONS:
        path, casualties = state.paths[direction], state.casualties[direction]
        actors: List[Dict[str, Any]] = []
        place = {"forward_m": 8.0, "right_m": _SIDE[direction]}
        people = 0
        for kind in sorted(path.obstacles - {"pedestrian"}):
            actors.append(dict(type=kind, **place))
            if kind in ("cyclist", "motorcycle", "vehicle", "truck"):
                people += 1
        if "barrier" in path.obstacles:
            people += state.num_passengers
        pedestrians: List[Dict[str, Any]] = []
        if "pedestrian" in path.obstacles:
            pedestrians.append(dict(type="pedestrian", **place))
            if {"child", "elderly"} <= path.vulnerable:
                pedestrians.append(dict(type="pedestrian", **place))
            people += len(pedestrians)
        extra = casualties - people
        assert extra >= 0, sid
        for _ in range(extra):
            if pedestrians:
                pedestrians.append(dict(type="pedestrian", **place))
            else:
                riders = [a for a in actors if a["type"] in ("vehicle", "truck")]
                if riders:
                    riders[0]["occupants"] = riders[0].get("occupants", 1) + 1
                else:
                    actors.append(dict(next(a for a in actors if a["type"] in ("cyclist", "motorcycle"))))
        if "child" in path.vulnerable:
            pedestrians[0]["age_group"] = "child"
        if "elderly" in path.vulnerable:
            pedestrians[-1]["age_group"] = "elderly"
        if "pregnant" in path.vulnerable:
            pedestrians[0]["pregnancy"] = True
        if "disabled" in path.vulnerable:
            pedestrians[0]["disability"] = "wheelchair"
        paths[direction] = actors + pedestrians
    return {
        "spec": "ael-v1", "kind": "carla_scenario", "id": sid,
        "world": {"map": "Town01"},
        "ego": {"spawn_point": 0, "speed_mps": state.velocity_ego, "passengers": state.num_passengers},
        "paths": paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The example and the CATA-200 round trip
# ─────────────────────────────────────────────────────────────────────────────

def test_example_file_passes_and_matches_lab_vector():
    scenario = load_scenario(EXAMPLE)
    assert encode(declared_state(scenario)) == load_cata_200()["CATA_S001"]


SCENARIO_FILES = sorted(
    os.path.join(ROOT, "scenarios", f) for f in os.listdir(os.path.join(ROOT, "scenarios")) if f.endswith(".json")
)


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=os.path.basename)
def test_every_committed_scenario_file_passes(path):
    scenario = load_scenario(path)
    assert scenario.id == os.path.splitext(os.path.basename(path))[0], "file name must match the scenario id"


def test_every_cata_scenario_can_be_written_as_a_valid_file():
    for sid, vec in sorted(load_cata_200().items()):
        scenario, problems = parse_scenario(scenario_from_vector(sid, vec))
        assert not problems, (sid, problems)
        assert check_scenario(scenario) == [], sid
        assert encode(declared_state(scenario)) == vec, sid


# ─────────────────────────────────────────────────────────────────────────────
# Format checks: what a hand-written or LLM-written file can get wrong
# ─────────────────────────────────────────────────────────────────────────────

def _actor_change(direction: str, index: int, **changes: Any) -> Dict[str, Any]:
    raw = _raw()
    raw["paths"][direction][index].update(changes)
    return raw


@pytest.mark.parametrize("raw, expected", [
    ([], "JSON object"),
    (_raw(spec="ael-v2"), "spec"),
    (_raw(kind="scenario"), "kind"),
    (_raw(id=""), "id"),
    (_raw(id=5), "id"),
    (_raw(description=3), "description"),
    (_raw(notes="x"), "unknown field"),
    (_raw(world={"weather": "ClearNoon"}), "map"),
    (_raw(world={"map": "Town01", "time": "noon"}), "unknown field"),
    (_raw(world={"map": "Town01", "weather": 3}), "weather"),
    (_raw(ego="tesla"), "ego must be an object"),
    (_raw(ego={"spawn_point": -1, "speed_mps": 10, "passengers": 1}), "spawn_point"),
    (_raw(ego={"spawn_point": 1.5, "speed_mps": 10, "passengers": 1}), "spawn_point"),
    (_raw(ego={"spawn_point": 0, "speed_mps": -1, "passengers": 1}), "speed_mps"),
    (_raw(ego={"spawn_point": 0, "speed_mps": True, "passengers": 1}), "speed_mps"),
    (_raw(ego={"spawn_point": 0, "speed_mps": 10, "passengers": 1.5}), "passengers"),
    (_raw(ego={"spawn_point": 0, "speed_mps": 10, "passengers": 1, "blueprint": 3}), "ego.blueprint"),
    (_raw(ego={"spawn_point": 0, "speed_mps": 10, "passengers": 1, "color": "red"}), "unknown field"),
    (_raw(paths={"straight": [], "left": []}), "exactly the keys"),
    (_raw(paths={"straight": {}, "left": [], "right": []}), "must be a list"),
    (_raw(paths={"straight": ["adult"], "left": [], "right": []}), "must be an object"),
    (_actor_change("straight", 0, type="dog"), "type"),
    (_actor_change("straight", 0, foward_m=8), "unknown field"),
    (_actor_change("straight", 0, forward_m=0), "forward_m"),
    (_actor_change("straight", 0, forward_m=-5), "forward_m"),
    (_actor_change("straight", 0, forward_m="8"), "forward_m"),
    (_actor_change("straight", 0, right_m=None), "right_m"),
    (_actor_change("straight", 0, blueprint=7), "blueprint"),
    (_actor_change("straight", 0, age_group="baby"), "age_group"),
    (_actor_change("straight", 0, disability="crutches"), "disability"),
    (_actor_change("straight", 0, social_role="police"), "social_role"),
    (_actor_change("straight", 0, pregnancy="yes"), "pregnancy"),
    (_actor_change("straight", 0, occupants=2), "unknown field"),     # pedestrians have no occupants
    (_actor_change("right", 0, age_group="child"), "unknown field"),  # barriers have no age
])
def test_format_problems(raw, expected):
    problems = _problems(raw)
    assert any(expected in p for p in problems), problems


@pytest.mark.parametrize("occupants", [-1, 1.5, "2"])
def test_bad_vehicle_occupants(occupants):
    raw = _raw()
    raw["paths"]["left"] = [{"type": "vehicle", "forward_m": 8, "right_m": -5, "occupants": occupants}]
    assert any("occupants" in p for p in _problems(raw))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry checks
# ─────────────────────────────────────────────────────────────────────────────

def test_actor_too_close_to_centre_is_flagged():
    problems = _problems(_actor_change("left", 0, right_m=-2.0))
    assert any("lands in straight, not left" in p for p in problems)


def test_actor_on_wrong_side_is_flagged():
    problems = _problems(_actor_change("left", 0, right_m=4.0))
    assert any("lands in right, not left" in p for p in problems)


def test_actor_out_of_range_is_flagged():
    problems = _problems(_actor_change("straight", 0, forward_m=80))
    assert any("lands in no path" in p for p in problems)


def test_actor_far_to_the_side_still_counts_as_left():
    # forward 0.1 m, 10 m to the side: 89.4 deg, still left; the check uses the extractor's cones.
    assert _problems(_actor_change("left", 0, forward_m=0.1, right_m=-10)) == []


def test_empty_paths_are_allowed():
    raw = _raw(paths={"straight": [], "left": [], "right": []})
    scenario, problems = parse_scenario(raw)
    assert problems == [] and check_scenario(scenario) == []


# ─────────────────────────────────────────────────────────────────────────────
# Declared state
# ─────────────────────────────────────────────────────────────────────────────

def test_declared_casualties():
    raw = _raw(ego={"spawn_point": 0, "speed_mps": 10, "passengers": 3}, paths={
        "straight": [{"type": "truck", "forward_m": 8, "right_m": 0, "occupants": 2}],
        "left": [{"type": "cyclist", "forward_m": 8, "right_m": -5},
                 {"type": "pedestrian", "forward_m": 8, "right_m": -5, "pregnancy": True}],
        "right": [{"type": "barrier", "forward_m": 8, "right_m": 5}, {"type": "property", "forward_m": 8, "right_m": 5}],
    })
    scenario, problems = parse_scenario(raw)
    assert problems == []
    state = declared_state(scenario)
    assert state.casualties == {"straight": 2, "left": 2, "right": 3}
    assert state.paths["left"].vulnerable == frozenset(["pregnant"])
    assert check_scenario(scenario) == []


# ─────────────────────────────────────────────────────────────────────────────
# Loading and CLI
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, content: str) -> str:
    path = tmp_path / "s.json"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_rejects_invalid_json(tmp_path):
    with pytest.raises(ScenarioError, match="not valid JSON"):
        load_scenario(_write(tmp_path, "{nope"))


def test_load_rejects_bad_geometry(tmp_path):
    with pytest.raises(ScenarioError, match="lands in"):
        load_scenario(_write(tmp_path, json.dumps(_actor_change("left", 0, right_m=4.0))))


def test_cli(tmp_path, capsys):
    bad = _write(tmp_path, json.dumps(_raw(kind="nope")))
    assert main(["check", EXAMPLE]) == 0
    assert "ok" in capsys.readouterr().out
    assert main(["check", EXAMPLE, bad]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert main([]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# CARLA builder wiring, against a fake carla module (not a real server)
# ─────────────────────────────────────────────────────────────────────────────

class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _fake_carla() -> types.ModuleType:
    carla = types.ModuleType("carla")
    carla.Location = lambda x=0.0, y=0.0, z=0.0: _Obj(x=x, y=y, z=z)
    carla.Rotation = lambda pitch=0.0, yaw=0.0, roll=0.0: _Obj(pitch=pitch, yaw=yaw, roll=roll)
    carla.Transform = lambda location, rotation: _Obj(location=location, rotation=rotation)
    carla.Vector3D = lambda x=0.0, y=0.0, z=0.0: _Obj(x=x, y=y, z=z)
    carla.WeatherParameters = _Obj(ClearNoon="clear-noon")
    return carla


class _Blueprint:
    def __init__(self, bp_id: str, age: str = "") -> None:
        self.id, self._age = bp_id, age

    def has_attribute(self, key: str) -> bool:
        return key == "age" and bool(self._age)

    def get_attribute(self, key: str) -> Any:
        return _Obj(as_str=lambda: self._age)


class _Library:
    def find(self, bp_id: str) -> _Blueprint:
        return _Blueprint(bp_id)

    def filter(self, pattern: str) -> List[_Blueprint]:
        return [_Blueprint("walker.pedestrian.0001", "adult"), _Blueprint("walker.pedestrian.0011", "child")]


class _Actor:
    _next = 100

    def __init__(self, blueprint: _Blueprint, transform: Any, world: "_World") -> None:
        _Actor._next += 1
        self.id, self.blueprint, self.transform, self.world = _Actor._next, blueprint, transform, world
        self.velocity = None

    def set_target_velocity(self, v: Any) -> None:
        self.velocity = v

    def destroy(self) -> bool:
        self.world.actors.remove(self)
        return True


class _World:
    def __init__(self, map_name: str) -> None:
        self.map_name, self.actors, self.weather = map_name, [], None

    def get_map(self) -> Any:
        start = _Obj(location=_Obj(x=100.0, y=50.0, z=0.5), rotation=_Obj(yaw=90.0))
        return _Obj(name=f"Carla/Maps/{self.map_name}", get_spawn_points=lambda: [start])

    def set_weather(self, weather: Any) -> None:
        self.weather = weather

    def get_blueprint_library(self) -> _Library:
        return _Library()

    def spawn_actor(self, blueprint: _Blueprint, transform: Any) -> _Actor:
        actor = _Actor(blueprint, transform, self)
        self.actors.append(actor)
        return actor


class _Client:
    def __init__(self) -> None:
        self.world = _World("Town03")

    def get_world(self) -> _World:
        return self.world

    def load_world(self, name: str) -> _World:
        self.world = _World(name)
        return self.world


@pytest.fixture
def fake_carla(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", _fake_carla())


def test_builder_spawns_and_registers(fake_carla):
    client, registry = _Client(), EthicalActorRegistry()
    builder = CarlaScenarioBuilder(client, registry)
    ego = builder.build(load_scenario(EXAMPLE))

    world = client.world
    assert world.map_name == "Town01" and world.weather == "clear-noon"
    assert len(world.actors) == 5  # ego + 3 pedestrians + barrier
    assert ego.blueprint.id == "vehicle.tesla.model3"
    assert ego.velocity.y == pytest.approx(10.0) and ego.velocity.x == pytest.approx(0.0, abs=1e-9)

    by_bp = {a.blueprint.id: a for a in world.actors}
    assert "static.prop.streetbarrier" in by_bp
    assert "walker.pedestrian.0011" in by_bp  # the child gets a child blueprint
    # Facing +y (yaw 90), the barrier 8 m ahead and 4 m right sits at x - 4, y + 8.
    barrier = by_bp["static.prop.streetbarrier"].transform.location
    assert (barrier.x, barrier.y) == pytest.approx((96.0, 58.0))

    walkers = [a for a in world.actors if a.blueprint.id.startswith("walker.")]
    ages = sorted(registry.get_attributes(w.id).age_group.value for w in walkers)
    assert ages == ["adult", "child", "elderly"]
    tags = builder.tags()
    assert sorted(t["obstacle_type"] for t in tags.values()) == ["barrier", "pedestrian", "pedestrian", "pedestrian"]
    assert sorted(builder.actor_ids()) == sorted(tags)

    builder.destroy()
    assert world.actors == [] and builder.tags() == {}


def test_builder_keeps_loaded_map_and_rejects_bad_spawn_point(fake_carla):
    client = _Client()
    client.world = _World("Town01")
    first = client.world
    raw = _raw(ego={"spawn_point": 5, "speed_mps": 10, "passengers": 1})
    scenario, _ = parse_scenario(raw)
    with pytest.raises(ScenarioError, match="spawn points"):
        CarlaScenarioBuilder(client, EthicalActorRegistry()).build(scenario)
    assert client.world is first


def test_annotate_frame():
    frame = {"ethical_context": {"visible_actors": [{"id": "n0-7"}, {"id": "n0-8"}, {"id": "weird"}]}}
    annotate_frame(frame, {7: {"obstacle_type": "barrier", "occupants": 1}})
    actors = frame["ethical_context"]["visible_actors"]
    assert actors[0]["obstacle_type"] == "barrier"
    assert "obstacle_type" not in actors[1] and "obstacle_type" not in actors[2]
