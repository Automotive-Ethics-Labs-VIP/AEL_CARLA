from __future__ import annotations

"""
Scenario files
==============
A scenario file describes one dilemma scene for CARLA as JSON, frozen at the
moment of decision: the town, the ego vehicle and its speed, and what is in each
path (straight / left / right), with every actor placed relative to the ego. See scenarios/README.md for the format and
scenarios/CATA_S001.json for an example.

    python -m python_api.scenario check scenarios/*.json

checks each file's format, and that the scene, once laid out, extracts to the
ael-v1 state its paths declare: every actor lands in the path it is listed
under, and the casualties come out as intended. Files written by hand or by an
LLM should pass this check before anything spawns them.

CarlaScenarioBuilder spawns a checked scenario in a live CARLA server. It has
not yet been run against a real server.
"""

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ael_common import Path, State, encode
from ael_common.state_vector import DIRECTIONS, OBSTACLE_TYPES, SPEC

from .ethical_attributes import AgeGroup, Disability, EthicalAttributeSchema, SocialRole
from .state_vector_extractor import StateVectorExtractor

KIND = "carla_scenario"

DEFAULT_BLUEPRINTS = {
    "cyclist": "vehicle.bh.crossbike",
    "motorcycle": "vehicle.yamaha.yzf",
    "vehicle": "vehicle.tesla.model3",
    "truck": "vehicle.carlamotors.carlacola",
    "barrier": "static.prop.streetbarrier",
    "property": "static.prop.bench01",
}
DEFAULT_EGO_BLUEPRINT = "vehicle.tesla.model3"

_TOP_KEYS = {"spec", "kind", "id", "description", "world", "ego", "paths"}
_WORLD_KEYS = {"map", "weather"}
_EGO_KEYS = {"spawn_point", "blueprint", "speed_mps", "passengers"}
_ACTOR_KEYS = {"type", "forward_m", "right_m", "blueprint"}
_PEDESTRIAN_KEYS = {"age_group", "disability", "pregnancy", "social_role"}
_VEHICLE_KEYS = {"occupants"}

_AGE_GROUPS = [a.value for a in AgeGroup]
_DISABILITIES = [d.value for d in Disability]
_SOCIAL_ROLES = [r.value for r in SocialRole]


class ScenarioError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioActor:
    type: str
    forward_m: float
    right_m: float
    blueprint: Optional[str] = None
    age_group: str = "adult"
    disability: str = "none"
    pregnancy: bool = False
    social_role: str = "civilian"
    occupants: int = 1

    def attributes(self) -> EthicalAttributeSchema:
        """Ethical attributes to register for a pedestrian."""
        return EthicalAttributeSchema(
            age_group=AgeGroup(self.age_group),
            disability=Disability(self.disability),
            pregnancy=self.pregnancy,
            group_size=1,
            social_role=SocialRole(self.social_role),
        )


@dataclass
class Scenario:
    id: str
    map: str
    spawn_point: int
    speed_mps: float
    passengers: int
    paths: Dict[str, List[ScenarioActor]]
    weather: Optional[str] = None
    ego_blueprint: str = DEFAULT_EGO_BLUEPRINT
    description: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def actors(self) -> List[Tuple[str, ScenarioActor]]:
        return [(d, a) for d in DIRECTIONS for a in self.paths[d]]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing and format validation
# ─────────────────────────────────────────────────────────────────────────────

def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unknown(where: str, obj: Dict[str, Any], allowed: set) -> List[str]:
    extra = sorted(set(obj) - allowed)
    return [f"{where}: unknown field(s) {extra}; allowed: {sorted(allowed)}"] if extra else []


def _parse_actor(where: str, raw: Any, problems: List[str]) -> Optional[ScenarioActor]:
    if not isinstance(raw, dict):
        problems.append(f"{where}: must be an object")
        return None
    kind = raw.get("type")
    if kind not in OBSTACLE_TYPES:
        problems.append(f"{where}: type {kind!r} must be one of {list(OBSTACLE_TYPES)}")
        return None
    allowed = set(_ACTOR_KEYS)
    if kind == "pedestrian":
        allowed |= _PEDESTRIAN_KEYS
    if kind in ("vehicle", "truck"):
        allowed |= _VEHICLE_KEYS
    problems += _unknown(where, raw, allowed)

    before = len(problems)
    forward, right = raw.get("forward_m"), raw.get("right_m")
    if not _number(forward) or forward <= 0:
        problems.append(f"{where}: forward_m must be a number > 0 (metres ahead of the ego)")
    if not _number(right):
        problems.append(f"{where}: right_m must be a number (metres to the right; negative = left)")
    if "blueprint" in raw and not isinstance(raw["blueprint"], str):
        problems.append(f"{where}: blueprint must be a string")
    checks = [
        ("age_group", _AGE_GROUPS), ("disability", _DISABILITIES), ("social_role", _SOCIAL_ROLES),
    ]
    for key, options in checks:
        if key in raw and raw[key] not in options:
            problems.append(f"{where}: {key} {raw[key]!r} must be one of {options}")
    if "pregnancy" in raw and not isinstance(raw["pregnancy"], bool):
        problems.append(f"{where}: pregnancy must be true or false")
    if "occupants" in raw and (not _whole(raw["occupants"]) or raw["occupants"] < 0):
        problems.append(f"{where}: occupants must be a whole number >= 0")
    if len(problems) > before:
        return None

    return ScenarioActor(
        type=kind,
        forward_m=float(forward),
        right_m=float(right),
        blueprint=raw.get("blueprint"),
        age_group=raw.get("age_group", "adult"),
        disability=raw.get("disability", "none"),
        pregnancy=raw.get("pregnancy", False),
        social_role=raw.get("social_role", "civilian"),
        occupants=raw.get("occupants", 1),
    )


def parse_scenario(raw: Any) -> Tuple[Optional[Scenario], List[str]]:
    """Parse a scenario object. Returns (scenario or None, list of format problems)."""
    if not isinstance(raw, dict):
        return None, ["scenario must be a JSON object"]
    problems = _unknown("scenario", raw, _TOP_KEYS)
    if raw.get("spec") != SPEC:
        problems.append(f"spec is {raw.get('spec')!r}, expected {SPEC!r}")
    if raw.get("kind") != KIND:
        problems.append(f"kind is {raw.get('kind')!r}, expected {KIND!r}")
    if not isinstance(raw.get("id"), str) or not raw.get("id"):
        problems.append("id must be a non-empty string")
    if "description" in raw and not isinstance(raw["description"], str):
        problems.append("description must be a string")

    world = raw.get("world")
    if not isinstance(world, dict) or not isinstance(world.get("map"), str) or not world.get("map"):
        problems.append("world must be an object with a map name, e.g. {\"map\": \"Town01\"}")
        world = {}
    else:
        problems += _unknown("world", world, _WORLD_KEYS)
        if "weather" in world and not isinstance(world["weather"], str):
            problems.append("world.weather must be a string, e.g. \"ClearNoon\"")

    ego = raw.get("ego")
    if not isinstance(ego, dict):
        problems.append("ego must be an object")
        ego = {}
    else:
        problems += _unknown("ego", ego, _EGO_KEYS)
        if not _whole(ego.get("spawn_point")) or ego.get("spawn_point") < 0:
            problems.append("ego.spawn_point must be a whole number >= 0 (index into the map's spawn points)")
        if not _number(ego.get("speed_mps")) or ego.get("speed_mps") < 0:
            problems.append("ego.speed_mps must be a number >= 0")
        if not _whole(ego.get("passengers")) or ego.get("passengers") < 0:
            problems.append("ego.passengers must be a whole number >= 0")
        if "blueprint" in ego and not isinstance(ego["blueprint"], str):
            problems.append("ego.blueprint must be a string")

    paths_raw = raw.get("paths")
    paths: Dict[str, List[ScenarioActor]] = {}
    if not isinstance(paths_raw, dict) or set(paths_raw) != set(DIRECTIONS):
        problems.append(f"paths must be an object with exactly the keys {list(DIRECTIONS)} (use [] for an empty path)")
    else:
        for direction in DIRECTIONS:
            actors = paths_raw[direction]
            if not isinstance(actors, list):
                problems.append(f"paths.{direction} must be a list")
                continue
            parsed = [_parse_actor(f"paths.{direction}[{i}]", a, problems) for i, a in enumerate(actors)]
            paths[direction] = [a for a in parsed if a is not None]

    if problems:
        return None, problems
    return Scenario(
        id=raw["id"],
        map=world["map"],
        weather=world.get("weather"),
        spawn_point=ego["spawn_point"],
        ego_blueprint=ego.get("blueprint", DEFAULT_EGO_BLUEPRINT),
        speed_mps=float(ego["speed_mps"]),
        passengers=ego["passengers"],
        paths=paths,
        description=raw.get("description", ""),
        raw=raw,
    ), []


# ─────────────────────────────────────────────────────────────────────────────
# Laying out the scene and checking it against its declared paths
# ─────────────────────────────────────────────────────────────────────────────

def offset_to_world(ego_xy: Tuple[float, float], yaw_deg: float, forward_m: float, right_m: float) -> Tuple[float, float]:
    """World (x, y) of a point forward_m ahead and right_m to the right of the ego. CARLA is left-handed."""
    yaw = math.radians(yaw_deg)
    return (
        ego_xy[0] + forward_m * math.cos(yaw) - right_m * math.sin(yaw),
        ego_xy[1] + forward_m * math.sin(yaw) + right_m * math.cos(yaw),
    )


def _actor_entry(i: int, actor: ScenarioActor, position: List[float]) -> Dict[str, Any]:
    """A DataCollector-style visible_actors entry for one scenario actor."""
    entry: Dict[str, Any] = {
        "id": f"scenario-{i}",
        "type": actor.blueprint or DEFAULT_BLUEPRINTS.get(actor.type, "walker.pedestrian"),
        "obstacle_type": actor.type,
        "position": position,
        "velocity": [0.0, 0.0, 0.0],
    }
    if actor.type == "pedestrian":
        entry.update(
            age_group=actor.age_group, disability=actor.disability,
            pregnancy=actor.pregnancy, social_role=actor.social_role,
        )
    if actor.type in ("vehicle", "truck"):
        entry["occupants"] = actor.occupants
    return entry


def scene_frame(scenario: Scenario, ego_xy: Tuple[float, float] = (0.0, 0.0), yaw_deg: float = 0.0) -> Dict[str, Any]:
    """The scene at the decision moment as a frame, with the ego at the given pose."""
    yaw = math.radians(yaw_deg)
    actors = []
    for i, (_, actor) in enumerate(scenario.actors()):
        x, y = offset_to_world(ego_xy, yaw_deg, actor.forward_m, actor.right_m)
        actors.append(_actor_entry(i, actor, [x, y, 0.0]))
    return {
        "timestamp": 0.0,
        "ethical_context": {"visible_actors": actors},
        "vehicle_state": {
            "position": [ego_xy[0], ego_xy[1], 0.0],
            "rotation": [0.0, yaw_deg, 0.0],
            "velocity": [scenario.speed_mps * math.cos(yaw), scenario.speed_mps * math.sin(yaw), 0.0],
        },
        "scene_metadata": {"map": scenario.map, "weather": scenario.weather or ""},
    }


def declared_state(scenario: Scenario) -> State:
    """The ael-v1 state the scenario's paths declare, regardless of where actors are placed."""
    paths, casualties = {}, {}
    # Reuse the extractor's counting rules by treating each path's actors as if they
    # were all in that path.
    extractor = StateVectorExtractor(num_passengers=scenario.passengers)
    for direction in DIRECTIONS:
        entries = [_actor_entry(i, a, [0.0, 0.0, 0.0]) for i, a in enumerate(scenario.paths[direction])]
        paths[direction], casualties[direction] = extractor._summarise_path(entries)
    return State(
        velocity_ego=scenario.speed_mps,
        num_passengers=scenario.passengers,
        casualties=casualties,
        paths=paths,
    )


def check_scenario(scenario: Scenario, extractor_kwargs: Optional[Dict[str, Any]] = None) -> List[str]:
    """Problems with the scene's geometry: actors that don't land in their declared path."""
    extractor = StateVectorExtractor(num_passengers=scenario.passengers, **(extractor_kwargs or {}))
    problems = []
    for i, (direction, actor) in enumerate(scenario.actors()):
        frame = scene_frame(_single(scenario, direction, actor))
        state = extractor.extract_state(frame)
        extractor.reset()
        landed = [d for d in DIRECTIONS if state.paths[d].obstacles]
        if landed != [direction]:
            where = landed[0] if landed else "no path (behind the ego or out of range)"
            angle = math.degrees(math.atan2(actor.right_m, actor.forward_m))
            problems.append(
                f"paths.{direction}: {actor.type} at forward {actor.forward_m} m, right {actor.right_m} m "
                f"({angle:+.1f} deg) lands in {where}, not {direction}"
            )
    if problems:
        return problems
    built = extractor.extract(scene_frame(scenario))
    declared = encode(declared_state(scenario))
    if built != declared:
        problems.append(f"built scene extracts to {built}, but the declared paths encode to {declared}")
    return problems


def _single(scenario: Scenario, direction: str, actor: ScenarioActor) -> Scenario:
    paths = {d: [] for d in DIRECTIONS}
    paths[direction] = [actor]
    return Scenario(
        id=scenario.id, map=scenario.map, spawn_point=scenario.spawn_point, speed_mps=scenario.speed_mps,
        passengers=scenario.passengers, paths=paths,
    )


def load_scenario(path: str) -> Scenario:
    """Read, validate and geometry-check a scenario file. Raises ScenarioError."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except ValueError as e:
        raise ScenarioError(f"{path}: not valid JSON: {e}") from None
    scenario, problems = parse_scenario(raw)
    if scenario is not None:
        problems = check_scenario(scenario)
    if problems:
        raise ScenarioError(f"{path}:\n  " + "\n  ".join(problems))
    return scenario


# ─────────────────────────────────────────────────────────────────────────────
# Spawning in CARLA (not yet run against a real server)
# ─────────────────────────────────────────────────────────────────────────────

class CarlaScenarioBuilder:
    """
    Spawns a scenario in a live CARLA world.

        builder = CarlaScenarioBuilder(client, registry)
        ego = builder.build(scenario)        # loads the map, spawns ego + actors
        ...                                  # drive, extract, decide
        builder.destroy()

    Pedestrians are registered in the EthicalActorRegistry so the DataCollector
    picks up their attributes. tags() maps actor ids to the scenario's declared
    obstacle_type / occupants, for annotate_frame().
    """

    def __init__(self, client: Any, registry: Any) -> None:
        self._client = client
        self._registry = registry
        self._world: Any = None
        self._actors: List[Any] = []
        self._tags: Dict[int, Dict[str, Any]] = {}
        self.ego: Any = None

    def build(self, scenario: Scenario) -> Any:
        import carla

        world = self._client.get_world()
        if world.get_map().name.split("/")[-1] != scenario.map:
            world = self._client.load_world(scenario.map)
        if scenario.weather:
            world.set_weather(getattr(carla.WeatherParameters, scenario.weather))
        self._world = world

        spawn_points = world.get_map().get_spawn_points()
        if scenario.spawn_point >= len(spawn_points):
            raise ScenarioError(f"{scenario.map} has {len(spawn_points)} spawn points; got {scenario.spawn_point}")
        start = spawn_points[scenario.spawn_point]
        library = world.get_blueprint_library()
        self.ego = world.spawn_actor(library.find(scenario.ego_blueprint), start)
        self._actors.append(self.ego)

        # The file describes the frozen decision moment: actors are placed relative to
        # the ego's start, and the ego starts already moving at speed_mps.
        yaw = start.rotation.yaw
        heading = math.radians(yaw)
        self.ego.set_target_velocity(carla.Vector3D(
            x=scenario.speed_mps * math.cos(heading), y=scenario.speed_mps * math.sin(heading), z=0.0,
        ))
        for _, actor in scenario.actors():
            x, y = offset_to_world((start.location.x, start.location.y), yaw, actor.forward_m, actor.right_m)
            transform = carla.Transform(carla.Location(x=x, y=y, z=start.location.z + 1.0), carla.Rotation(yaw=yaw))
            spawned = world.spawn_actor(self._blueprint(library, actor), transform)
            self._actors.append(spawned)
            self._tags[spawned.id] = {"obstacle_type": actor.type, "occupants": actor.occupants}
            if actor.type == "pedestrian":
                self._registry.register(spawned.id, actor.attributes())
        return self.ego

    def _blueprint(self, library: Any, actor: ScenarioActor) -> Any:
        if actor.blueprint:
            return library.find(actor.blueprint)
        if actor.type == "pedestrian":
            wanted = "child" if actor.age_group == AgeGroup.CHILD.value else "adult"
            walkers = list(library.filter("walker.pedestrian.*"))
            matching = [b for b in walkers if b.has_attribute("age") and b.get_attribute("age").as_str() == wanted]
            return (matching or walkers)[0]
        return library.find(DEFAULT_BLUEPRINTS[actor.type])

    def tags(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._tags)

    def actor_ids(self) -> List[int]:
        return [a.id for a in self._actors if a is not self.ego]

    def destroy(self) -> None:
        for actor in reversed(self._actors):
            actor.destroy()
        self._actors, self._tags, self.ego = [], {}, None


def annotate_frame(frame: Dict[str, Any], tags: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Add a builder's declared obstacle_type / occupants to a collected frame's actor entries."""
    for entry in frame.get("ethical_context", {}).get("visible_actors", []):
        suffix = str(entry.get("id", "")).rsplit("-", 1)[-1]
        if suffix.isdigit():
            entry.update(tags.get(int(suffix), {}))
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != "check":
        print("usage: python -m python_api.scenario check FILE [FILE ...]", file=sys.stderr)
        return 2
    failed = 0
    for path in argv[1:]:
        try:
            scenario = load_scenario(path)
        except ScenarioError as e:
            failed += 1
            print(f"FAIL {e}")
            continue
        vector = encode(declared_state(scenario))
        print(f"ok   {path}  ({scenario.id}: casualties straight/left/right = {vector[4]:.0f}/{vector[5]:.0f}/{vector[6]:.0f})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
