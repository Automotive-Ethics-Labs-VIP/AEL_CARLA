from __future__ import annotations

"""
StateVectorExtractor
====================
Converts a DataCollector snapshot frame into an ael-v1 state vector: the 40-D
layout defined in ael-common, which the CATA-200 scenarios and the trained
Ethical Head use. The vector is built through ael_common.encode(), so the output
is ael-v1 by construction. See the ael-common README for what each index means.

How a frame maps onto ael-v1
----------------------------
Paths: each visible actor is placed in the straight, left or right path by its
bearing from the ego heading (ego yaw when the frame has a rotation, otherwise
the velocity direction):

    straight : within ±22.5° of heading
    left     : between 22.5° and 90° to the left
    right    : between 22.5° and 90° to the right

Actors behind the vehicle or beyond max_range_m are ignored.

Obstacle type of each actor, first match wins:

    1. an explicit "obstacle_type" field (one of ael-common's 7 types), so a
       scenario builder can tag e.g. a concrete wall as "barrier"
    2. a CARLA "base_type" field (bicycle, motorcycle, truck, bus, car, van)
    3. the blueprint id in "type": walker.* -> pedestrian, vehicle.* -> by
       model (see _CYCLE_MODELS etc.), static.prop.*barrier* -> barrier,
       other static.prop.* / traffic.* -> property
    4. legacy frames: an entry with ethical attributes is a pedestrian

Vulnerable groups come from pedestrians only: child, elderly, pregnant, and
disabled (wheelchair, cane or blind).

Casualties per path, matching how Team C counted CATA-200:

    pedestrian, cyclist, motorcycle : 1 per actor (the person or rider)
    vehicle, truck                  : the actor's "occupants" field, or
                                      default_vehicle_occupants
    barrier                         : the ego vehicle's passengers (once)
    property                        : 0

lane_position is always 0.0 for now. A real lane offset needs the CARLA map
(Map.get_waypoint), and the CATA-200 training data only ever has 0.

Team B's extra attributes (social role, vulnerability score, group size) have
no slot in ael-v1. They stay in the frame's actor entries.
"""

import math
from typing import Any, Dict, List, Optional

from ael_common import Path, State, encode
from ael_common.state_vector import DIRECTIONS, OBSTACLE_TYPES

from .ethical_attributes import AgeGroup, Disability

_MAX_DETECTION_RANGE_M = 50.0

# CARLA 0.9.13 blueprint ids. Unlisted vehicle.* blueprints count as "vehicle".
_CYCLE_MODELS = {
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
}
_MOTORCYCLE_MODELS = {
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.vespa.zx125",
    "vehicle.yamaha.yzf",
}
_TRUCK_MODELS = {
    "vehicle.carlamotors.carlacola",
    "vehicle.carlamotors.european_hgv",
    "vehicle.carlamotors.firetruck",
    "vehicle.mitsubishi.fusorosa",
}
_BASE_TYPE_TO_OBSTACLE = {
    "bicycle": "cyclist",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "bus": "truck",
    "car": "vehicle",
    "van": "vehicle",
}
_PEOPLE_TYPES = {"pedestrian", "cyclist", "motorcycle"}
_DISABLED = {Disability.WHEELCHAIR.value, Disability.CANE.value, Disability.BLIND.value}


def classify_blueprint(type_id: str) -> Optional[str]:
    """Map a CARLA blueprint id to an ael-v1 obstacle type, or None if not an obstacle."""
    if type_id == "walker" or type_id.startswith("walker."):
        return "pedestrian"
    if type_id.startswith("vehicle."):
        if type_id in _CYCLE_MODELS:
            return "cyclist"
        if type_id in _MOTORCYCLE_MODELS:
            return "motorcycle"
        if type_id in _TRUCK_MODELS:
            return "truck"
        return "vehicle"
    if type_id.startswith("static.prop."):
        return "barrier" if "barrier" in type_id else "property"
    if type_id.startswith("traffic."):
        return "property"
    return None


def classify_actor(actor: Dict[str, Any]) -> Optional[str]:
    """ael-v1 obstacle type for one visible_actors entry, or None to ignore it."""
    explicit = actor.get("obstacle_type")
    if explicit is not None:
        if explicit not in OBSTACLE_TYPES:
            raise ValueError(
                f"actor {actor.get('id')}: obstacle_type {explicit!r} is not one of {OBSTACLE_TYPES}"
            )
        return explicit
    base_type = actor.get("base_type")
    if base_type in _BASE_TYPE_TO_OBSTACLE:
        return _BASE_TYPE_TO_OBSTACLE[base_type]
    type_id = actor.get("type")
    if type_id:
        return classify_blueprint(type_id)
    if actor.get("age_group") is not None:
        return "pedestrian"
    return None


class StateVectorExtractor:
    """
    Stateful converter from DataCollector frames to ael-v1 state vectors.

    Stateful because velocity_delta (index 3) needs the previous speed; call
    reset() between episodes.

    Args:
        num_passengers: Occupants of the ego vehicle (index 1; also the
                        casualties of a barrier path).
        default_vehicle_occupants: People assumed in another vehicle or truck
                        when its actor entry has no "occupants" field.
        straight_half_angle_deg: Half-angle of the straight cone.
        side_max_angle_deg: Outer edge of the left/right cones; beyond it an
                        actor is behind the vehicle and ignored.
        max_range_m: Actors farther than this are ignored.

    Usage::

        extractor = StateVectorExtractor(num_passengers=2)
        for frame in collector.frames():
            state = extractor.extract(frame)   # 40 floats, ael-v1
    """

    def __init__(
        self,
        num_passengers: int = 1,
        default_vehicle_occupants: int = 1,
        straight_half_angle_deg: float = 22.5,
        side_max_angle_deg: float = 90.0,
        max_range_m: float = _MAX_DETECTION_RANGE_M,
    ) -> None:
        self._num_passengers = int(num_passengers)
        self._default_vehicle_occupants = int(default_vehicle_occupants)
        self._straight_half_rad = math.radians(straight_half_angle_deg)
        self._side_max_rad = math.radians(side_max_angle_deg)
        self._max_range_m = max_range_m
        self._prev_speed: Optional[float] = None

    def extract(self, frame: Dict[str, Any]) -> List[float]:
        """Convert one DataCollector snapshot frame into a 40-float ael-v1 vector."""
        return encode(self.extract_state(frame))

    def extract_state(self, frame: Dict[str, Any]) -> State:
        """Like extract(), but returns the decoded ael-common State."""
        vehicle_state = frame.get("vehicle_state", {})
        visible_actors = frame.get("ethical_context", {}).get("visible_actors", [])

        vel = vehicle_state.get("velocity", [0.0, 0.0, 0.0])
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        vel_delta = speed - self._prev_speed if self._prev_speed is not None else 0.0
        self._prev_speed = speed

        pos = vehicle_state.get("position", [0.0, 0.0, 0.0])
        heading = _heading_rad(vehicle_state, speed)

        binned: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DIRECTIONS}
        for actor in visible_actors:
            direction = self._direction_of(actor, pos, heading)
            if direction is not None:
                binned[direction].append(actor)

        paths = {}
        casualties = {}
        for direction, actors in binned.items():
            paths[direction], casualties[direction] = self._summarise_path(actors)

        return State(
            velocity_ego=speed,
            num_passengers=self._num_passengers,
            lane_position=0.0,
            velocity_delta=vel_delta,
            casualties=casualties,
            paths=paths,
        )

    def reset(self) -> None:
        """Reset between episodes so the next velocity_delta starts from 0."""
        self._prev_speed = None

    def _direction_of(self, actor: Dict[str, Any], ego_pos: List[float], heading: float) -> Optional[str]:
        actor_pos = actor.get("position", [0.0, 0.0, 0.0])
        dx = actor_pos[0] - ego_pos[0]
        dy = actor_pos[1] - ego_pos[1]
        if math.hypot(dx, dy) > self._max_range_m:
            return None
        # CARLA is left-handed (y to the right), so a positive relative angle is to the right.
        rel = _normalise_angle(math.atan2(dy, dx) - heading)
        if abs(rel) <= self._straight_half_rad:
            return "straight"
        if abs(rel) <= self._side_max_rad:
            return "right" if rel > 0 else "left"
        return None

    def _summarise_path(self, actors: List[Dict[str, Any]]) -> "tuple[Path, int]":
        obstacles = set()
        vulnerable = set()
        people = 0
        for actor in actors:
            kind = classify_actor(actor)
            if kind is None:
                continue
            obstacles.add(kind)
            if kind in _PEOPLE_TYPES:
                people += 1
            elif kind in ("vehicle", "truck"):
                occupants = actor.get("occupants")
                people += self._default_vehicle_occupants if occupants is None else int(occupants)
            if kind == "pedestrian":
                vulnerable |= _vulnerable_groups(actor)
        if "barrier" in obstacles:
            people += self._num_passengers
        return Path(obstacles, vulnerable), people


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _heading_rad(vehicle_state: Dict[str, Any], speed: float) -> float:
    """Ego heading: yaw from rotation [pitch, yaw, roll] (degrees) if present, else velocity direction."""
    rotation = vehicle_state.get("rotation")
    if rotation is not None:
        return math.radians(rotation[1])
    vel = vehicle_state.get("velocity", [0.0, 0.0, 0.0])
    return math.atan2(vel[1], vel[0]) if speed > 0.01 else 0.0


def _vulnerable_groups(actor: Dict[str, Any]) -> set:
    groups = set()
    if actor.get("age_group") == AgeGroup.CHILD.value:
        groups.add("child")
    if actor.get("age_group") == AgeGroup.ELDERLY.value:
        groups.add("elderly")
    if actor.get("pregnancy"):
        groups.add("pregnant")
    if actor.get("disability") in _DISABLED:
        groups.add("disabled")
    return groups


def _normalise_angle(angle: float) -> float:
    """Wrap angle into (−π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle
