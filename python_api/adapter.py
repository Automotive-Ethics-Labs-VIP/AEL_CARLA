from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import time
import math
import uuid
from .ethical_attributes import AgeGroup
import random

@runtime_checkable
class SimulatorAdapter(Protocol):
    """
    Typed protocol defining every simulator call the ethical extensions need.

    CARLAAdapter implements this against a live carla.Client + carla.World.
    MockAdapter implements this with pure-Python synthetic data for testing.

    Any object satisfying this protocol is accepted anywhere an adapter is
    expected — no isinstance() checks or base-class inheritance required.
    """

    def get_world(self) -> Any:
        """Return the underlying world object (carla.World or mock equivalent)."""
        ...

    def get_spawn_points(self) -> List[Any]:
        """Return a list of valid pedestrian spawn transforms."""
        ...

    def get_blueprint_library(self) -> Any:
        """Return the blueprint library so callers can filter walker blueprints."""
        ...

    def make_spawn_command(self, blueprint: Any, spawn_point: Any) -> Any:
        """
        Build a single spawn command suitable for passing to spawn_walkers_batch().

        CARLAAdapter returns a carla.command.SpawnActor instance.
        MockAdapter returns a plain dict.

        Keeping command construction inside the adapter means spawn_walkers.py
        never has to import carla directly.
        """
        ...

    def spawn_walker(self, spawn_point: Any, blueprint: Any) -> Optional[int]:
        """
        Spawn a single walker and return its actor_id, or None on failure.
        """
        ...

    def spawn_walkers_batch(self, commands: List[Any]) -> List[Any]:
        """
        Execute a batch of spawn commands in a single RPC call.

        Returns a list of response objects. Each response has:
            .actor_id : int   — ID of the spawned actor (valid if .error is falsy)
            .error    : str   — non-empty string on failure, empty/None on success
        """
        ...

    def get_actor(self, actor_id: int) -> Optional[Any]:
        """Return the actor with the given ID, or None if it doesn't exist."""
        ...

    def destroy_actor(self, actor_id: int) -> bool:
        """Destroy the actor with the given ID. Returns True on success."""
        ...

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Capture the current simulation state and return it as a dict matching
        the streaming data schema:

            {
                "timestamp":       float,
                "sensor_bundle":   {"rgb": str, "depth": str, "lidar": str},
                "ethical_context": {"visible_actors": [...]},
                "vehicle_state":   {"position": [...], "velocity": [...], "controls": {...}},
                "scene_metadata":  {"weather": str, "map": str},
            }

        sensor_bundle values are base64-encoded bytes or empty strings.
        ethical_context.visible_actors is populated by DataCollector using the
        registry — the adapter returns an empty list and collector fills it in.
        """
        ...

    def tick(self) -> None:
        """Advance the simulation by one tick (synchronous mode only)."""
        ...


class CARLAAdapter:
    """
    Concrete adapter over carla.Client and carla.World.

    All carla.* calls are confined to this class. Nothing else in the package
    imports carla directly.

    Args:
        client:      carla.Client connected to the CARLA server.
        world:       carla.World retrieved from the client.
        ego_vehicle: Optional carla.Actor for the ego vehicle. Required for
                     vehicle_state in get_snapshot(). Can be set later via
                     set_ego_vehicle().
    """

    def __init__(self, client: Any, world: Any, ego_vehicle: Optional[Any] = None) -> None:
        self._client      = client
        self._world       = world
        self._ego_vehicle = ego_vehicle
        self._map         = None
        self._destroyed_ids: set = set()

    def set_ego_vehicle(self, vehicle: Any) -> None:
        """Attach or replace the ego vehicle used in get_snapshot()."""
        self._ego_vehicle = vehicle


    def get_world(self) -> Any:
        return self._world

    def get_spawn_points(self) -> List[Any]:
        return self._world.get_map().get_spawn_points()

    def get_blueprint_library(self) -> Any:
        return self._world.get_blueprint_library()

    def make_spawn_command(self, blueprint: Any, spawn_point: Any) -> Any:
        import carla
        return carla.command.SpawnActor(blueprint, spawn_point)

    def spawn_walker(self, spawn_point: Any, blueprint: Any) -> Optional[int]:
        actor = self._world.try_spawn_actor(blueprint, spawn_point)
        return actor.id if actor is not None else None

    def spawn_walkers_batch(self, commands: List[Any]) -> List[Any]:
        # do_tick=False: caller controls when the world ticks (synchronous mode)
        return self._client.apply_batch_sync(commands, do_tick=False)

    def get_actor(self, actor_id: int) -> Optional[Any]:
        if actor_id in self._destroyed_ids:
            return None
        return self._world.get_actor(actor_id)

    def destroy_actor(self, actor_id: int) -> bool:
        actor = self._world.get_actor(actor_id)
        if actor is not None:
            result = actor.destroy()
            if result:
                self._destroyed_ids.add(actor_id) # ← track it
            return result
        return False


    def get_snapshot(self) -> Dict[str, Any]:
        """
        Capture a world snapshot.

        sensor_bundle is intentionally empty here — sensors are async callbacks
        managed by DataCollector, which merges their latest frames into the
        snapshot before publishing.  ethical_context.visible_actors is also
        filled in by DataCollector after it queries the registry.
        """
        world_snapshot = self._world.get_snapshot()
        timestamp: float = world_snapshot.timestamp.elapsed_seconds

        vehicle_state: Dict[str, Any] = {}
        if self._ego_vehicle is not None:
            transform = self._ego_vehicle.get_transform()
            velocity  = self._ego_vehicle.get_velocity()
            control   = self._ego_vehicle.get_control()
            vehicle_state = {
                "position": [
                    transform.location.x,
                    transform.location.y,
                    transform.location.z,
                ],
                "velocity": [velocity.x, velocity.y, velocity.z],
                "controls": {
                    "throttle": control.throttle,
                    "steer":    control.steer,
                    "brake":    control.brake,
                },
            }

        weather     = self._world.get_weather()

        if self._map is None:
            self._map = self._world.get_map()

        world_map   = self._map

        scene_metadata = {
            "weather": str(weather),
            "map":     world_map.name.split("/")[-1],
        }

        return {
            "timestamp":       timestamp,
            "sensor_bundle":   {"rgb": "", "depth": "", "lidar": ""},
            "ethical_context": {"visible_actors": []},
            "vehicle_state":   vehicle_state,
            "scene_metadata":  scene_metadata,
        }

    def tick(self) -> None:
        self._world.tick()

class MockAdapter:
    """
    Drop-in replacement for CARLAAdapter that requires no CARLA installation.

    Generates deterministic synthetic data so tests are reproducible.
    Satisfies the SimulatorAdapter protocol exactly — all the same methods,
    same return shapes, same field names.

    Usage:
        from python_api.adapter import MockAdapter
        adapter = MockAdapter(num_spawn_points=20)
        spawner = EthicalWalkerSpawner(adapter, registry)
    """

    def __init__(self, num_spawn_points: int = 50) -> None:
        self._num_spawn_points = num_spawn_points
        self._actors: Dict[int, Dict[str, Any]] = {}
        self._destroyed_ids = set()
        self._tick_count = 0
        self._sim_time   = 0.0


    def get_world(self) -> None:
        return None  # no real world object in mock

    def get_spawn_points(self) -> List[Dict[str, float]]:
        """Return synthetic spawn point dicts spread on a grid."""
        cols = math.ceil(math.sqrt(self._num_spawn_points))
        return [
            {
                "x": float((i % cols) * 2),
                "y": float((i // cols) * 2),
                "z": 0.0,
            }
            for i in range(self._num_spawn_points)
        ]
    def get_blueprint_library(self) -> "_MockBlueprintLibrary":
        return _MockBlueprintLibrary()


    def make_spawn_command(self, blueprint: Any, spawn_point: Any) -> Dict[str, Any]:
        """Return a plain dict that spawn_walkers_batch() can consume."""
        return {"blueprint": blueprint, "spawn_point": spawn_point}

    def spawn_walker(self, spawn_point: Any, blueprint: Any) -> int:
        actor_id = uuid.uuid4().int
        self._actors[actor_id] = _make_mock_actor(actor_id)
        return actor_id

    def spawn_walkers_batch(self, commands: List[Any]) -> List["_MockBatchResponse"]:
        return [
        _MockBatchResponse(
            actor_id=self.spawn_walker(cmd["spawn_point"], cmd["blueprint"]),
            error=None,
        )
        for cmd in commands
    ]

    def get_actor(self, actor_id: int) -> Optional[Dict[str, Any]]:
        if actor_id in self._destroyed_ids:
            return None
        return self._actors.get(actor_id)

    def destroy_actor(self, actor_id: int) -> bool:
        if actor_id in self._actors:
            del self._actors[actor_id]
            self._destroyed_ids.add(actor_id)
            return True
        return False


    def get_snapshot(self) -> Dict[str, Any]:
        self._tick_count += 1
        self._sim_time   += 0.05
        
        return {
            "timestamp": time.monotonic(),
            "sensor_bundle": {
                "rgb":   "",
                "depth": "",
                "lidar": "",
            },
            "ethical_context": {
                "visible_actors": []   # filled in by DataCollector
            },
            "vehicle_state": {
                "position": [0.0, 0.0, 0.0],
                "velocity": [5.0, 0.0, 0.0],
                "controls": {"throttle": 0.5, "steer": 0.0, "brake": 0.0},
            },
            "scene_metadata": {
                # the metadata is fixed in MockAdapter, but in CARLAAdapter, they will change depending on the map and the weather in the map
                "weather": "clear_noon",
                "map":     "MockTown", 
            },
        }

    def tick(self) -> None:
        self._tick_count += 1
        self._sim_time   += 0.05


    @property
    def actor_count(self) -> int:
        """Convenience property for assertions in tests."""
        return len(self._actors)

    @property
    def tick_count(self) -> int:
        return self._tick_count


def _make_mock_actor(actor_id: int) -> Dict[str, Any]:
    return {
        "id":       actor_id,
        "type":     "walker",
        "position": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
    }


class _MockBatchResponse:
    """Mirrors the shape of carla.command.Response for batch spawn results."""

    def __init__(self, actor_id: int, error: Optional[str]) -> None:
        self.actor_id = actor_id
        self.error    = error  # None or empty string = success, non-empty = failure


class _MockBlueprintLibrary:
    """Minimal stand-in for carla.BlueprintLibrary."""

    _GENDERS = ["male", "female"]
    _AGES = [a.value for a in AgeGroup]

    def filter(self, pattern: str) -> List["_MockBlueprint"]:
        """Return synthetic blueprints matching a glob pattern (pattern ignored in mock)."""
        return [
        _MockBlueprint(
            f"walker.pedestrian.{i:04d}",
            random.choice(self._GENDERS)
        )
        for i in range(6)
    ]


class _MockBlueprint:
    """Mirrors carla.ActorBlueprint — supports .get_attribute('gender').as_str()."""

    def __init__(self, blueprint_id: str, gender: str) -> None:
        self.id          = blueprint_id
        self._attributes = {"gender": gender}

    def get_attribute(self, key: str) -> "_MockAttribute":
        return _MockAttribute(self._attributes.get(key, ""))


class _MockAttribute:
    """Mirrors carla.ActorAttribute — supports .as_str()."""

    def __init__(self, value: str) -> None:
        self._value = value

    def as_str(self) -> str:
        return self._value
