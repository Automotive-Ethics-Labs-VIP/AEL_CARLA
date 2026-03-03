from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
import os

from .actor_registry import EthicalActorRegistry
from .adapter import SimulatorAdapter
from .stream.server import StreamServer


class DataCollector:
    """
    Assembles snapshot frames enriched with ethical metadata and publishes
    them to the StreamServer at a fixed rate (default 20 Hz).

    Two modes of operation:

    1. Manual (no background thread):
       Call collect(visible_actor_ids=[...]) yourself each tick.
       Useful for synchronous tests or when you want explicit control.

    2. Automatic 20 Hz loop (recommended for live simulation):
       Call set_visible_actors([...]) from your simulation loop every tick.
       Call start() once to launch the background collection thread.
       The thread fires every 50 ms regardless of simulation FPS, reads the
       latest visible actor list, assembles a snapshot, and publishes it.
       Call stop() (or use as a context manager) to shut down cleanly.

       Example:
           collector = DataCollector(adapter, registry, server)
           collector.start(hz=20.0)

           # In the simulation loop (runs at ~30 FPS):
           while True:
               adapter.tick()
               collector.set_visible_actors(get_actors_in_camera_range())

           collector.stop()

       Or as a context manager:
           with DataCollector(adapter, registry, server) as collector:
               collector.start(hz=20.0)
               while True:
                   adapter.tick()
                   collector.set_visible_actors(...)

    Args:
        adapter:  Any SimulatorAdapter implementation (CARLAAdapter or MockAdapter).
        registry: EthicalActorRegistry tracking spawned walker attributes.
        server:   StreamServer to publish frames to. Pass None to collect
                  locally without streaming.
        node_id:  Node identifier prefix for distributed mode (Sprint 3).
                  Actor IDs in ethical_context are formatted as
                  "{node_id}-{actor_id}" to guarantee uniqueness across nodes.
    """

    def __init__(
        self,
        adapter: SimulatorAdapter,
        registry: EthicalActorRegistry,
        server: Optional[StreamServer] = None,
        node_id: str = "n0",
    ) -> None:
        self._adapter  = adapter
        self._registry = registry
        self._server   = server
        self._node_id  = node_id

        # Shared state between the simulation loop and the collection thread.
        # The sim loop writes via set_visible_actors(); the collection thread
        # reads it each cycle. Protected by _actors_lock.
        self._visible_actor_ids: Optional[List[int]] = None
        self._actors_lock = threading.Lock()

        # Background thread state.
        self._thread:  Optional[threading.Thread] = None
        self._running: bool = False

        # Running stats - accessible for monitoring and tests.
        self.frames_collected     = 0
        self.actors_missing_attrs = 0  # visible actors absent from the registry
        self.frames_late          = 0  # collection cycles that ran over their interval

    def start(self, hz: float = 20.0) -> None:
        """
        Launch the background collection thread at the given rate.

        Args:
            hz: Target collection frequency in Hz. Default is 20.0.
                Must be > 0.

        Raises:
            ValueError:   If hz <= 0.
            RuntimeError: If the collector is already running.
        """
        if hz <= 0:
            raise ValueError(f"hz must be > 0, got {hz}")
        if self._running:
            raise RuntimeError("DataCollector is already running. Call stop() first.")

        self._interval = 1.0 / hz
        self._running  = True
        self._pid = os.getpid()
        self._thread   = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="data-collector",
        )
        self._thread.start()
        self._thread_id = self._thread.ident

    def stop(self) -> None:
        """
        Signal the background thread to exit and wait for it to finish.

        Safe to call even if start() was never called.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_visible_actors(self, actor_ids: List[int]) -> None:
        """
        Update the list of actor IDs currently visible to the ego vehicle.

        Called by the simulation loop every tick. The background collection
        thread picks up the latest list on its next 50 ms cycle. Writing a
        new list between two collection cycles is fine — the thread always
        reads the most recent value, so it naturally tracks the simulation
        state without queuing up stale frames.

        Args:
            actor_ids: List of CARLA actor IDs in the ego vehicle's sensor range.
        """
        with self._actors_lock:
            self._visible_actor_ids = list(actor_ids)

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "DataCollector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def collect(self, visible_actor_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Capture one snapshot, enrich it with ethical metadata, and publish it.

        In automatic mode you don't need to call this directly — the background
        thread calls it for you. It remains public so you can drive collection
        manually in tests or synchronous scripts.

        Args:
            visible_actor_ids: Actor IDs to include in ethical_context. Pass
                               None to emit a snapshot with an empty actor list
                               (e.g. during warm-up ticks before actors spawn).

        Returns:
            The fully assembled snapshot dict that was published.
        """
        snapshot = self._adapter.get_snapshot()

        if visible_actor_ids is not None:
            snapshot["ethical_context"]["visible_actors"] = (
                self._build_actor_context(visible_actor_ids)
            )

        if self._server is not None:
            self._server.publish(snapshot)

        self.frames_collected += 1
        return snapshot


    def _collection_loop(self) -> None:
        """
        Fire collect() at the configured rate using a drift-corrected sleep.

        Timing strategy:
          - Record wall time before collect().
          - Subtract elapsed time from the target interval before sleeping.
          - This keeps the long-term average rate accurate even if individual
            calls take a variable amount of time.
          - If a single cycle takes longer than the full interval (overrun),
            skip the sleep entirely, increment frames_late, and continue
            immediately so we don't fall further behind.
        """
        self._thread_native_id = threading.get_native_id()
        while self._running:
            cycle_start = time.monotonic()

            # Snapshot the visible actor list under lock so the sim loop can
            # keep writing to it concurrently without us seeing a torn update.
            with self._actors_lock:
                actor_ids = (
                    list(self._visible_actor_ids)
                    if self._visible_actor_ids is not None
                    else None
                )

            if actor_ids is None:
                time.sleep(self._interval)
                continue

            self.collect(visible_actor_ids=actor_ids)

            elapsed    = time.monotonic() - cycle_start
            sleep_time = self._interval - elapsed

            if sleep_time <= 0:
                # collect() took longer than the interval, log it and carry on.
                self.frames_late += 1
            else:
                time.sleep(sleep_time)

    def _build_actor_context(self, actor_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Build the ethical_context.visible_actors list for a snapshot frame.

        For each actor ID:
          - Retrieve ethical attributes from the registry.
          - Retrieve position and velocity from the adapter.
          - Combine into the schema expected by the training pipeline.

        Actors present in visible_actor_ids but absent from the registry
        (e.g. vehicles or non-pedestrian actors) are included with null
        attribute fields so the training pipeline always receives a complete
        actor list and can filter as needed.
        """
        result: List[Dict[str, Any]] = []

        for actor_id in actor_ids:
            attrs = self._registry.get_attributes(actor_id)

            if attrs is None:
                self.actors_missing_attrs += 1

            actor_obj = self._adapter.get_actor(actor_id)
            position, velocity = _extract_kinematics(actor_obj)

            entry: Dict[str, Any] = {
                # Node-prefixed ID guarantees uniqueness across distributed nodes.
                "id":                  f"{self._node_id}-{actor_id}",
                "age_group":           attrs.age_group.value       if attrs else None,
                "disability":          attrs.disability.value      if attrs else None,
                "social_role":         attrs.social_role.value     if attrs else None,
                "vulnerability_score": attrs.vulnerability_score   if attrs else None,
                "position":            position,
                "velocity":            velocity,
            }
            result.append(entry)

        return result


def _extract_kinematics(actor_obj: Any) -> tuple[List[float], List[float]]:
    """
    Extract [x, y, z] position and velocity from a CARLA actor or mock dict.

    Live CARLA actors have .get_transform() and .get_velocity() methods.
    """
    if actor_obj is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    if hasattr(actor_obj, "get_transform"):
        t = actor_obj.get_transform()
        v = actor_obj.get_velocity()
        return (
            [t.location.x, t.location.y, t.location.z],
            [v.x, v.y, v.z],
        )

    if isinstance(actor_obj, dict):
        return (
            list(actor_obj.get("position", [0.0, 0.0, 0.0])),
            list(actor_obj.get("velocity", [0.0, 0.0, 0.0])),
        )

    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]