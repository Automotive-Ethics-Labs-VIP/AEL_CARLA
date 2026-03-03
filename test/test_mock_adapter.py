
from __future__ import annotations

import threading
import time
from typing import List

import pytest

from python_api.adapter         import MockAdapter
from python_api.actor_registry  import EthicalActorRegistry
from python_api.collector       import DataCollector
from python_api                 import StreamClient
from python_api                 import StreamServer

_MOCK_PORT_BASE = 19400


@pytest.fixture()
def mock_adapter() -> MockAdapter:
    return MockAdapter(num_spawn_points=50)


@pytest.fixture()
def mock_registry() -> EthicalActorRegistry:
    return EthicalActorRegistry()


@pytest.fixture()
def mock_spawner(mock_adapter, mock_registry):
    from python_api.spawn_walkers import EthicalWalkerSpawner
    return EthicalWalkerSpawner(mock_adapter, mock_registry)

def _spawn_and_register(adapter: MockAdapter,
                        registry: EthicalActorRegistry,
                        n: int) -> List[int]:
    """
    Spawn N walkers via the mock adapter and register ethical attributes for
    each one so they are 'known' to the collector.  Returns list of actor IDs.
    """
    from python_api.ethical_attributes import (
        EthicalAttributeSchema, AgeGroup, Disability, SocialRole,
    )
    bp = adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
    spawn_pts = adapter.get_spawn_points()
    actor_ids: List[int] = []

    for i in range(n):
        aid = adapter.spawn_walker(spawn_pts[i % len(spawn_pts)], bp)
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN,
        )
        registry.register(aid, attrs)
        actor_ids.append(aid)

    return actor_ids

class TestMockAdapter:

    def test_get_snapshot_returns_correct_schema(self, mock_adapter):
        snap = mock_adapter.get_snapshot()
        for key in ("timestamp", "sensor_bundle", "ethical_context",
                    "vehicle_state", "scene_metadata"):
            assert key in snap, f"Missing top-level key: {key}"
        assert "visible_actors" in snap["ethical_context"]
        assert "rgb"   in snap["sensor_bundle"]
        assert "depth" in snap["sensor_bundle"]
        assert "lidar" in snap["sensor_bundle"]

    def test_get_snapshot_timestamp_advances(self, mock_adapter):
        """Each get_snapshot() call should return a strictly increasing timestamp."""
        snap1 = mock_adapter.get_snapshot()
        time.sleep(0.01)   # small real-time gap
        snap2 = mock_adapter.get_snapshot()
        assert snap2["timestamp"] > snap1["timestamp"], (
            "MockAdapter timestamp did not advance between calls"
        )

    def test_scene_metadata_fields_present(self, mock_adapter):
        snap = mock_adapter.get_snapshot()
        assert "weather" in snap["scene_metadata"]
        assert "map"     in snap["scene_metadata"]

    def test_vehicle_state_fields_present(self, mock_adapter):
        snap = mock_adapter.get_snapshot()
        vs = snap["vehicle_state"]
        assert "position" in vs
        assert "velocity" in vs
        assert "controls" in vs
        assert len(vs["position"]) == 3

    def test_get_spawn_points_returns_list(self, mock_adapter):
        pts = mock_adapter.get_spawn_points()
        assert isinstance(pts, list)
        assert len(pts) > 0

    def test_spawn_walker_returns_int(self, mock_adapter):
        bp = mock_adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
        sp = mock_adapter.get_spawn_points()[0]
        aid = mock_adapter.spawn_walker(sp, bp)
        assert isinstance(aid, int)
        assert aid > 0

    def test_spawn_walker_actor_retrievable(self, mock_adapter):
        bp = mock_adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
        sp = mock_adapter.get_spawn_points()[0]
        aid = mock_adapter.spawn_walker(sp, bp)
        assert mock_adapter.get_actor(aid) is not None

    def test_destroy_actor_removes_it(self, mock_adapter):
        bp = mock_adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
        sp = mock_adapter.get_spawn_points()[0]
        aid = mock_adapter.spawn_walker(sp, bp)

        assert mock_adapter.get_actor(aid) is not None
        result = mock_adapter.destroy_actor(aid)
        assert result is True
        assert mock_adapter.get_actor(aid) is None

    def test_destroy_nonexistent_actor_returns_false(self, mock_adapter):
        assert mock_adapter.destroy_actor(99999999) is False

    def test_spawn_walkers_batch(self, mock_adapter):
        bp  = mock_adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
        sps = mock_adapter.get_spawn_points()[:5]
        commands = [mock_adapter.make_spawn_command(bp, sp) for sp in sps]
        responses = mock_adapter.spawn_walkers_batch(commands)
        assert len(responses) == 5
        for resp in responses:
            assert not resp.error
            assert resp.actor_id > 0

    def test_actor_count_property(self, mock_adapter):
        initial = mock_adapter.actor_count
        bp = mock_adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
        sp = mock_adapter.get_spawn_points()[0]
        mock_adapter.spawn_walker(sp, bp)
        assert mock_adapter.actor_count == initial + 1

    def test_tick_increments_tick_count(self, mock_adapter):
        before = mock_adapter.tick_count
        mock_adapter.tick()
        assert mock_adapter.tick_count == before + 1


class TestDataCollectorMock:

    def test_collector_manual_collect(self, mock_adapter, mock_registry):
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=10)

        collector = DataCollector(mock_adapter, mock_registry, server=None, node_id="n0")
        snap = collector.collect(visible_actor_ids=actor_ids)

        assert snap is not None
        assert len(snap["ethical_context"]["visible_actors"]) == 10
        assert collector.frames_collected == 1

    def test_collector_actor_ids_are_node_prefixed(self, mock_adapter, mock_registry):
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=5)

        collector = DataCollector(mock_adapter, mock_registry, node_id="n7")
        snap = collector.collect(visible_actor_ids=actor_ids)

        for actor in snap["ethical_context"]["visible_actors"]:
            assert actor["id"].startswith("n7-"), (
                f"Actor ID '{actor['id']}' not prefixed with 'n7-'"
            )

    def test_collector_missing_attrs_counter_and_null_fields(
        self, mock_adapter, mock_registry
    ):
        """
        Two improvements over the original test:

        1. actors_missing_attrs is incremented once per unregistered actor
           (counter semantics check).
        2. The snapshot still contains both actors — they appear with all
           ethical attribute fields set to None (null-field contract check).

        This validates the docstring guarantee on _build_actor_context:
        "Actors absent from the registry are included with null attribute fields
        so the training pipeline always receives a complete actor list."
        """
        collector = DataCollector(mock_adapter, mock_registry, server=None)
        fake_ids = [888888, 999999]
        snap = collector.collect(visible_actor_ids=fake_ids)

        # Counter
        assert collector.actors_missing_attrs == 2

        # Null-field contract — actors still present in the snapshot
        actors = snap["ethical_context"]["visible_actors"]
        assert len(actors) == 2

        for actor in actors:
            assert actor["age_group"]           is None, f"Expected null age_group for {actor['id']}"
            assert actor["disability"]          is None, f"Expected null disability for {actor['id']}"
            assert actor["social_role"]         is None, f"Expected null social_role for {actor['id']}"
            assert actor["vulnerability_score"] is None, f"Expected null vulnerability_score for {actor['id']}"

    def test_collector_mixed_known_and_unknown_actors(
        self, mock_adapter, mock_registry
    ):
        """
        When some actors are registered and some are not, known actors must have
        full attribute fields while unknown actors must have null fields.
        The counter must only count the unknown ones.
        """
        known_ids   = _spawn_and_register(mock_adapter, mock_registry, n=3)
        unknown_ids = [777001, 777002]

        collector = DataCollector(mock_adapter, mock_registry, server=None, node_id="nx")
        snap = collector.collect(visible_actor_ids=known_ids + unknown_ids)

        actors_by_id = {a["id"]: a for a in snap["ethical_context"]["visible_actors"]}

        # Known actors have real attribute values
        for aid in known_ids:
            actor = actors_by_id[f"nx-{aid}"]
            assert actor["age_group"]   is not None
            assert actor["disability"]  is not None
            assert actor["social_role"] is not None

        # Unknown actors have null values
        for aid in unknown_ids:
            actor = actors_by_id[f"nx-{aid}"]
            assert actor["age_group"]   is None
            assert actor["disability"]  is None
            assert actor["social_role"] is None

        assert collector.actors_missing_attrs == 2

    def test_collector_empty_visible_actors(self, mock_adapter, mock_registry):
        """collect(visible_actor_ids=[]) should produce a valid frame with no actors."""
        collector = DataCollector(mock_adapter, mock_registry)
        snap = collector.collect(visible_actor_ids=[])
        assert snap["ethical_context"]["visible_actors"] == []
        assert collector.frames_collected == 1

    def test_collector_collect_with_none_produces_empty_context(
        self, mock_adapter, mock_registry
    ):
        """collect(None) is the warm-up path — ethical_context list is not touched."""
        collector = DataCollector(mock_adapter, mock_registry)
        snap = collector.collect(visible_actor_ids=None)
        # The adapter returns [] as the default; collector must not modify it
        assert snap["ethical_context"]["visible_actors"] == []
        assert collector.frames_collected == 1

    def test_collector_start_raises_if_already_running(
        self, mock_adapter, mock_registry
    ):
        collector = DataCollector(mock_adapter, mock_registry)
        collector.start(hz=20.0)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                collector.start(hz=20.0)
        finally:
            collector.stop()

    def test_collector_start_raises_on_invalid_hz(self, mock_adapter, mock_registry):
        collector = DataCollector(mock_adapter, mock_registry)
        with pytest.raises(ValueError, match="hz must be > 0"):
            collector.start(hz=0)
        with pytest.raises(ValueError, match="hz must be > 0"):
            collector.start(hz=-10.0)

    def test_collector_context_manager_stops_cleanly(
        self, mock_adapter, mock_registry
    ):
        with DataCollector(mock_adapter, mock_registry) as collector:
            collector.start(hz=20.0)
            time.sleep(0.2)
        assert not collector.is_running

    def test_collector_stop_is_idempotent(self, mock_adapter, mock_registry):
        """stop() called without start() must not raise."""
        collector = DataCollector(mock_adapter, mock_registry)
        collector.stop()   # first call — never started
        collector.stop()   # second call — still safe

    def test_collector_hz_rate_1_second(self, mock_adapter, mock_registry):
        """
        Run at 20 Hz for 1 second and verify ~20 frames are emitted.

        Tolerance is ±8 frames (40%) — wider than the CARLA 3-second test to
        account for OS scheduling jitter on CI machines. The key property being
        tested is that the background thread fires repeatedly at approximately
        the right rate, not that it hits exactly 20.0000 Hz.

        For tighter rate validation, see test_collector_frame_rate_proportional.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=5)
        collector = DataCollector(mock_adapter, mock_registry, server=None)
        collector.start(hz=20.0)
        collector.set_visible_actors(actor_ids)
        time.sleep(1.0)
        collector.stop()

        assert 12 <= collector.frames_collected <= 28, (
            f"Expected ~20 frames in 1 s at 20 Hz, got {collector.frames_collected}"
        )

    def test_collector_frame_rate_proportional(self, mock_adapter, mock_registry):
        """
        Verify 40 Hz emits roughly twice as many frames as 20 Hz over the same
        wall-clock duration. This validates the proportional relationship between
        hz and output rate without relying on absolute counts.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=5)

        def run_and_count(hz: float, duration: float) -> int:
            c = DataCollector(mock_adapter, mock_registry, server=None)
            c.start(hz=hz)
            c.set_visible_actors(actor_ids)
            time.sleep(duration)
            c.stop()
            return c.frames_collected

        count_20hz = run_and_count(20.0, 0.5)
        count_40hz = run_and_count(40.0, 0.5)

        # 40 Hz should produce at least 1.5x as many frames as 20 Hz
        assert count_40hz >= count_20hz * 1.5, (
            f"40 Hz ({count_40hz} frames) is not proportionally faster than "
            f"20 Hz ({count_20hz} frames)"
        )

    def test_collector_no_late_frames_under_mock_load(
        self, mock_adapter, mock_registry
    ):
        """
        MockAdapter.get_snapshot() is near-instantaneous so no frames should
        run over their 50 ms interval.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=20)
        collector = DataCollector(mock_adapter, mock_registry, server=None)
        collector.start(hz=20.0)
        collector.set_visible_actors(actor_ids)
        time.sleep(1.0)
        collector.stop()

        assert collector.frames_late == 0, (
            f"{collector.frames_late} frames ran over the 50 ms interval with mock adapter"
        )

    def test_collector_set_visible_actors_thread_safe(
        self, mock_adapter, mock_registry
    ):
        """
        Writing set_visible_actors() from a separate thread while the collector
        is running must not raise or corrupt state.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=10)
        collector = DataCollector(mock_adapter, mock_registry, server=None)
        collector.start(hz=20.0)

        errors: List[Exception] = []

        def spam_set():
            try:
                for _ in range(200):
                    collector.set_visible_actors(actor_ids)
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        writer = threading.Thread(target=spam_set)
        writer.start()
        time.sleep(0.5)
        collector.stop()
        writer.join(timeout=2.0)

        assert not errors, f"Thread safety violation: {errors}"

class TestStreamingEndToEndMock:

    def test_full_pipeline_frame_received_by_client(
        self, mock_adapter, mock_registry
    ):
        """
        MockAdapter → DataCollector → StreamServer → StreamClient end-to-end.
        Validates the entire pipeline without CARLA.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=10)
        port      = _MOCK_PORT_BASE + 0

        server    = StreamServer(host="127.0.0.1", port=port)
        server.start()

        collector = DataCollector(mock_adapter, mock_registry, server=server, node_id="n0")
        collector.start(hz=20.0)
        collector.set_visible_actors(actor_ids)

        try:
            with StreamClient(host="127.0.0.1", port=port, timeout=5.0) as client:
                time.sleep(0.1)
                frame = next(client.frames())

            assert "ethical_context" in frame
            actors = frame["ethical_context"]["visible_actors"]
            assert len(actors) == 10

            for actor in actors:
                assert actor["id"].startswith("n0-")
                assert actor["age_group"]           is not None
                assert actor["vulnerability_score"] is not None

        finally:
            collector.stop()
            server.stop()

    def test_stream_frame_matches_snapshot_schema(
        self, mock_adapter, mock_registry
    ):
        """Every streamed frame must contain all top-level schema keys."""
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=5)
        port      = _MOCK_PORT_BASE + 1

        server    = StreamServer(host="127.0.0.1", port=port)
        server.start()

        collector = DataCollector(mock_adapter, mock_registry, server=server)
        collector.start(hz=20.0)
        collector.set_visible_actors(actor_ids)

        try:
            with StreamClient(host="127.0.0.1", port=port, timeout=5.0) as client:
                time.sleep(0.1)
                frame = next(client.frames())

            for key in ("timestamp", "sensor_bundle", "ethical_context",
                        "vehicle_state", "scene_metadata"):
                assert key in frame, f"Missing key in streamed frame: {key}"
        finally:
            collector.stop()
            server.stop()

    def test_no_frames_dropped_at_20hz_with_mock_actors(
        self, mock_adapter, mock_registry
    ):
        """
        At 20 Hz with 100 mock actors, the stream buffer must not overflow.
        frames_published must equal frames_sent (1 client connected).
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=100)
        port      = _MOCK_PORT_BASE + 2

        server    = StreamServer(host="127.0.0.1", port=port, buffer_size=1000)
        server.start()

        collector = DataCollector(mock_adapter, mock_registry, server=server)
        collector.start(hz=20.0)

        received: List[dict] = []

        def consume() -> None:
            with StreamClient(host="127.0.0.1", port=port, timeout=5.0) as c:
                for frame in c.frames():
                    received.append(frame)
                    if len(received) >= 40:
                        break

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()

        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            collector.set_visible_actors(actor_ids)
            time.sleep(0.01)

        collector.stop()
        server.stop()
        consumer.join(timeout=3.0)

        assert server.frames_dropped == 0, (
            f"{server.frames_dropped} frames dropped — buffer too small"
        )
        assert len(received) >= 35, (
            f"Consumer received only {len(received)} frames in 2.5 s"
        )

    def test_ethical_metadata_coverage_is_100_percent(
        self, mock_adapter, mock_registry
    ):
        """
        Sprint 2 Integration Gate: 100% ethical metadata coverage.
        Every actor in visible_actors across 5 consecutive frames must have
        non-null ethical attributes.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=50)
        port      = _MOCK_PORT_BASE + 3

        server    = StreamServer(host="127.0.0.1", port=port)
        server.start()

        collector = DataCollector(mock_adapter, mock_registry, server=server)
        collector.start(hz=20.0)
        collector.set_visible_actors(actor_ids)

        frames_to_check: List[dict] = []
        try:
            with StreamClient(host="127.0.0.1", port=port, timeout=5.0) as client:
                time.sleep(0.15)
                for frame in client.frames():
                    frames_to_check.append(frame)
                    if len(frames_to_check) == 5:
                        break

            assert len(frames_to_check) == 5, "Did not receive 5 frames"

            for frame in frames_to_check:
                for actor in frame["ethical_context"]["visible_actors"]:
                    assert actor["age_group"]           is not None, f"Missing age_group for {actor['id']}"
                    assert actor["disability"]          is not None, f"Missing disability for {actor['id']}"
                    assert actor["social_role"]         is not None, f"Missing social_role for {actor['id']}"
                    assert actor["vulnerability_score"] is not None, f"Missing vulnerability_score for {actor['id']}"

        finally:
            collector.stop()
            server.stop()

    def test_node_id_prefix_appears_in_all_actor_ids(
        self, mock_adapter, mock_registry
    ):
        """
        Sprint 3 readiness: actor IDs in every streamed frame must carry the
        node_id prefix so downstream consumers can disambiguate actors from
        different simulation nodes.
        """
        actor_ids = _spawn_and_register(mock_adapter, mock_registry, n=5)
        port      = _MOCK_PORT_BASE + 4

        server    = StreamServer(host="127.0.0.1", port=port)
        server.start()

        collector = DataCollector(mock_adapter, mock_registry, server=server, node_id="sim3")
        collector.start(hz=20.0)
        collector.set_visible_actors(actor_ids)

        try:
            with StreamClient(host="127.0.0.1", port=port, timeout=5.0) as client:
                time.sleep(0.1)
                frame = next(client.frames())

            for actor in frame["ethical_context"]["visible_actors"]:
                assert actor["id"].startswith("sim3-"), (
                    f"Actor ID '{actor['id']}' does not carry 'sim3-' prefix"
                )

        finally:
            collector.stop()
            server.stop()