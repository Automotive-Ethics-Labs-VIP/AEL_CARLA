import pytest
import time

from python_api import CARLAAdapter
from python_api import DataCollector
from python_api import StreamClient
from python_api import StreamServer

pytestmark = pytest.mark.carla

_PORT_BASE = 19300

class TestCARLAAdapter:

    def test_get_snapshot_returns_correct_schema(self, adapter, carla_world):
        snap = adapter.get_snapshot()
        for key in ('timestamp', 'sensor_bundle', 'ethical_context', 'vehicle_state', 'scene_metadata'):
            assert key in snap, f'Missing key in snapshot: {key}'
        assert 'visible_actors' in snap['ethical_context']
        assert 'rgb'   in snap['sensor_bundle']
        assert 'depth' in snap['sensor_bundle']
        assert 'lidar' in snap['sensor_bundle']

    def test_get_snapshot_timestamp_advances(self, adapter, carla_world):
        snap1 = adapter.get_snapshot()
        carla_world.tick()
        snap2 = adapter.get_snapshot()
        assert snap2['timestamp'] > snap1['timestamp'], (
            'Timestamp did not advance after world.tick()'
        )

    def test_scene_metadata_map_name(self, adapter, carla_world):
        snap = adapter.get_snapshot()
        assert snap['scene_metadata']['map'] == 'Town03'

    def test_get_map_called_only_once(self, carla_client, carla_world):
        """
        get_map() should be cached — calling get_snapshot() 10 times should
        only result in one actual get_map() call.  We verify by checking the
        adapter's internal _map cache is populated after the first call.
        """
        adapter = CARLAAdapter(carla_client, carla_world)
        assert adapter._map is None

        for _ in range(10):
            adapter.get_snapshot()
            carla_world.tick()

        # _map must be populated (cached) after the first call
        assert adapter._map is not None

    def test_vehicle_state_populated_with_ego_vehicle(self, carla_client, carla_world):
        """When an ego vehicle is set, vehicle_state must include position and velocity."""
        import carla
        bp_lib = carla_world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_pts  = carla_world.get_map().get_spawn_points()
        ego = carla_world.try_spawn_actor(vehicle_bp, spawn_pts[0])
        assert ego is not None, 'Could not spawn ego vehicle'
        carla_world.tick()

        try:
            adapter = CARLAAdapter(carla_client, carla_world, ego_vehicle=ego)
            snap = adapter.get_snapshot()
            vs = snap['vehicle_state']
            assert 'position' in vs
            assert 'velocity' in vs
            assert 'controls' in vs
            assert len(vs['position']) == 3
        finally:
            ego.destroy()
            carla_world.tick()

    def test_get_spawn_points_returns_list(self, adapter):
        pts = adapter.get_spawn_points()
        assert isinstance(pts, list)
        assert len(pts) > 0

    def test_destroy_actor_removes_it_from_world(self, adapter, carla_world, spawn_points):
        walker_id = adapter.spawn_walker(spawn_points[0], adapter.get_blueprint_library().filter('walker.pedestrian.*')[0])
        carla_world.tick()
        assert adapter.get_actor(walker_id) is not None

        result = adapter.destroy_actor(walker_id)
        carla_world.tick()
        assert result is True
        assert adapter.get_actor(walker_id) is None

    def test_destroy_nonexistent_actor_returns_false(self, adapter):
        assert adapter.destroy_actor(99999999) is False

class TestDataCollector:

    def test_collector_manual_collect(self, adapter, registry, spawner, carla_world, spawn_points):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=10)
        carla_world.tick()

        collector = DataCollector(adapter, registry, server=None, node_id='n0')
        snap = collector.collect(visible_actor_ids=[w.id for w in walkers])

        assert snap is not None
        assert len(snap['ethical_context']['visible_actors']) == 10
        assert collector.frames_collected == 1

    def test_collector_actor_ids_are_node_prefixed(self, adapter, registry, spawner, carla_world, spawn_points):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=5)
        carla_world.tick()

        collector = DataCollector(adapter, registry, node_id='n2')
        snap = collector.collect(visible_actor_ids=[w.id for w in walkers])

        for actor in snap['ethical_context']['visible_actors']:
            assert actor['id'].startswith('n2-'), (
                f"Actor ID '{actor['id']}' not prefixed with 'n2-'"
            )

    def test_collector_20hz_over_3_seconds(self, adapter, registry, spawner, carla_world, spawn_points):
        """
        Run the simulation loop for 3 seconds and verify the collector
        emits ~60 frames (20 Hz × 3 s), within ±5 frames tolerance.
        """
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=20)
        carla_world.tick()

        collector = DataCollector(adapter, registry, server=None, node_id='n0')
        collector.start(hz=20.0)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            carla_world.tick()
            collector.set_visible_actors([w.id for w in walkers])

        collector.stop()

        assert 55 <= collector.frames_collected <= 65, (
            f'Expected ~60 frames in 3s at 20 Hz, got {collector.frames_collected}'
        )

    def test_collector_no_late_frames_under_normal_load(self, adapter, registry, spawner, carla_world, spawn_points):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=20)
        carla_world.tick()

        collector = DataCollector(adapter, registry, server=None)
        collector.start(hz=20.0)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            carla_world.tick()
            collector.set_visible_actors([w.id for w in walkers])

        collector.stop()
        assert collector.frames_late == 0, (
            f'{collector.frames_late} frames ran over the 50ms interval'
        )

    def test_collector_missing_attrs_counter(self, adapter, registry, carla_world, spawn_points):
        """
        Passing actor IDs that are not in the registry increments frames_missing_attrs.
        """
        # collector = DataCollector(adapter, registry, server=None)
        # # Use fake IDs that were never registered
        # snap = collector.collect(visible_actor_ids=[888888, 999999])
        # assert collector.actors_missing_attrs == 2
        collector = DataCollector(adapter, registry, server=None)
        fake_ids  = [888888, 999999]
        snap      = collector.collect(visible_actor_ids=fake_ids)

        # Counter
        assert collector.actors_missing_attrs == 2

        # Null-field contract — both actors still present in the snapshot
        actors = snap['ethical_context']['visible_actors']
        assert len(actors) == 2, (
            f'Expected 2 actors in snapshot, got {len(actors)}'
        )
        for actor in actors:
            assert actor['age_group']           is None, f"Expected null age_group for {actor['id']}"
            assert actor['disability']          is None, f"Expected null disability for {actor['id']}"
            assert actor['social_role']         is None, f"Expected null social_role for {actor['id']}"
            assert actor['vulnerability_score'] is None, f"Expected null vulnerability_score for {actor['id']}"

    def test_collector_context_manager_stops_cleanly(self, adapter, registry):
        with DataCollector(adapter, registry) as collector:
            collector.start(hz=20.0)
            time.sleep(0.2)
        assert not collector.is_running


class TestStreamingEndToEnd:

    def test_full_pipeline_frame_received_by_client(
        self, adapter, registry, spawner, carla_world, spawn_points
    ):
        """
        Spawn walkers → start collector at 20 Hz → connect stream client →
        receive one frame → verify ethical context is present and correct.
        """
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=10)
        carla_world.tick()

        server = StreamServer(host='127.0.0.1', port=_PORT_BASE + 0)
        server.start()

        collector = DataCollector(adapter, registry, server=server, node_id='n0')
        collector.start(hz=20.0)
        collector.set_visible_actors([w.id for w in walkers])

        try:
            with StreamClient(host='127.0.0.1', port=_PORT_BASE + 0, timeout=5.0) as client:
                time.sleep(0.1)   # let at least one frame be published
                frame = next(client.frames())

            assert 'ethical_context' in frame
            actors = frame['ethical_context']['visible_actors']
            assert len(actors) == 10

            for actor in actors:
                assert actor['id'].startswith('n0-')
                assert actor['age_group'] is not None
                assert actor['vulnerability_score'] is not None

        finally:
            collector.stop()
            server.stop()

    def test_stream_frame_matches_snapshot_schema(
        self, adapter, registry, spawner, carla_world, spawn_points
    ):
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=5)
        carla_world.tick()

        server = StreamServer(host='127.0.0.1', port=_PORT_BASE + 1)
        server.start()

        collector = DataCollector(adapter, registry, server=server)
        collector.start(hz=20.0)
        collector.set_visible_actors([w.id for w in walkers])

        try:
            with StreamClient(host='127.0.0.1', port=_PORT_BASE + 1, timeout=5.0) as client:
                time.sleep(0.1)
                frame = next(client.frames())

            for key in ('timestamp', 'sensor_bundle', 'ethical_context', 'vehicle_state', 'scene_metadata'):
                assert key in frame, f'Missing key in streamed frame: {key}'
            assert frame['scene_metadata']['map'] == 'Town03'

        finally:
            collector.stop()
            server.stop()

    def test_no_frames_dropped_at_20hz_with_100_walkers(
        self, adapter, registry, spawner, carla_world, spawn_points
    ):
        """
        At 20 Hz with 100 walkers, the stream buffer should never overflow.
        The design doc allows < 10% drops — we enforce 0% here because
        100 walkers at 20 Hz is well within the system's capacity.
        """
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=100)
        carla_world.tick()

        server    = StreamServer(host='127.0.0.1', port=_PORT_BASE + 2, buffer_size=1000)
        server.start()

        collector = DataCollector(adapter, registry, server=server)
        collector.start(hz=20.0)

        received = []

        def consume():
            with StreamClient(host='127.0.0.1', port=_PORT_BASE + 2, timeout=5.0) as client:
                for frame in client.frames():
                    received.append(frame)
                    if len(received) >= 40:   # 2 seconds worth
                        break

        import threading
        consumer_thread = threading.Thread(target=consume, daemon=True)
        consumer_thread.start()

        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            carla_world.tick()
            collector.set_visible_actors([w.id for w in walkers])

        collector.stop()
        server.stop()
        consumer_thread.join(timeout=3.0)

        assert server.frames_dropped == 0, (
            f'{server.frames_dropped} frames were dropped — buffer too small or pipeline too slow'
        )
        assert len(received) >= 35, (
            f'Consumer only received {len(received)} frames in 2.5s — expected ~40'
        )
        
    def test_ethical_metadata_coverage_is_100_percent(
        self, adapter, registry, spawner, carla_world, spawn_points
    ):
        """
        Design doc acceptance criterion: 100% ethical metadata coverage.
        Every actor in visible_actors must have non-null ethical attributes.
        """
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=50)
        carla_world.tick()

        server    = StreamServer(host='127.0.0.1', port=_PORT_BASE + 3)
        server.start()

        collector = DataCollector(adapter, registry, server=server)
        collector.start(hz=20.0)
        collector.set_visible_actors([w.id for w in walkers])

        frames_to_check = []
        try:
            with StreamClient(host='127.0.0.1', port=_PORT_BASE + 3, timeout=5.0) as client:
                time.sleep(0.15)
                for frame in client.frames():
                    frames_to_check.append(frame)
                    if len(frames_to_check) == 5:
                        break

            for frame in frames_to_check:
                for actor in frame['ethical_context']['visible_actors']:
                    assert actor['age_group']   is not None, f"Missing age_group for {actor['id']}"
                    assert actor['disability']  is not None, f"Missing disability for {actor['id']}"
                    assert actor['social_role'] is not None, f"Missing social_role for {actor['id']}"
                    assert actor['vulnerability_score'] is not None

        finally:
            collector.stop()
            server.stop()
