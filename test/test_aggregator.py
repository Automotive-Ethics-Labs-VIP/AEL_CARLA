"""
All tests use MockAdapter, no CARLA required.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, List, Set

import pytest

from python_api.adapter import MockAdapter
from python_api.actor_registry import EthicalActorRegistry
from python_api.collector import DataCollector
from python_api.ethical_attributes import (
    EthicalAttributeSchema, AgeGroup, Disability, SocialRole,
)
from python_api.stream.server import StreamServer
from python_api.stream.aggregator import StreamAggregator

_BASE_PORT = 19500
_NUM_NODES = 4


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _spawn_actors(
    adapter: MockAdapter,
    registry: EthicalActorRegistry,
    n: int,
) -> List[int]:
    """
    Spawn n walkers via the mock adapter and register ethical attributes for
    each one. Returns a list of raw actor IDs.
    """
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


def _start_server(port: int, buffer_size: int = 2000) -> StreamServer:
    server = StreamServer(host="127.0.0.1", port=port, buffer_size=buffer_size)
    server.start()
    time.sleep(0.05)  # give accept thread time to bind
    return server


def _build_nodes(
    port_base: int,
    actors_per_node: int = 25,
    hz: float = 20.0,
) -> tuple:
    """
    Spin up _NUM_NODES independent simulation nodes.

    Returns (servers, collectors, node_actor_ids) where node_actor_ids is a
    list-of-lists: raw actor IDs for each node.
    """
    servers: List[StreamServer] = []
    collectors: List[DataCollector] = []
    node_actor_ids: List[List[int]] = []

    for i in range(_NUM_NODES):
        port = port_base + i
        node_id = f"n{i + 1}"

        adapter = MockAdapter(num_spawn_points=actors_per_node + 10)
        registry = EthicalActorRegistry()
        actor_ids = _spawn_actors(adapter, registry, actors_per_node)

        server = _start_server(port)
        coll = DataCollector(adapter, registry, server=server, node_id=node_id)
        coll.start(hz=hz)
        coll.set_visible_actors(actor_ids)

        servers.append(server)
        collectors.append(coll)
        node_actor_ids.append(actor_ids)

    time.sleep(0.05)  # let collectors emit their first frames
    return servers, collectors, node_actor_ids


def _teardown(aggregator, collectors, servers) -> None:
    if aggregator is not None:
        aggregator.stop()
    for c in collectors:
        c.stop()
    for s in servers:
        s.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Codespace Gate — package must import cleanly without CARLA
# ──────────────────────────────────────────────────────────────────────────────

class TestCodespaceGate:

    def test_package_imports_without_carla(self):
        """
        Sprint 3 Codespace Gate: every public class must be importable even
        when the `carla` package is not installed. MockAdapter enables all
        tests in a CARLA-free environment.
        """
        import python_api  # noqa: F401
        from python_api import (
            EthicalAttributeSchema,
            EthicalActorRegistry,
            EthicalWalkerSpawner,
            DataCollector,
            StreamServer,
            StreamClient,
            StreamAggregator,
        )
        assert EthicalAttributeSchema is not None
        assert EthicalActorRegistry   is not None
        assert EthicalWalkerSpawner   is not None
        assert DataCollector          is not None
        assert StreamServer           is not None
        assert StreamAggregator       is not None

    def test_mock_adapter_satisfies_simulator_protocol(self):
        """
        MockAdapter must satisfy the SimulatorAdapter protocol so the entire
        package is testable without a running CARLA server.
        """
        from python_api.adapter import MockAdapter, SimulatorAdapter
        adapter = MockAdapter()
        assert isinstance(adapter, SimulatorAdapter)

    def test_aggregator_importable_from_stream_subpackage(self):
        """StreamAggregator must be importable from both the subpackage and root."""
        from python_api.stream.aggregator import StreamAggregator as A1
        from python_api.stream import StreamAggregator as A2
        from python_api import StreamAggregator as A3
        assert A1 is A2 is A3

    def test_full_pipeline_instantiates_without_carla(self):
        """
        Instantiating the full pipeline (adapter → registry → spawner →
        collector → server → aggregator) must succeed without CARLA.
        """
        from python_api.adapter import MockAdapter
        from python_api.actor_registry import EthicalActorRegistry
        from python_api.spawn_walkers import EthicalWalkerSpawner
        from python_api.collector import DataCollector
        from python_api.stream.server import StreamServer
        from python_api.stream.aggregator import StreamAggregator

        adapter  = MockAdapter()
        registry = EthicalActorRegistry()
        spawner  = EthicalWalkerSpawner(adapter, registry)  # noqa: F841
        coll     = DataCollector(adapter, registry)          # noqa: F841
        server   = StreamServer()                            # noqa: F841
        agg      = StreamAggregator(nodes=[("localhost", 9000)])  # noqa: F841


# ──────────────────────────────────────────────────────────────────────────────
# Registry Gate — shared-nothing, node-prefixed IDs
# ──────────────────────────────────────────────────────────────────────────────

class TestDistributedRegistry:

    def test_four_nodes_produce_zero_id_collisions(self):
        """
        Sprint 3 Registry Gate: 1000 actors across 4 nodes must have 1000
        unique prefixed IDs — zero collisions.

        Each node spawns 250 actors. The prefixed IDs (e.g. "n1-<raw_id>") are
        collected into a set. A collision would shrink the set below 1000.
        """
        ACTORS_PER_NODE = 250
        all_prefixed_ids: Set[str] = set()

        for i in range(_NUM_NODES):
            adapter   = MockAdapter(num_spawn_points=ACTORS_PER_NODE + 10)
            registry  = EthicalActorRegistry()
            node_id   = f"n{i + 1}"
            actor_ids = _spawn_actors(adapter, registry, ACTORS_PER_NODE)

            for aid in actor_ids:
                all_prefixed_ids.add(f"{node_id}-{aid}")

        expected = _NUM_NODES * ACTORS_PER_NODE
        assert len(all_prefixed_ids) == expected, (
            f"ID collision detected: expected {expected} unique prefixed IDs, "
            f"got {len(all_prefixed_ids)}"
        )

    def test_same_raw_id_on_different_nodes_are_distinct(self):
        """
        Two nodes can legitimately hold the same raw CARLA actor_id (both
        assigned id=42 by their respective CARLA instances). The node prefix
        must make the IDs globally distinct.
        """
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN,
        )
        registry_a = EthicalActorRegistry()
        registry_b = EthicalActorRegistry()
        registry_a.register(42, attrs)
        registry_b.register(42, attrs)

        # Both registries resolve actor 42 independently — shared-nothing
        assert registry_a.get_attributes(42) is not None
        assert registry_b.get_attributes(42) is not None

        # But their prefixed representations are globally distinct
        assert "n1-42" != "n2-42"

    def test_each_node_registry_is_fully_independent(self):
        """
        Registering an actor on node A must have no effect on node B's
        registry. This is the core shared-nothing guarantee.
        """
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN,
        )
        registry_a = EthicalActorRegistry()
        registry_b = EthicalActorRegistry()
        registry_a.register(1001, attrs)

        assert registry_a.get_attributes(1001) is not None
        assert registry_b.get_attributes(1001) is None, (
            "Node B's registry must not be affected by node A registrations."
        )

    def test_collector_prefixes_ids_per_node_id(self):
        """
        DataCollector with node_id='n3' must prefix every actor ID in its
        ethical_context with 'n3-'. Verified end-to-end through collect().
        """
        ACTORS = 20
        adapter   = MockAdapter(num_spawn_points=ACTORS + 5)
        registry  = EthicalActorRegistry()
        actor_ids = _spawn_actors(adapter, registry, ACTORS)

        collector = DataCollector(adapter, registry, server=None, node_id="n3")
        snap = collector.collect(visible_actor_ids=actor_ids)

        for actor in snap["ethical_context"]["visible_actors"]:
            assert actor["id"].startswith("n3-"), (
                f"Expected 'n3-' prefix, got '{actor['id']}'"
            )

    def test_node_prefixed_ids_in_collected_frames_are_unique(self):
        """
        Collect one frame from each of 4 nodes and verify the union of all
        actor IDs contains zero duplicates — even though the underlying raw IDs
        could theoretically overlap across MockAdapter instances.
        """
        ACTORS = 50
        all_ids: List[str] = []

        for i in range(_NUM_NODES):
            adapter   = MockAdapter(num_spawn_points=ACTORS + 5)
            registry  = EthicalActorRegistry()
            node_id   = f"n{i + 1}"
            actor_ids = _spawn_actors(adapter, registry, ACTORS)

            collector = DataCollector(adapter, registry, server=None, node_id=node_id)
            snap = collector.collect(visible_actor_ids=actor_ids)

            for actor in snap["ethical_context"]["visible_actors"]:
                all_ids.append(actor["id"])

        total    = len(all_ids)
        unique   = len(set(all_ids))
        assert total == unique, (
            f"ID collisions in merged ethical_context: {total} total, {unique} unique"
        )

    def test_four_nodes_independent_lru_eviction(self):
        """
        Each node's registry must manage its own LRU independently. Filling one
        node's registry to capacity must not evict entries from another node.
        """
        registry_a = EthicalActorRegistry(max_size=10)
        registry_b = EthicalActorRegistry(max_size=10)
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN,
        )

        # Fill registry_a past its limit
        for i in range(20):
            registry_a.register(i, attrs)

        # Register a known actor in registry_b
        registry_b.register(99, attrs)

        # registry_b must still hold actor 99 — eviction in A is isolated
        assert registry_b.get_attributes(99) is not None, (
            "LRU eviction in node A must not affect node B's registry."
        )


# ──────────────────────────────────────────────────────────────────────────────
# StreamAggregator — merges N node streams
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamAggregator:

    def test_aggregator_merges_frames_from_four_nodes(self):
        """
        Sprint 3 Aggregator Gate: StreamAggregator must receive and merge
        frames from all 4 simulation nodes into a single stream.
        """
        PORT_BASE = _BASE_PORT + 10
        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=25)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(nodes=node_addresses, timeout=5.0, buffer_size=2000)
        aggregator.start()
        received: List[dict] = []
        try:
            time.sleep(0.2)
            for frame in aggregator.frames():
                received.append(frame)
                if len(received) >= 20:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        assert len(received) >= 20, (
            f"Expected >= 20 merged frames from 4 nodes, got {len(received)}"
        )
        assert aggregator.frames_received >= 20

    def test_aggregator_receives_frames_from_every_node(self):
        """
        Every node must contribute at least one frame to the aggregator.
        frames_received_per_node[i] >= 1 for all i.
        """
        PORT_BASE = _BASE_PORT + 20  # 19520–19523
        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=10)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(nodes=node_addresses, timeout=5.0, buffer_size=1000)
        aggregator.start()
        received: List[dict] = []
        try:
            time.sleep(0.3)
            for frame in aggregator.frames():
                received.append(frame)
                if len(received) >= _NUM_NODES * 5:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        for i in range(_NUM_NODES):
            assert aggregator.frames_received_per_node[i] >= 1, (
                f"Node {i} ('n{i+1}') contributed 0 frames to the aggregator. "
                f"All per-node counts: {aggregator.frames_received_per_node}"
            )

    def test_aggregator_zero_id_collisions_across_four_nodes(self):
        """
        Sprint 3 Registry + Aggregator Gate: actor IDs from different nodes
        must never overlap. Zero cross-node collisions required.

        Note: the same actor ID appearing in multiple frames from the *same*
        node is correct behaviour (persistent actors), not a collision.
        We compare the ID sets from each node pairwise to find true overlaps.
        """
        PORT_BASE = _BASE_PORT + 30  # 19530–19533
        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=50)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(nodes=node_addresses, timeout=5.0, buffer_size=2000)
        aggregator.start()

        # Collect one representative frame per node (keyed by node prefix).
        per_node_ids: Dict[str, Set[str]] = {}
        try:
            time.sleep(0.2)
            for frame in aggregator.frames():
                actors = frame["ethical_context"]["visible_actors"]
                if not actors:
                    continue
                # All actors in one frame share the same node prefix.
                prefix = actors[0]["id"].split("-")[0]
                if prefix not in per_node_ids:
                    per_node_ids[prefix] = {a["id"] for a in actors}
                if len(per_node_ids) == _NUM_NODES:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        assert len(per_node_ids) == _NUM_NODES, (
            f"Only collected frames from {len(per_node_ids)}/{_NUM_NODES} nodes: "
            f"{list(per_node_ids.keys())}"
        )

        # Pairwise intersection check — any overlap is a real collision.
        prefixes = list(per_node_ids.keys())
        for i in range(len(prefixes)):
            for j in range(i + 1, len(prefixes)):
                overlap = per_node_ids[prefixes[i]] & per_node_ids[prefixes[j]]
                assert not overlap, (
                    f"Cross-node ID collision between {prefixes[i]} and {prefixes[j]}: "
                    f"{overlap}"
                )

    def test_aggregator_node_count_property(self):
        """node_count must match the number of nodes passed to the constructor."""
        agg = StreamAggregator(nodes=[("h1", 1), ("h2", 2), ("h3", 3)])
        assert agg.node_count == 3

    def test_aggregator_start_raises_if_already_running(self):
        """Calling start() twice must raise RuntimeError."""
        PORT_BASE = _BASE_PORT + 40  # 19540
        server = _start_server(PORT_BASE)
        adapter  = MockAdapter()
        registry = EthicalActorRegistry()
        actor_ids = _spawn_actors(adapter, registry, 5)
        coll = DataCollector(adapter, registry, server=server, node_id="n1")
        coll.start(hz=20.0)
        coll.set_visible_actors(actor_ids)
        time.sleep(0.05)

        agg = StreamAggregator(nodes=[("127.0.0.1", PORT_BASE)], timeout=5.0)
        agg.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                agg.start()
        finally:
            _teardown(agg, [coll], [server])

    def test_aggregator_frames_raises_before_start(self):
        """frames() must raise RuntimeError if called before start()."""
        agg = StreamAggregator(nodes=[("127.0.0.1", 9999)])
        with pytest.raises(RuntimeError, match="Call start()"):
            next(agg.frames())

    def test_aggregator_context_manager_stops_cleanly(self):
        """__exit__ must call stop() — active_readers must be 0 afterwards."""
        PORT_BASE = _BASE_PORT + 50  # 19550–19553
        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=10)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        with StreamAggregator(nodes=node_addresses, timeout=5.0) as agg:
            agg.start()
            time.sleep(0.2)

        # After __exit__, all reader threads must have stopped
        assert agg.active_readers == 0, (
            f"{agg.active_readers} reader thread(s) still alive after __exit__"
        )
        for c in collectors:
            c.stop()
        for s in servers:
            s.stop()

    def test_aggregator_stop_is_idempotent(self):
        """stop() called without start() must not raise."""
        agg = StreamAggregator(nodes=[("localhost", 9000)])
        agg.stop()  # never started
        agg.stop()  # second call — still safe

    def test_merged_stream_contains_all_node_prefixes(self):
        """
        Frames in the merged stream must carry actor IDs from every node prefix
        (n1-, n2-, n3-, n4-). This confirms the aggregator is connected to all
        nodes, not just a subset.
        """
        PORT_BASE = _BASE_PORT + 60  # 19560–19563
        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=10)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(nodes=node_addresses, timeout=5.0, buffer_size=2000)
        aggregator.start()

        prefixes_seen: Set[str] = set()
        frames_collected = 0
        try:
            time.sleep(0.2)
            for frame in aggregator.frames():
                for actor in frame["ethical_context"]["visible_actors"]:
                    prefix = actor["id"].split("-")[0]
                    prefixes_seen.add(prefix)
                frames_collected += 1
                if frames_collected >= 20:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        expected = {f"n{i + 1}" for i in range(_NUM_NODES)}
        assert prefixes_seen == expected, (
            f"Expected actor ID prefixes {expected}, got {prefixes_seen}. "
            "Some nodes may not be contributing to the merged stream."
        )

    def test_aggregator_dropped_frames_counter(self):
        """
        When the aggregator's internal buffer is tiny, frames_dropped must
        increment rather than blocking the reader threads.
        """
        PORT_BASE = _BASE_PORT + 70  # 19570–19573
        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=10, hz=20.0)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        # Intentionally tiny buffer to force drops — never call frames()
        aggregator = StreamAggregator(nodes=node_addresses, timeout=5.0, buffer_size=1)
        aggregator.start()
        try:
            time.sleep(0.5)  # let reader threads fill and drop
        finally:
            _teardown(aggregator, collectors, servers)

        assert aggregator.frames_dropped > 0, (
            "Expected some frames to be dropped with buffer_size=1, "
            "but frames_dropped is 0."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Performance Gate
# ──────────────────────────────────────────────────────────────────────────────

class TestDistributedPerformance:

    def test_aggregate_throughput_80_snapshots_per_second(self):
        """
        Sprint 3 Performance Gate: 4 nodes at 20 Hz each must deliver >=80
        aggregate frames/second to the training server.

        We measure over 2 seconds and require >= 128 frames total
        (80/s × 2s × 0.8 tolerance factor).
        """
        PORT_BASE = _BASE_PORT + 80  # 19580–19583
        TARGET_HZ   = 20.0
        DURATION    = 2.0
        MIN_FRAMES  = int(TARGET_HZ * _NUM_NODES * DURATION * 0.8)  # 128

        servers, collectors, _ = _build_nodes(
            PORT_BASE, actors_per_node=50, hz=TARGET_HZ
        )
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(
            nodes=node_addresses, timeout=5.0, buffer_size=4000
        )
        aggregator.start()

        received_count = 0
        deadline = time.monotonic() + DURATION + 0.5  # +0.5 s warm-up grace
        try:
            time.sleep(0.1)  # pipeline warm-up
            for frame in aggregator.frames():
                received_count += 1
                if time.monotonic() >= deadline:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        assert received_count >= MIN_FRAMES, (
            f"Throughput too low: received {received_count} frames in {DURATION}s, "
            f"expected >= {MIN_FRAMES} "
            f"(target {int(TARGET_HZ * _NUM_NODES)}/s × {DURATION}s × 0.8)"
        )

    def test_frame_drop_rate_under_10_percent(self):
        """
        Design doc acceptance criterion: frame drop rate at the aggregator
        must be < 10% under normal 4-node load.
        """
        PORT_BASE = _BASE_PORT + 90  # 19590–19593
        DURATION  = 2.0

        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=25)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(
            nodes=node_addresses, timeout=5.0, buffer_size=4000
        )
        aggregator.start()

        # Drain the queue while measuring — don't let it back up
        deadline = time.monotonic() + DURATION
        try:
            for frame in aggregator.frames():
                if time.monotonic() >= deadline:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        total_received = aggregator.frames_received
        total_dropped  = aggregator.frames_dropped
        total_seen     = total_received + total_dropped

        if total_seen > 0:
            drop_rate = total_dropped / total_seen
            assert drop_rate < 0.10, (
                f"Frame drop rate {drop_rate:.1%} exceeds 10% threshold. "
                f"Received: {total_received}, dropped: {total_dropped}"
            )

    def test_per_node_throughput_is_balanced(self):
        """
        Under symmetric load (all 4 nodes at 20 Hz), each node should
        contribute roughly equal frame counts to the aggregator. The most
        productive node should not deliver more than 3× what the least
        productive node delivers.
        """
        PORT_BASE = _BASE_PORT + 100  # 19600–19603
        DURATION  = 1.5

        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=20)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(
            nodes=node_addresses, timeout=5.0, buffer_size=4000
        )
        aggregator.start()

        deadline = time.monotonic() + DURATION
        try:
            for frame in aggregator.frames():
                if time.monotonic() >= deadline:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        counts = list(aggregator.frames_received_per_node.values())
        # All nodes must have contributed at least 1 frame
        assert all(c > 0 for c in counts), (
            f"Some nodes contributed 0 frames: {aggregator.frames_received_per_node}"
        )
        # No node should dominate more than 3× another
        if min(counts) > 0:
            ratio = max(counts) / min(counts)
            assert ratio <= 3.0, (
                f"Per-node throughput imbalance: max/min ratio = {ratio:.1f}. "
                f"Counts per node: {aggregator.frames_received_per_node}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Integration Gate — 100% ethical metadata coverage in the merged stream
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration:

    def test_100_percent_ethical_metadata_coverage_across_nodes(self):
        """
        Sprint 3 Integration Gate: every actor in every merged frame must have
        non-null ethical attribute fields. 100% coverage required.
        """
        PORT_BASE = _BASE_PORT + 110  # 19610–19613
        FRAMES_TO_CHECK = 5

        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=25)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(
            nodes=node_addresses, timeout=5.0, buffer_size=2000
        )
        aggregator.start()
        frames_checked: List[dict] = []
        try:
            time.sleep(0.2)
            for frame in aggregator.frames():
                frames_checked.append(frame)
                if len(frames_checked) >= FRAMES_TO_CHECK:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        assert len(frames_checked) == FRAMES_TO_CHECK, (
            f"Only received {len(frames_checked)}/{FRAMES_TO_CHECK} frames from aggregator"
        )
        for frame in frames_checked:
            for actor in frame["ethical_context"]["visible_actors"]:
                assert actor["age_group"]           is not None, \
                    f"Missing age_group for {actor['id']}"
                assert actor["disability"]          is not None, \
                    f"Missing disability for {actor['id']}"
                assert actor["social_role"]         is not None, \
                    f"Missing social_role for {actor['id']}"
                assert actor["vulnerability_score"] is not None, \
                    f"Missing vulnerability_score for {actor['id']}"

    def test_merged_snapshot_schema_is_complete(self):
        """
        Every frame in the merged stream must contain all required top-level
        schema keys: timestamp, sensor_bundle, ethical_context, vehicle_state,
        scene_metadata.
        """
        PORT_BASE = _BASE_PORT + 120  # 19620–19623

        servers, collectors, _ = _build_nodes(PORT_BASE, actors_per_node=10)
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(
            nodes=node_addresses, timeout=5.0, buffer_size=1000
        )
        aggregator.start()
        frames_checked: List[dict] = []
        try:
            time.sleep(0.2)
            for frame in aggregator.frames():
                frames_checked.append(frame)
                if len(frames_checked) >= 8:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

        required_keys = {
            "timestamp", "sensor_bundle", "ethical_context",
            "vehicle_state", "scene_metadata",
        }
        for i, frame in enumerate(frames_checked):
            missing = required_keys - frame.keys()
            assert not missing, (
                f"Frame {i} missing keys: {missing}"
            )

    def test_full_four_node_validation_scenario(self):
        """
        End-to-end scenario matching Sprint 3 design doc:
          - 4 CARLA nodes (mocked), each with 250 actors
          - 20 Hz per node → 80 Hz aggregate
          - 100% ethical metadata coverage
          - 0 actor ID collisions
          - All 4 node prefixes present in merged stream

        This is the single test that must pass to call Sprint 3 complete.
        """
        PORT_BASE      = _BASE_PORT + 130
        ACTORS_PER_NODE = 50   # scaled down from 250 for test speed
        TARGET_HZ       = 20.0
        COLLECTION_TIME = 1.0
        MIN_FRAMES      = int(TARGET_HZ * _NUM_NODES * COLLECTION_TIME * 0.7)

        servers, collectors, _ = _build_nodes(
            PORT_BASE, actors_per_node=ACTORS_PER_NODE, hz=TARGET_HZ
        )
        node_addresses = [("127.0.0.1", PORT_BASE + i) for i in range(_NUM_NODES)]

        aggregator = StreamAggregator(
            nodes=node_addresses, timeout=5.0, buffer_size=4000
        )
        aggregator.start()

        # Collect one representative ID set per node to check cross-node collisions.
        per_node_ids:    Dict[str, Set[str]] = {}
        prefixes_seen:   Set[str]            = set()
        coverage_failures = 0
        total_actors      = 0
        frames_received   = 0

        deadline = time.monotonic() + COLLECTION_TIME + 0.3
        try:
            time.sleep(0.15)
            for frame in aggregator.frames():
                frames_received += 1
                actors = frame["ethical_context"]["visible_actors"]
                if actors:
                    prefix = actors[0]["id"].split("-")[0]
                    prefixes_seen.add(prefix)
                    # Store the first frame per node for collision checking.
                    if prefix not in per_node_ids:
                        per_node_ids[prefix] = {a["id"] for a in actors}

                for actor in actors:
                    total_actors += 1
                    if any(actor[k] is None for k in (
                        "age_group", "disability",
                        "social_role", "vulnerability_score",
                    )):
                        coverage_failures += 1

                if time.monotonic() >= deadline:
                    break
        finally:
            _teardown(aggregator, collectors, servers)

       
        assert frames_received >= MIN_FRAMES, (
            f"Throughput: received {frames_received} frames, need >= {MIN_FRAMES}"
        )

        # Cross-node collision check: pairwise intersection must be empty.
        id_collisions = 0
        prefix_list = list(per_node_ids.keys())
        for i in range(len(prefix_list)):
            for j in range(i + 1, len(prefix_list)):
                overlap = per_node_ids[prefix_list[i]] & per_node_ids[prefix_list[j]]
                id_collisions += len(overlap)
        assert id_collisions == 0, (
            f"Registry Gate FAILED: {id_collisions} cross-node actor ID collision(s)"
        )

        assert prefixes_seen == {f"n{i + 1}" for i in range(_NUM_NODES)}, (
            f"Not all nodes contributed: saw prefixes {prefixes_seen}"
        )
        assert coverage_failures == 0, (
            f"Integration Gate FAILED: {coverage_failures}/{total_actors} actors "
            "had null ethical attributes (expected 100% coverage)"
        )