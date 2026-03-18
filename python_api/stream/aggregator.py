from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple

from .client import StreamClient


class StreamAggregator:
    """
    Merges JSON/TCP frame streams from multiple CARLA simulation nodes into a
    single output stream for the training server.

    Each node runs its own StreamServer and DataCollector with a unique node_id.
    The aggregator connects a StreamClient to every node and fans all incoming
    frames into one shared queue. The frames() generator yields from that queue
    so the training server sees a unified stream regardless of how many nodes
    are running.

    Shared-nothing guarantee: actor IDs are node-prefixed by DataCollector on
    each node (e.g. "n1-0042"), so no coordination is needed here. ID uniqueness
    is guaranteed before frames arrive at the aggregator.

    Args:
        nodes:       List of (host, port) tuples — one per simulation node.
        timeout:     Per-client socket timeout in seconds. Reader threads exit
                     cleanly if a node goes silent for longer than this.
        buffer_size: Max frames held in the internal queue before drops. The
                     default (4 × 1000) accommodates 4 nodes at 20 Hz each
                     with a 50-second backlog before any frame is dropped.

    Usage::

        aggregator = StreamAggregator(
            nodes=[("sim1", 9000), ("sim2", 9000), ("sim3", 9000), ("sim4", 9000)],
        )
        aggregator.start()

        for frame in aggregator.frames():
            # frame["ethical_context"]["visible_actors"] contains prefixed IDs
            # from whichever node produced this frame.
            training_pipeline.ingest(frame)

        aggregator.stop()

    Or as a context manager::

        with StreamAggregator(nodes=[...]) as agg:
            agg.start()
            for frame in agg.frames():
                ...
    """

    def __init__(
        self,
        nodes: List[Tuple[str, int]],
        timeout: float = 5.0,
        buffer_size: int = 4000,
    ) -> None:
        self._nodes       = list(nodes)
        self._timeout     = timeout
        self._buffer_size = buffer_size

        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=buffer_size)
        self._clients:  List[StreamClient]     = []
        self._threads:  List[threading.Thread] = []
        self._running   = False
        self._stats_lock = threading.Lock()

        # Counters — written by reader threads, read by callers.
        self.frames_received:          int            = 0
        self.frames_received_per_node: Dict[int, int] = {
            i: 0 for i in range(len(self._nodes))
        }
        self.frames_dropped: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """
        Connect to all nodes and launch per-node reader threads.

        Returns immediately — aggregation runs entirely in background threads.

        Raises:
            RuntimeError: If already running.
        """
        if self._running:
            raise RuntimeError(
                "StreamAggregator is already running. Call stop() first."
            )

        self._running = True
        for i, (host, port) in enumerate(self._nodes):
            client = StreamClient(host=host, port=port, timeout=self._timeout)
            client.connect()
            self._clients.append(client)

            t = threading.Thread(
                target=self._reader_loop,
                args=(client, i),
                daemon=True,
                name=f"aggregator-reader-{i}",
            )
            self._threads.append(t)
            t.start()

    def stop(self) -> None:
        """
        Disconnect all node clients and wait for reader threads to exit.

        Safe to call even if start() was never called — the method is a no-op
        in that case.
        """
        self._running = False
        for client in self._clients:
            client.disconnect()
        for t in self._threads:
            t.join(timeout=2.0)
        self._clients.clear()
        self._threads.clear()

    def frames(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield merged snapshot frames from all connected nodes.

        Frames arrive in queue-insertion order (approximately wall-clock arrival
        time). The generator blocks up to 0.1 s between frames and exits cleanly
        when stop() is called.

        Raises:
            RuntimeError: If called before start().
        """
        if not self._running:
            raise RuntimeError(
                "Call start() before iterating frames."
            )

        while self._running:
            try:
                yield self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def node_count(self) -> int:
        """Number of nodes this aggregator is (or was) connected to."""
        return len(self._nodes)

    @property
    def active_readers(self) -> int:
        """Number of per-node reader threads still alive."""
        return sum(1 for t in self._threads if t.is_alive())

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "StreamAggregator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _reader_loop(self, client: StreamClient, node_index: int) -> None:
        """
        Drain frames from one node into the shared queue.

        Runs in a daemon thread. Exits when:
          - _running is set to False (stop() was called), or
          - The client's socket is closed / times out (OSError).

        On queue overflow, the frame is dropped and frames_dropped is
        incremented rather than back-pressuring the node's StreamServer.
        """
        try:
            for frame in client.frames():
                if not self._running:
                    break
                try:
                    self._queue.put_nowait(frame)
                    with self._stats_lock:
                        self.frames_received += 1
                        self.frames_received_per_node[node_index] += 1
                except queue.Full:
                    with self._stats_lock:
                        self.frames_dropped += 1
        except OSError:
            # Client socket was closed (by stop() or by the remote node).
            pass