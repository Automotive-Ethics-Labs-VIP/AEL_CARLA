"""
test_stream.py — unit tests for StreamServer and StreamClient.

No CARLA required. Tests the TCP framing protocol, multi-client broadcast,
buffer overflow / frame dropping, and the client generator interface.
"""

import time
import threading
import pytest
from python_api.stream.server import StreamServer
from python_api.stream.client import StreamClient
import socket
import struct


def make_snapshot(timestamp: float = 0.0) -> dict:
    return {
        'timestamp':       timestamp,
        'sensor_bundle':   {'rgb': '', 'depth': '', 'lidar': ''},
        'ethical_context': {'visible_actors': []},
        'vehicle_state':   {
            'position': [0.0, 0.0, 0.0],
            'velocity': [5.0, 0.0, 0.0],
            'controls': {'throttle': 0.5, 'steer': 0.0, 'brake': 0.0},
        },
        'scene_metadata':  {'weather': 'clear_noon', 'map': 'Town03'},
    }


def start_server(port: int, buffer_size: int = 1000) -> StreamServer:
    server = StreamServer(host='127.0.0.1', port=port, buffer_size=buffer_size)
    server.start()
    time.sleep(0.05)   # give accept thread time to bind
    return server


def connect_client(port: int, timeout: float = 3.0) -> StreamClient:
    client = StreamClient(host='127.0.0.1', port=port, timeout=timeout)
    client.connect()
    time.sleep(0.05)   # give accept thread time to register the client
    return client


# Port base — each test class uses a different range to avoid conflicts
# if the OS is slow to release ports between tests.
BASE_PORT = 19200

class TestRoundTrip:

    def test_single_frame_delivered(self):
        server = start_server(BASE_PORT + 0)
        client = connect_client(BASE_PORT + 0)
        try:
            server.publish(make_snapshot(1.0))
            frame = next(client.frames())
            assert frame['timestamp'] == pytest.approx(1.0)
        finally:
            client.disconnect()
            server.stop()

    def test_frame_contents_preserved(self):
        server = start_server(BASE_PORT + 1)
        client = connect_client(BASE_PORT + 1)
        try:
            snap = make_snapshot()
            snap['ethical_context']['visible_actors'] = [
                {'id': 'n0-42', 'age_group': 'elderly', 'vulnerability_score': 0.45}
            ]
            server.publish(snap)
            frame = next(client.frames())
            actors = frame['ethical_context']['visible_actors']
            assert len(actors) == 1
            assert actors[0]['id'] == 'n0-42'
            assert actors[0]['age_group'] == 'elderly'
        finally:
            client.disconnect()
            server.stop()

    def test_multiple_frames_in_order(self):
        server = start_server(BASE_PORT + 2)
        client = connect_client(BASE_PORT + 2)
        try:
            for i in range(10):
                server.publish(make_snapshot(float(i)))
            time.sleep(0.05)

            received = []
            for frame in client.frames():
                received.append(frame['timestamp'])
                if len(received) == 10:
                    break

            assert received == [float(i) for i in range(10)]
        finally:
            client.disconnect()
            server.stop()

    def test_context_manager_closes_cleanly(self):
        server = start_server(BASE_PORT + 3)
        try:
            with StreamClient(host='127.0.0.1', port=BASE_PORT + 3, timeout=3.0) as client:
                time.sleep(0.05)
                server.publish(make_snapshot())
                frame = next(client.frames())
                assert 'timestamp' in frame
            # after __exit__, socket should be closed — reconnect should work
            assert client._sock is None
        finally:
            server.stop()

class TestMultiClientBroadcast:

    def test_two_clients_both_receive_frame(self):
        server = start_server(BASE_PORT + 10)
        client_a = connect_client(BASE_PORT + 10)
        client_b = connect_client(BASE_PORT + 10)
        try:
            server.publish(make_snapshot(99.0))
            frame_a = next(client_a.frames())
            frame_b = next(client_b.frames())
            assert frame_a['timestamp'] == pytest.approx(99.0)
            assert frame_b['timestamp'] == pytest.approx(99.0)
        finally:
            client_a.disconnect()
            client_b.disconnect()
            server.stop()

    def test_client_count_property(self):
        server = start_server(BASE_PORT + 11)
        clients = [connect_client(BASE_PORT + 11) for _ in range(4)]
        try:
            time.sleep(0.1)
            assert server.client_count == 4
        finally:
            for c in clients:
                c.disconnect()
            server.stop()

    def test_disconnected_client_removed_from_list(self):
        server = start_server(BASE_PORT + 12)
        client_a = connect_client(BASE_PORT + 12)
        client_b = connect_client(BASE_PORT + 12)
        try:
            time.sleep(0.05)
            client_a.disconnect()
            time.sleep(0.05)
            # Trigger sender loop to detect the dead client
            server.publish(make_snapshot())
            time.sleep(0.1)
            assert server.client_count == 1
        finally:
            client_b.disconnect()
            server.stop()


class TestFrameDropping:

    def test_frames_dropped_when_buffer_full(self):
        server = start_server(BASE_PORT + 20, buffer_size=5)
        try:
            # Publish 10 frames with no consumer — buffer holds only 5
            for i in range(10):
                server.publish(make_snapshot(float(i)))
            assert server.frames_dropped == 5
            assert server.frames_published == 10
        finally:
            server.stop()

    def test_no_drops_when_consumer_keeps_up(self):
        server = start_server(BASE_PORT + 21, buffer_size=100)
        client = connect_client(BASE_PORT + 21)
        try:
            for i in range(20):
                server.publish(make_snapshot(float(i)))

            received = []
            for frame in client.frames():
                received.append(frame)
                if len(received) == 20:
                    break

            assert server.frames_dropped == 0
            assert len(received) == 20
        finally:
            client.disconnect()
            server.stop()


class TestServerStats:

    def test_frames_published_counter(self):
        server = start_server(BASE_PORT + 30)
        try:
            for _ in range(7):
                server.publish(make_snapshot())
            assert server.frames_published == 7
        finally:
            server.stop()

    def test_frames_sent_increments_per_client(self):
        server = start_server(BASE_PORT + 31)
        client = connect_client(BASE_PORT + 31)
        try:
            for i in range(5):
                server.publish(make_snapshot(float(i)))
            time.sleep(0.1)
            assert server.frames_sent == 5
        finally:
            client.disconnect()
            server.stop()



class TestClientErrorHandling:

    def test_frames_before_connect_raises_runtime_error(self):
        client = StreamClient(host='127.0.0.1', port=BASE_PORT + 40, timeout=1.0)
        with pytest.raises(RuntimeError, match='connect'):
            next(client.frames())

    def test_generator_exits_when_server_stops(self):
        server = start_server(BASE_PORT + 41)
        client = connect_client(BASE_PORT + 41)
        try:
            server.publish(make_snapshot(1.0))

            received = []
            def consume():
                for frame in client.frames():
                    received.append(frame)

            t = threading.Thread(target=consume, daemon=True)
            t.start()
            time.sleep(0.1)
            server.stop()
            t.join(timeout=2.0)
            # Generator should have exited cleanly (at least the published frame received)
            assert len(received) >= 1
        finally:
            client.disconnect()

    def test_oversized_frame_raises_value_error(self):
        """
        Manually inject a frame header claiming a 20 MB payload — the client
        must raise ValueError before trying to allocate 20 MB.
        """

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('127.0.0.1', BASE_PORT + 42))
        server_sock.listen(1)
        server_sock.settimeout(3.0)

        def send_oversized():
            conn, _ = server_sock.accept()
            # Header claiming 200 MB payload
            conn.sendall(struct.pack('>I', 200 * 1024 * 1024))
            conn.close()
            server_sock.close()

        t = threading.Thread(target=send_oversized, daemon=True)
        t.start()

        client = StreamClient(host='127.0.0.1', port=BASE_PORT + 42, timeout=3.0)
        client.connect()
        with pytest.raises(ValueError, match='100 MB cap'):
            next(client.frames())
        client.disconnect()
        t.join(timeout=2.0)

class TestClientReconnectAfterTimeout:
    def test_client_timeout_raises_oserror(self):
        """
        A client with a short timeout that receives no frames should raise
        OSError (socket timeout), not hang indefinitely.
        """
        server = start_server(BASE_PORT + 50)
        # timeout=0.2 — short enough that no frames will arrive in time
        client = StreamClient(host="127.0.0.1", port=BASE_PORT + 50, timeout=0.2)
        client.connect()
        time.sleep(0.05)

        try:
            with pytest.raises(OSError):
                next(client.frames())
        finally:
            client.disconnect()
            server.stop()

    def test_reconnect_after_timeout_delivers_new_frames(self):
        """
        After a timeout, the client can call disconnect() + connect() and
        resume receiving frames from the same server on the same port.
        The reconnected client receives whatever the server has queued after
        reconnect — it does NOT replay frames sent before the timeout.
        """
        server = start_server(BASE_PORT + 51)

        # First connection: receive one frame then simulate timeout by disconnecting
        client = StreamClient(host="127.0.0.1", port=BASE_PORT + 51, timeout=3.0)
        client.connect()
        time.sleep(0.05)
        server.publish(make_snapshot(1.0))
        first_frame = next(client.frames())
        assert first_frame["timestamp"] == pytest.approx(1.0)
        client.disconnect()

        # Server keeps running — publish more frames while client is gone
        server.publish(make_snapshot(2.0))
        server.publish(make_snapshot(3.0))
        time.sleep(0.05)

        # Reconnect on the same port
        client.connect()
        time.sleep(0.05)
        server.publish(make_snapshot(4.0))

        try:
            second_frame = next(client.frames())
            # The reconnected client must NOT see timestamp 1.0 again.
            # It will receive whatever is next in the queue after reconnect.
            assert second_frame["timestamp"] == pytest.approx(4.0), (
                f"Expected timestamp 4.0 (first frame published after reconnect), "
                f"got {second_frame['timestamp']}. "
                "Frames 2.0 and 3.0 should have been drained while client was disconnected."
            )
        finally:
            client.disconnect()
            server.stop()

    def test_server_restart_same_port_client_must_reconnect(self):
        """
        Server stops and restarts on the same port (e.g. after a crash).
        An existing client's generator exits cleanly on server stop.
        A new connection to the restarted server works correctly.
        """
        port = BASE_PORT + 52

        # First server lifetime
        server = start_server(port)
        client = connect_client(port)

        server.publish(make_snapshot(10.0))
        pre_restart_frames = []

        def consume_until_disconnect():
            for frame in client.frames():
                pre_restart_frames.append(frame)

        t = threading.Thread(target=consume_until_disconnect, daemon=True)
        t.start()
        time.sleep(0.1)

        server.stop() # kills the server and client generator should exit
        t.join(timeout=2.0)
        client.disconnect()

        assert len(pre_restart_frames) >= 1

        # Give the OS a moment to release the port
        time.sleep(0.15)

        # Second server lifetime — same port
        server2 = start_server(port)
        try:
            client2 = connect_client(port)
            server2.publish(make_snapshot(99.0))
            frame = next(client2.frames())
            assert frame["timestamp"] == pytest.approx(99.0), (
                "After server restart on same port, new client should receive fresh frames "
                "with no bleed-over from the previous server session."
            )
            client2.disconnect()
        finally:
            server2.stop()


class TestSlowConsumer:
    """
    A consumer that reads slower than the publish rate. The server's bounded
    queue should absorb the burst and then start dropping — never blocking
    the publish() caller.
    """

    def test_publish_never_blocks_when_consumer_is_slow(self):
        """
        publish() must return immediately even when the consumer can't keep up.
        We verify this by timing 200 rapid publish() calls — if any block,
        the total time will be >> 1 second.
        """
        server = start_server(BASE_PORT + 60, buffer_size=50)

        # Connect a client that never reads, simulates a stalled training pipeline
        slow_client = connect_client(BASE_PORT + 60)

        try:
            start = time.monotonic()
            for i in range(200):
                server.publish(make_snapshot(float(i)))
            elapsed = time.monotonic() - start

            # 200 non-blocking puts should complete in well under 1 second
            assert elapsed < 1.0, (
                f"publish() blocked: 200 calls took {elapsed:.2f}s. "
                "Simulation loop must never stall on a slow consumer."
            )
            # Buffer holds 50; the rest should have been dropped
            assert server.frames_dropped > 0
        finally:
            slow_client.disconnect()
            server.stop()

class TestConcurrentPublishers:
    """
    Multiple threads calling publish() simultaneously
    """

    def test_concurrent_publish_no_frame_corruption(self):
        """
        Publish from 10 threads simultaneously. Every frame received by the
        client must be a valid, fully-formed snapshot.
        """
        server = start_server(BASE_PORT + 70, buffer_size=1000)
        client = connect_client(BASE_PORT + 70)

        errors = []

        def publish_batch(thread_id: int):
            for i in range(10):
                snap = make_snapshot(float(thread_id * 100 + i))
                snap["scene_metadata"]["map"] = f"thread-{thread_id}"
                server.publish(snap)

        threads = [threading.Thread(target=publish_batch, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        received = []
        for frame in client.frames():
            # Every frame must have the required top-level keys
            for key in ("timestamp", "sensor_bundle", "ethical_context", "vehicle_state", "scene_metadata"):
                if key not in frame:
                    errors.append(f"Frame missing key '{key}': {frame}")
            received.append(frame)
            if len(received) >= 100:
                break

        try:
            assert not errors, f"Corrupt frames detected:\n" + "\n".join(errors)
            assert len(received) == 100
        finally:
            client.disconnect()
            server.stop()

    def test_frames_published_counter_is_accurate_under_concurrency(self):
        """
        frames_published is written from multiple threads. Verify it's not
        subject to race conditions by checking the final count is exact.
        """
        server = start_server(BASE_PORT + 71, buffer_size=10000)

        def publish_100():
            for i in range(100):
                server.publish(make_snapshot(float(i)))

        threads = [threading.Thread(target=publish_100) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            assert server.frames_published == 1000, (
                f"Expected 1000 published frames across 10 threads, "
                f"got {server.frames_published} — possible race condition on counter."
            )
        finally:
            server.stop()


class TestMidFrameDisconnect:
    """
    The server drops a connection after sending only a partial frame.
    The client must exit its generator cleanly rather than hanging or
    raising an unhandled exception.
    """

    def test_partial_header_disconnect_exits_cleanly(self):
        """
        Server sends only 2 of the 4 header bytes then closes.
        _recv_exact returns None → frames() generator exits without raising.
        """
        port = BASE_PORT + 80

        raw_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw_server.bind(("127.0.0.1", port))
        raw_server.listen(1)
        raw_server.settimeout(3.0)

        def send_partial_header():
            conn, _ = raw_server.accept()
            conn.sendall(b"\x00\x00")   # only 2 of 4 header bytes
            conn.close()
            raw_server.close()

        t = threading.Thread(target=send_partial_header, daemon=True)
        t.start()

        client = StreamClient(host="127.0.0.1", port=port, timeout=3.0)
        client.connect()

        received = list(client.frames())   # must not raise, must not hang

        assert received == [], (
            "Expected empty frame list after mid-header disconnect, "
            f"got {received}"
        )
        client.disconnect()
        t.join(timeout=2.0)

    def test_partial_payload_disconnect_exits_cleanly(self):
        """
        Server sends a valid 4-byte header (claiming N bytes) but only sends
        N/2 payload bytes before closing. frames() must exit without raising.
        """
        port = BASE_PORT + 81
        payload = b'{"timestamp": 1.0}'
        header = struct.pack(">I", len(payload))

        raw_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw_server.bind(("127.0.0.1", port))
        raw_server.listen(1)
        raw_server.settimeout(3.0)

        def send_truncated_payload():
            conn, _ = raw_server.accept()
            conn.sendall(header)
            conn.sendall(payload[: len(payload) // 2])   # send only half
            conn.close()
            raw_server.close()

        t = threading.Thread(target=send_truncated_payload, daemon=True)
        t.start()

        client = StreamClient(host="127.0.0.1", port=port, timeout=3.0)
        client.connect()

        received = list(client.frames())   # must not raise

        assert received == [], (
            "Expected empty frame list after mid-payload disconnect, "
            f"got {received}"
        )
        client.disconnect()
        t.join(timeout=2.0)

    def test_zero_length_frame_is_skipped_gracefully(self):
        """
        Server sends a header claiming 0 payload bytes.
        This is a degenerate but legal frame — _recv_exact(sock, 0) returns b'',
        and json.loads(b'') raises JSONDecodeError. Verify client behaviour
        is predictable.
        """
        port = BASE_PORT + 82

        raw_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw_server.bind(("127.0.0.1", port))
        raw_server.listen(1)
        raw_server.settimeout(3.0)

        def send_zero_length():
            conn, _ = raw_server.accept()
            conn.sendall(struct.pack(">I", 0))   # 0-byte payload
            conn.close()
            raw_server.close()

        t = threading.Thread(target=send_zero_length, daemon=True)
        t.start()

        client = StreamClient(host="127.0.0.1", port=port, timeout=3.0)
        client.connect()

        with pytest.raises(ValueError):
            list(client.frames())
        client.disconnect()
        t.join(timeout=2.0)