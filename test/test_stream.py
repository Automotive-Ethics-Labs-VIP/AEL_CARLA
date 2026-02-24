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
        import socket
        import struct

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('127.0.0.1', BASE_PORT + 42))
        server_sock.listen(1)
        server_sock.settimeout(3.0)

        def send_oversized():
            conn, _ = server_sock.accept()
            # Header claiming 20 MB payload
            conn.sendall(struct.pack('>I', 20 * 1024 * 1024))
            conn.close()
            server_sock.close()

        t = threading.Thread(target=send_oversized, daemon=True)
        t.start()

        client = StreamClient(host='127.0.0.1', port=BASE_PORT + 42, timeout=3.0)
        client.connect()
        with pytest.raises(ValueError, match='10 MB cap'):
            next(client.frames())
        client.disconnect()
        t.join(timeout=2.0)