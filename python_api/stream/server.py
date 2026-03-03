from __future__ import annotations

import json
import queue
import select
import socket
import struct
import threading
from typing import Dict, Any, List


# Length of the frame header: 4-byte big-endian uint32 carrying payload size.
_HEADER_FMT    = ">I"          # big-endian unsigned int

class StreamServer:
    """
    TCP server that broadcasts JSON frames to all connected consumers.

    Frames are length-prefixed so consumers can parse them reliably over a
    streaming TCP connection without delimiter scanning.

    Args:
        host:         Interface to listen on. '0.0.0.0' accepts all interfaces.
        port:         TCP port to listen on.
        buffer_size:  Maximum number of frames queued before dropping. Default
                      matches the design doc spec (1000 frames).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        buffer_size: int = 1000,
    ) -> None:
        self._host        = host
        self._port        = port
        self._buffer      = queue.Queue(maxsize=buffer_size)
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()

        self._server_sock: socket.socket | None = None
        self._running     = False

        # Stats for monitoring drop rate in long-duration tests.
        self.frames_published  = 0
        self.frames_dropped    = 0
        self.frames_sent       = 0 # is incremented once per (frame * client)


    def start(self) -> None:
        """
        Bind the socket and launch the accept + sender background threads.
        Returns immediately — the server runs in the background.
        """
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(10)
        self._server_sock.setblocking(False)
        self._running = True

        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="stream-accept"
        )
        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="stream-sender"
        )
        self._accept_thread.start()
        self._sender_thread.start()

    def publish(self, snapshot: Dict[str, Any]) -> None:
        """
        Enqueue a snapshot for delivery to all connected consumers.

        Non-blocking: if the buffer is full the frame is silently dropped and
        frames_dropped is incremented. This prevents the simulation loop from
        ever blocking on network I/O.

        Args:
            snapshot: Dict matching the streaming data schema. Typically the
                      output of DataCollector.collect() which has already merged
                      sensor data and ethical context into the base snapshot.
        """
        self.frames_published += 1
        try:
            self._buffer.put_nowait(snapshot)
        except queue.Full:
            self.frames_dropped += 1

    def stop(self) -> None:
        """Signal the background threads to exit and close the server socket."""
        self._running = False
        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None

        with self._clients_lock:
            for sock in self._clients:
                _close_quietly(sock)
            self._clients.clear()

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def _prune_dead_clients(self):
        with self._clients_lock:
            if not self._clients:
                return
            candidates = list(self._clients)

        # select with timeout=0: instant, non-blocking poll.
        # Readable sockets either have data (unexpected - clients don't send)
        # or have reached EOF (peer closed). Exceptional sockets have errors.
        try:
            readable, _, exceptional = select.select(candidates, [], candidates, 0)
        except OSError:
            # A socket in the list was already closed externally; skip this cycle.
            return

        dead: List[socket.socket] = list(exceptional)
        for sock in readable:
            try:
                data = sock.recv(1, socket.MSG_PEEK)
                if not data:
                    dead.append(sock)
            except OSError:
                dead.append(sock)

        if dead:
            with self._clients_lock:
                for sock in dead:
                    if sock in self._clients:
                        self._clients.remove(sock)
                    _close_quietly(sock)

    def _accept_loop(self) -> None:
        """Accept incoming client connections and add them to the client list."""
        while self._running:
            if self._server_sock is None:
                break
            # select with a short timeout so we can check _running between calls
            readable, _, _ = select.select([self._server_sock], [], [], 0.5)
            if not readable:
                continue
            try:
                conn, addr = self._server_sock.accept()
                conn.setblocking(True)
                with self._clients_lock:
                    self._clients.append(conn)
            except (OSError, AttributeError):
                break  # server socket was closed by stop()

    def _sender_loop(self) -> None:
        """
        Dequeue frames and send them to every connected client.

        Clients that have disconnected are removed from the list.
        """
        while self._running:
            self._prune_dead_clients()
            try:
                snapshot = self._buffer.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = _encode_frame(snapshot)

            dead_clients: List[socket.socket] = []
            with self._clients_lock:
                current_clients = list(self._clients)

            for sock in current_clients:
                try:
                    _send_all(sock, frame)
                    self.frames_sent += 1
                except OSError:
                    dead_clients.append(sock)

            if dead_clients:
                with self._clients_lock:
                    for sock in dead_clients:
                        if sock in self._clients:
                            self._clients.remove(sock)
                        _close_quietly(sock)


def _encode_frame(data: Dict[str, Any]) -> bytes:
    """Serialise data as a length-prefixed JSON frame."""
    payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
    header  = struct.pack(_HEADER_FMT, len(payload))
    return header + payload


def _send_all(sock: socket.socket, data: bytes) -> None:
    """Send all bytes, blocking until complete."""
    total = 0
    while total < len(data):
        sent = sock.send(data[total:])
        if sent == 0:
            raise OSError("Socket connection closed by remote end")
        total += sent


def _close_quietly(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass