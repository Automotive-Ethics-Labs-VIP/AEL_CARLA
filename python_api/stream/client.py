from __future__ import annotations

import json
import socket
import struct
from typing import Dict, Any, Generator, Optional


_HEADER_FMT   = ">I"
_HEADER_BYTES = struct.calcsize(_HEADER_FMT)  # == 4


class StreamClient:
    """
    TCP consumer for the JSON/length-prefix streaming protocol.

    Args:
        host:    Server hostname or IP address.
        port:    Server TCP port.
        timeout: Socket receive timeout in seconds. None = block indefinitely.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        timeout: Optional[float] = None,
    ) -> None:
        self._host    = host
        self._port    = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        """Open the TCP connection to the server."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self._timeout is not None:
            self._sock.settimeout(self._timeout)
        self._sock.connect((self._host, self._port))

    def disconnect(self) -> None:
        """Close the connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self) -> "StreamClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    def frames(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield decoded snapshot dicts as they arrive from the server.

        Raises:
            RuntimeError:  If called before connect().
            OSError:       On socket errors (connection reset, timeout, etc.).
            ValueError:    If a frame cannot be decoded as JSON.

        The generator exits cleanly when the server closes the connection.
        """
        if self._sock is None:
            raise RuntimeError("Call connect() before iterating frames.")

        while True:
            # Read exactly 4-byte header
            header = _recv_exact(self._sock, _HEADER_BYTES)
            if header is None:
                break  # server closed connection gracefully

            (payload_length,) = struct.unpack(_HEADER_FMT, header)

            # Guard against absurdly large frames (e.g. corrupt stream)
            if payload_length > 10 * 1024 * 1024:  # 10 MB hard cap
                raise ValueError(
                    f"Frame payload length {payload_length} exceeds 10 MB cap. "
                    "Possible stream corruption."
                )

            payload = _recv_exact(self._sock, payload_length)
            if payload is None:
                break  # server closed connection mid-frame

            yield json.loads(payload.decode("utf-8"))

    # def read_one(self) -> Optional[Dict[str, Any]]:
    #     """
    #     Read and return a single frame, or None if the connection is closed.

    #     Convenience wrapper around frames() for cases where you want to pull
    #     frames manually rather than in a for-loop.
    #     """
    #     try:
    #         return next(self.frames())
    #     except StopIteration:
    #         return None

def _recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    """
    Read exactly num_bytes from sock.

    Returns None if the connection is closed before all bytes arrive.
    Raises OSError on socket errors (timeout, reset, etc.).
    """
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            return None  # clean EOF from server
        buf.extend(chunk)
    return bytes(buf)