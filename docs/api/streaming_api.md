# Streaming API Reference

The streaming system exports simulation snapshots with ethical metadata at up to 20 Hz over a plain TCP socket. It uses only the Python standard library.

---

## Table of Contents

1. [Overview](#overview)
2. [Wire Protocol](#wire-protocol)
3. [Frame Schema](#frame-schema)
4. [DataCollector](#datacollector)
5. [StreamServer](#streamserver)
6. [StreamClient](#streamclient)
7. [End-to-End Example](#end-to-end-example)

---

## Overview

```
Simulation loop
      │
      │  set_visible_actors([...])
      ▼
 DataCollector  ──  20 Hz background thread
      │              │
      │              │  collect() → get_snapshot() + _build_actor_context()
      │              ▼
      │         StreamServer.publish(frame)
      │              │
      │         [queue, max 1000 frames]
      │              │
      │         sender thread
      │              │
      └──────────────┤
                     ▼
              StreamClient.frames()  →  Training Pipeline
```

`DataCollector` and `StreamServer` are decoupled — the collector publishes frames into a non-blocking queue; the server drains it in a background thread. The simulation loop is never blocked by network I/O.

---

## Wire Protocol

Each message on the TCP stream is a **length-prefixed JSON frame**:

```
┌─────────────────────────────┬──────────────────────────────────┐
│  4 bytes: uint32 big-endian │  N bytes: UTF-8 JSON payload     │
│  (payload length = N)       │                                  │
└─────────────────────────────┴──────────────────────────────────┘
```

The 4-byte header encodes the byte length of the JSON payload that follows. Consumers read the header first, then read exactly that many bytes — no delimiter scanning, no partial-read ambiguity. Frames larger than 10 MB are rejected by the client as corrupt.

---

## Frame Schema

Every published frame is a JSON object with the following top-level keys:

```json
{
  "timestamp": 42.817,
  "sensor_bundle": {
    "rgb":   "<base64-encoded bytes or empty string>",
    "depth": "<base64-encoded bytes or empty string>",
    "lidar": "<base64-encoded bytes or empty string>"
  },
  "ethical_context": {
    "visible_actors": [
      {
        "id":                  "n0-1042",
        "age_group":           "elderly",
        "disability":          "wheelchair",
        "social_role":         "civilian",
        "vulnerability_score": 0.45,
        "position":            [10.2, 3.5, 0.0],
        "velocity":            [0.5,  0.0, 0.0]
      }
    ]
  },
  "vehicle_state": {
    "position": [15.0, 3.5, 0.0],
    "velocity": [8.3,  0.0, 0.0],
    "controls": {
      "throttle": 0.5,
      "steer":    0.0,
      "brake":    0.0
    }
  },
  "scene_metadata": {
    "weather": "clear_noon",
    "map":     "Town03"
  }
}
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `timestamp` | `float` | Simulation elapsed time in seconds |
| `sensor_bundle.rgb` | `str` | Base64-encoded RGB image, or `""` if not collected |
| `sensor_bundle.depth` | `str` | Base64-encoded depth map, or `""` if not collected |
| `sensor_bundle.lidar` | `str` | Base64-encoded lidar point cloud, or `""` if not collected |
| `ethical_context.visible_actors` | `list` | Actors in sensor range — see below |
| `vehicle_state` | `dict` | Ego vehicle position, velocity, controls. Empty dict `{}` if no ego vehicle is set |
| `scene_metadata.weather` | `str` | Weather preset string from CARLA |
| `scene_metadata.map` | `str` | Map name, e.g. `"Town03"` |

### Actor entry fields

Actors that appear in `visible_actor_ids` but are absent from the registry (e.g. vehicles, props) are included with all ethical attribute fields set to `null`. The training pipeline receives a complete actor list and can filter as needed.

---

## DataCollector

```python
from python_api import DataCollector
```

Assembles snapshot frames enriched with ethical metadata and publishes them to a `StreamServer` at a configurable rate (default 20 Hz).

### Constructor

```python
DataCollector(
    adapter:  SimulatorAdapter,
    registry: EthicalActorRegistry,
    server:   Optional[StreamServer] = None,
    node_id:  str = "n0",
)
```

| Parameter | Description |
|---|---|
| `adapter` | `CARLAAdapter` or `MockAdapter` |
| `registry` | Registry of spawned walker attributes |
| `server` | `StreamServer` to publish to. Pass `None` to collect locally without streaming |
| `node_id` | Prefix applied to all actor IDs in `ethical_context`. Use distinct values per simulation node to guarantee uniqueness in distributed mode |

### Methods

#### `start(hz: float = 20.0) -> None`

Launch the background collection thread.

```python
collector.start(hz=20.0)
```

Raises `ValueError` if `hz <= 0`. Raises `RuntimeError` if already running.

#### `stop() -> None`

Signal the background thread to exit and wait up to 2 seconds for it to finish. Safe to call even if `start()` was never called.

#### `set_visible_actors(actor_ids: List[int]) -> None`

Update the list of actor IDs currently in sensor range. Called from the simulation loop every tick. Thread-safe — the collection thread always reads the most recent list.

```python
# In simulation loop:
while True:
    world.tick()
    collector.set_visible_actors(get_actors_in_sensor_range())
```

#### `collect(visible_actor_ids: Optional[List[int]] = None) -> dict`

Capture one snapshot, enrich it with ethical metadata, and publish it. In automatic mode the background thread calls this for you. Useful for manual control in synchronous scripts or tests.

```python
# Manual mode (no background thread):
snap = collector.collect(visible_actor_ids=[1001, 1002, 1003])
```

### Public attributes (monitoring)

| Attribute | Description |
|---|---|
| `frames_collected` | Total frames successfully assembled |
| `actors_missing_attrs` | Cumulative count of visible actors absent from the registry |
| `frames_late` | Collection cycles that ran over their target interval |
| `is_running` | `True` if background thread is active |

### Context manager

```python
with DataCollector(adapter, registry, server) as collector:
    collector.start(hz=20.0)
    while running:
        world.tick()
        collector.set_visible_actors(visible_ids)
# stop() called automatically on exit
```

---

## StreamServer

```python
from python_api import StreamServer
```

TCP server that broadcasts length-prefixed JSON frames to all connected consumers. Publish calls are non-blocking — frames are dropped if the buffer is full rather than stalling the simulation loop.

### Constructor

```python
StreamServer(
    host:        str = "0.0.0.0",
    port:        int = 9000,
    buffer_size: int = 1000,
)
```

| Parameter | Description |
|---|---|
| `host` | Interface to listen on. `"0.0.0.0"` accepts all interfaces |
| `port` | TCP port |
| `buffer_size` | Max frames queued before dropping. At 20 Hz this is 50 seconds of headroom |

### Methods

#### `start() -> None`

Bind the socket and launch background threads (accept loop + sender loop). Returns immediately.

#### `publish(snapshot: dict) -> None`

Enqueue a frame for delivery to all connected clients. Non-blocking — if the buffer is full, the frame is dropped and `frames_dropped` is incremented.

#### `stop() -> None`

Signal background threads to exit and close all client connections.

### Public attributes (monitoring)

| Attribute | Description |
|---|---|
| `frames_published` | Total `publish()` calls |
| `frames_dropped` | Frames discarded due to full buffer |
| `frames_sent` | Total successful sends, counted per (frame × client) |
| `client_count` | Number of currently connected clients |

---

## StreamClient

```python
from python_api import StreamClient
```

TCP consumer that yields decoded snapshot dicts as they arrive from the server.

### Constructor

```python
StreamClient(
    host:    str = "localhost",
    port:    int = 9000,
    timeout: Optional[float] = None,
)
```

| Parameter | Description |
|---|---|
| `host` | Server hostname or IP |
| `port` | Server TCP port |
| `timeout` | Socket receive timeout in seconds. `None` blocks indefinitely |

### Methods

#### `connect() -> None` / `disconnect() -> None`

Open or close the TCP connection manually.

#### `frames() -> Generator[dict, None, None]`

Yield decoded snapshot dicts as they arrive. The generator exits cleanly when the server closes the connection.

Raises `RuntimeError` if called before `connect()`. Raises `ValueError` if a frame header claims a payload larger than 10 MB (corrupt stream guard).

### Context manager

```python
with StreamClient(host="localhost", port=9000, timeout=10.0) as client:
    for frame in client.frames():
        process(frame["ethical_context"])
```

---

## End-to-End Example

```python
import time
import threading
from python_api.adapter import MockAdapter
from python_api import EthicalActorRegistry, EthicalWalkerSpawner
from python_api import DataCollector, StreamServer, StreamClient

# --- Setup ---
adapter  = MockAdapter(num_spawn_points=50)
registry = EthicalActorRegistry()
spawner  = EthicalWalkerSpawner(adapter, registry)

spawn_points = adapter.get_spawn_points()
walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=20)
actor_ids = list(range(1, 21))  # MockAdapter assigns sequential IDs from 1

# --- Start server ---
server = StreamServer(host="127.0.0.1", port=9000)
server.start()

# --- Start collector at 20 Hz ---
collector = DataCollector(adapter, registry, server=server, node_id="n0")
collector.start(hz=20.0)
collector.set_visible_actors(actor_ids)

# --- Consume frames in a background thread ---
received = []

def consume():
    with StreamClient(host="127.0.0.1", port=9000, timeout=5.0) as client:
        for frame in client.frames():
            received.append(frame)
            if len(received) >= 10:
                break

t = threading.Thread(target=consume)
t.start()

time.sleep(1.0)   # let 20 frames accumulate

# --- Teardown ---
collector.stop()
server.stop()
t.join(timeout=3.0)

print(f"Received {len(received)} frames")
print(f"Dropped:  {server.frames_dropped}")
for actor in received[0]["ethical_context"]["visible_actors"]:
    print(f"  {actor['id']}  {actor['age_group']}  score={actor['vulnerability_score']:.2f}")
```
