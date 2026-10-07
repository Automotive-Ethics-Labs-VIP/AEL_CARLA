# AEL_CARLA — Team B Simulation Layer

Team B's part of the Automotive Ethics Lab: a Python layer on top of CARLA that spawns pedestrians with ethical attributes, collects simulation data, and turns each frame into the 40-D state vector the Ethical Head (Team A) consumes.

**Default branch:** `main` (the lab's work). `master` is an untouched copy of upstream CARLA, kept for reference only.

## How this repo is laid out

```
AEL_CARLA/
├── python_api/              # The lab's Python package (this is where the work happens)
│   ├── ethical_attributes.py   # Pedestrian ethical attribute schema (child, elderly, wheelchair, ...)
│   ├── actor_registry.py       # Maps CARLA actor IDs to their ethical attributes
│   ├── spawn_walkers.py        # Spawns walkers with attributes attached
│   ├── adapter.py              # CARLA adapter + MockAdapter (runs without CARLA)
│   ├── collector.py            # DataCollector: snapshots of world state
│   ├── state_vector_extractor.py  # Snapshot -> 40-D state vector
│   ├── generate_training_data.py  # Produces trajectory pairs for annotation
│   └── stream/                 # Streaming API (server, client, aggregator)
├── test/                    # pytest suite (mostly runs against MockAdapter)
├── docs/                    # API docs and walker modification guide
├── container/, build-scripts/  # Docker/Singularity + HPC build attempts
├── context/                 # Sprint notes
└── AEL_CARLA/               # Full CARLA engine source (huge; skip it, see below)
```

You don't need the nested `AEL_CARLA/` engine folder for Python work. Clone without it:

```bash
git clone --filter=blob:none --sparse https://github.com/Automotive-Ethics-Labs-VIP/AEL_CARLA.git
cd AEL_CARLA
git sparse-checkout set --no-cone '/*' '!/AEL_CARLA/'
```

## Setup

```bash
pip install -e ".[dev]"
```

To talk to a real CARLA server you also need the client, which must match the server version (0.9.13) and requires Python 3.7 or 3.8:

```bash
pip install carla==0.9.13
```

## Quick start (no CARLA needed)

Run the tests. Tests that need a live server skip unless you pass `--carla`:

```bash
python -m pytest test -q
```

Generate trajectory pairs against the mock world:

```bash
python -m python_api.generate_training_data --mock --num_pairs 10 --output data/unannotated_pairs.json
```

Annotation of the output happens in the Ethical_Head repo (`scripts/annotate_trajectories.py`). With a CARLA server running, drop `--mock` and pass `--carla_host` / `--carla_port` (default `localhost:2000`).

## State vector: ael-v1

The extractor outputs **ael-v1**, the 40-D layout defined in [ael-common](https://github.com/Automotive-Ethics-Labs-VIP/ael-common): the layout the CATA-200 scenarios and the trained Ethical Head use. See the ael-common README for what each index means. Vectors are built with `ael_common.encode()` and checked again before `generate_training_data` writes them.

How a frame maps onto ael-v1 (details in `python_api/state_vector_extractor.py`):

- **Paths:** actors within ±22.5° of the ego heading are straight; 22.5–90° to either side are left or right; actors behind or beyond 50 m are ignored.
- **Obstacle types** come from the actor's blueprint (`walker.*` = pedestrian, bikes = cyclist, `static.prop.*barrier*` = barrier, ...). A scenario can force a type with an `obstacle_type` field.
- **Casualties** follow Team C's CATA-200 counting: 1 per pedestrian, cyclist or motorcyclist; a vehicle's `occupants` (default 1); the ego's passengers for a barrier.
- **Vulnerable groups** come from pedestrians: child, elderly, pregnant, disabled (wheelchair, cane or blind).

The test suite rebuilds all 200 CATA scenarios as scenes and checks the extractor returns exactly the lab's vectors.

## Current status and known gaps

- **No GPU environment yet.** Building the CARLA 0.9.13 Singularity image on SeaWulf gets killed during extraction (ticket open with Research Computing), so nothing here has run against a real CARLA server on HPC. The commands in `build-scripts/` and `container/` are not working yet.
- **No ego vehicle in the data pipeline.** `generate_training_data.py` never spawns or drives an ego vehicle, and the policy's actions are recorded but never applied to the simulation.
- **Lane position is always 0** until a real lane offset is computed from the CARLA map.
- **Untested on real CARLA:** the blueprint-to-obstacle mapping and the left/right convention are checked against the mock only. Verify both on the first real run.
- **Actions are still 5-action IDs** (Team A's annotation format) in generated trajectories; ael-v1 uses 3. Aligning them is the next step on the Ethical_Head side.

## Docs

- `docs/api/ethical_attributes_api.md` — attribute schema and registry
- `docs/api/streaming_api.md` — streaming server/client
- `docs/walker_modifications.md` — custom walker blueprints in the CARLA fork
