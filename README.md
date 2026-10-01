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

Run the tests. `test_ethical_attributes.py` imports `carla`, so skip it unless the client is installed:

```bash
python -m pytest test -q --ignore=test/test_ethical_attributes.py
```

Generate trajectory pairs against the mock world:

```bash
python -m python_api.generate_training_data --mock --num_pairs 10 --output data/unannotated_pairs.json
```

On Windows, set `PYTHONIOENCODING=utf-8` first, or the progress output crashes the console. Annotation of the output happens in the Ethical_Head repo (`scripts/annotate_trajectories.py`).

With a CARLA server running, drop `--mock` and pass `--carla_host` / `--carla_port` (default `localhost:2000`).

## State vector (40-D)

| Index | Feature |
|---|---|
| 0–3 | ego speed, passenger count, lane position, speed change |
| 4–6 | pedestrian count in straight / left / right path |
| 7–39 | 11 features × 3 directions (straight, left, right): actor count, child, elderly, wheelchair, cane, blind, pregnant, emergency, healthcare, max and mean vulnerability |

Full definition in `python_api/state_vector_extractor.py`.

## Current status and known gaps

- **No GPU environment yet.** Building the CARLA 0.9.13 Singularity image on SeaWulf gets killed during extraction (ticket open with Research Computing), so nothing here has run against a real CARLA server on HPC. The commands in `build-scripts/` and `container/` are not working yet.
- **No ego vehicle in the data pipeline.** `generate_training_data.py` never spawns or drives an ego vehicle, and the policy's actions are recorded but never applied to the simulation.
- **Lane position is world y**, not an offset from the lane centre.
- **Layout mismatch with the Ethical Head.** The extractor's 40-D layout differs from the one the Ethical Head was trained on (Team C's layout: left/straight/right blocks with object types). Don't feed this output to the trained model until the layouts are reconciled.

## Docs

- `docs/api/ethical_attributes_api.md` — attribute schema and registry
- `docs/api/streaming_api.md` — streaming server/client
- `docs/walker_modifications.md` — custom walker blueprints in the CARLA fork
