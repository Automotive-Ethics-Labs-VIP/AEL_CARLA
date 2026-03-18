"""
generate_training_data.py
=========================
Replacement for Team A's DummyEnvironment in collect_trajectories.py.

Runs a CARLA simulation (or MockAdapter for CARLA-free testing), collects
trajectory pairs using a random policy, and writes them to disk in the exact
JSON format that Team A's annotate_trajectories.py expects:

    [
      {
        "id": "pair_0000",
        "trajectory_a": {"states": [[40 floats], ...], "actions": [int, ...]},
        "trajectory_b": {"states": [[40 floats], ...], "actions": [int, ...]}
      },
      ...
    ]

Why a random policy?
    This script generates the *initial* unannotated pairs used to bootstrap
    the reward model. Team A's own collect_trajectories.py does the same:
    "No checkpoint found. Using randomly initialized policy." A random policy
    is sufficient and correct here — it produces varied trajectories for human
    annotators to compare without any cross-repo dependency.

After this script runs, Team A can immediately call:
    python scripts/annotate_trajectories.py --input <output_file>

HPC usage (submit as a batch job, no live display needed):
    python -m python_api.generate_training_data \\
        --num_pairs   50      \\
        --max_steps   100     \\
        --num_walkers 100     \\
        --output      /scratch/team-b/unannotated_pairs.json \\
        --mock                # omit this flag when CARLA is available

CARLA usage:
    python -m python_api.generate_training_data \\
        --num_pairs   50      \\
        --max_steps   100     \\
        --num_walkers 100     \\
        --carla_host  localhost \\
        --carla_port  2000    \\
        --output      /scratch/team-b/unannotated_pairs.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from .actor_registry import EthicalActorRegistry
from .adapter import MockAdapter
from .spawn_walkers import EthicalWalkerSpawner
from .state_vector_extractor import StateVectorExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Policy type alias + random policy
# ─────────────────────────────────────────────────────────────────────────────

# Any callable with this signature is a valid policy for run_rollout().
PolicyFn = Callable[[List[float]], int]


def _random_policy(state: List[float]) -> int:  # noqa: ARG001
    """
    Uniform random action over {0, 1, 2, 3, 4}.

    `state` is accepted but intentionally not used — this function must
    satisfy the PolicyFn interface so it can be passed anywhere a real
    learned policy would be used, without the caller needing to know the
    difference.
    """
    return random.randint(0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# CARLA environment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _CARLAEnvironment:
    """
    Wraps CARLAAdapter + EthicalWalkerSpawner into reset() / step() / teardown().
    Compatible with _MockEnvironment so run_rollout() works identically for both.
    """

    def __init__(
        self,
        adapter: Any,
        spawner: EthicalWalkerSpawner,
        num_walkers: int,
    ) -> None:
        self._adapter     = adapter
        self._spawner     = spawner
        self._num_walkers = num_walkers
        self._walkers:    List[Any] = []

    def reset(self) -> None:
        for w in self._walkers:
            try:
                self._adapter.destroy_actor(w.id)
            except Exception:
                pass
        self._walkers = []
        spawn_points  = self._adapter.get_spawn_points()
        self._walkers = self._spawner.spawn_walkers_batch(
            spawn_points, num_walkers=self._num_walkers
        )
        self._adapter.tick()

    def step(self) -> None:
        self._adapter.tick()

    def get_snapshot(self) -> Dict[str, Any]:
        return self._adapter.get_snapshot()

    @property
    def visible_actor_ids(self) -> List[int]:
        return [w.id for w in self._walkers]

    def teardown(self) -> None:
        for w in self._walkers:
            try:
                self._adapter.destroy_actor(w.id)
            except Exception:
                pass
        self._walkers = []


# ─────────────────────────────────────────────────────────────────────────────
# Mock environment (no CARLA required)
# ─────────────────────────────────────────────────────────────────────────────

class _MockEnvironment:
    """
    Mock environment backed by MockAdapter for CARLA-free testing.

    Spawns walkers via batch command and registers ethical attributes so
    the full pipeline can be validated without a running CARLA server.
    """

    def __init__(
        self,
        spawner: EthicalWalkerSpawner,
        adapter: MockAdapter,
        num_walkers: int,
    ) -> None:
        self._spawner     = spawner
        self._adapter     = adapter
        self._num_walkers = num_walkers
        self._actor_ids:  List[int] = []

    def reset(self) -> None:
        self._actor_ids = []
        bp      = self._adapter.get_blueprint_library().filter("walker.pedestrian.*")[0]
        pts     = self._adapter.get_spawn_points()
        n_pts   = len(pts)
        cmds    = [
            self._adapter.make_spawn_command(bp, pts[i % n_pts])
            for i in range(self._num_walkers)
        ]
        for resp in self._adapter.spawn_walkers_batch(cmds):
            if not resp.error:
                attrs = self._spawner._generate_ethical_attributes(bp)
                self._spawner.registry.register(resp.actor_id, attrs)
                self._actor_ids.append(resp.actor_id)

    def step(self) -> None:
        self._adapter.tick()

    def get_snapshot(self) -> Dict[str, Any]:
        return self._adapter.get_snapshot()

    @property
    def visible_actor_ids(self) -> List[int]:
        return self._actor_ids

    def teardown(self) -> None:
        self._actor_ids = []


# ─────────────────────────────────────────────────────────────────────────────
# Core rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_rollout(
    env: Any,
    extractor: StateVectorExtractor,
    registry: EthicalActorRegistry,
    policy_fn: PolicyFn,
    node_id: str,
    max_steps: int,
) -> Tuple[List[List[float]], List[int]]:
    """
    Run one trajectory rollout and return (states, actions).

    states  : list of 40-dim state vectors — one per step
    actions : list of action ints (0–4) — one per step

    Output format is identical to Team A's run_rollout() in
    collect_trajectories.py, so the JSON output is directly compatible.

    extractor.reset() is called at the start so velocity_delta (state[3])
    is always 0 on the first step of every trajectory.
    """
    from .collector import DataCollector

    extractor.reset()
    env.reset()

    collector = DataCollector(env._adapter, registry, server=None, node_id=node_id)

    states:  List[List[float]] = []
    actions: List[int]         = []

    for _ in range(max_steps):
        env.step()
        frame     = collector.collect(visible_actor_ids=env.visible_actor_ids)
        state_vec = extractor.extract(frame)
        action    = policy_fn(state_vec)
        states.append(state_vec)
        actions.append(action)

    return states, actions


# ─────────────────────────────────────────────────────────────────────────────
# Main data generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_data(
    num_pairs:      int  = 10,
    max_steps:      int  = 50,
    num_walkers:    int  = 50,
    output_file:    str  = "data/unannotated_pairs.json",
    node_id:        str  = "n0",
    use_mock:       bool = False,
    carla_host:     str  = "localhost",
    carla_port:     int  = 2000,
    num_passengers: int  = 1,
) -> None:
    """
    Generate trajectory pairs from CARLA (or MockAdapter) and write them to
    disk in Team A's unannotated_pairs.json format.

    Args:
        num_pairs:      Number of (traj_A, traj_B) pairs to generate.
        max_steps:      Maximum steps per trajectory rollout.
        num_walkers:    Pedestrians to spawn per episode.
        output_file:    Destination JSON file path.
        node_id:        Node prefix for actor IDs (e.g. 'n1').
        use_mock:       If True, use MockAdapter instead of CARLA.
        carla_host:     CARLA server hostname (ignored when use_mock=True).
        carla_port:     CARLA server port (ignored when use_mock=True).
        num_passengers: Ego vehicle occupant count (state index 1).
    """
    print(f"\n{'='*60}")
    print(f"Team B — generate_training_data.py")
    print(f"{'='*60}")
    print(f"  pairs      : {num_pairs}")
    print(f"  steps/traj : {max_steps}")
    print(f"  walkers    : {num_walkers}")
    print(f"  output     : {output_file}")
    print(f"  mode       : {'mock (no CARLA)' if use_mock else f'CARLA @ {carla_host}:{carla_port}'}")
    print(f"{'='*60}\n")

    registry  = EthicalActorRegistry()
    extractor = StateVectorExtractor(num_passengers=num_passengers)

    if use_mock:
        adapter = MockAdapter(num_spawn_points=max(num_walkers + 10, 60))
        spawner = EthicalWalkerSpawner(adapter, registry)
        env: Any = _MockEnvironment(spawner, adapter, num_walkers)
    else:
        try:
            import carla as _carla  # noqa: F401
        except ImportError:
            print("ERROR: `carla` package not installed. Use --mock for testing.")
            sys.exit(1)

        from .adapter import CARLAAdapter
        client  = _carla.Client(carla_host, carla_port)
        client.set_timeout(15.0)
        world   = client.get_world()
        adapter = CARLAAdapter(client, world)
        spawner = EthicalWalkerSpawner(adapter, registry)
        env     = _CARLAEnvironment(adapter, spawner, num_walkers)

    pairs_data: List[Dict[str, Any]] = []
    t_start = time.monotonic()

    for i in range(num_pairs):
        pair_id = f"pair_{i:04d}"
        print(f"  [{i+1:>4}/{num_pairs}] {pair_id} ...", end="", flush=True)

        states_a, actions_a = run_rollout(
            env, extractor, registry, _random_policy, node_id, max_steps
        )
        states_b, actions_b = run_rollout(
            env, extractor, registry, _random_policy, node_id, max_steps
        )

        pairs_data.append({
            "id": pair_id,
            "trajectory_a": {"states": states_a, "actions": actions_a},
            "trajectory_b": {"states": states_b, "actions": actions_b},
        })

        elapsed  = time.monotonic() - t_start
        per_pair = elapsed / (i + 1)
        eta      = per_pair * (num_pairs - i - 1)
        print(f" done  (ETA {eta:.0f}s)")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(pairs_data, f, indent=4)

    total = time.monotonic() - t_start
    print(f"\nSaved {num_pairs} pairs → {out_path}  ({total:.1f}s total)")
    print(f"\nNext step:")
    print(f"  python scripts/annotate_trajectories.py --input {output_file}")

    try:
        env.teardown()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RLHF trajectory pairs from CARLA simulation. "
            "Output is a JSON file readable by Team A's annotate_trajectories.py."
        )
    )
    parser.add_argument("--num_pairs",      type=int,  default=10)
    parser.add_argument("--max_steps",      type=int,  default=50)
    parser.add_argument("--num_walkers",    type=int,  default=50)
    parser.add_argument("--output",         type=str,  default="data/unannotated_pairs.json")
    parser.add_argument("--node_id",        type=str,  default="n0")
    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--carla_host",     type=str,  default="localhost")
    parser.add_argument("--carla_port",     type=int,  default=2000)
    parser.add_argument("--num_passengers", type=int,  default=1)

    args = parser.parse_args()

    generate_training_data(
        num_pairs     =args.num_pairs,
        max_steps     =args.max_steps,
        num_walkers   =args.num_walkers,
        output_file   =args.output,
        node_id       =args.node_id,
        use_mock      =args.mock,
        carla_host    =args.carla_host,
        carla_port    =args.carla_port,
        num_passengers=args.num_passengers,
    )


if __name__ == "__main__":
    _main()