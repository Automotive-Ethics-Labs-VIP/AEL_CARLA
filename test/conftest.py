import os
import pytest

# use --carla to run carla tests

def pytest_addoption(parser):
    parser.addoption(
        '--carla',
        action='store_true',
        default=False,
        help='Run integration tests that require a live CARLA server.',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'carla: mark test as requiring a live CARLA server (use --carla to enable)',
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--carla'):
        skip = pytest.mark.skip(reason='requires --carla flag and a running CARLA server')
        for item in items:
            if 'carla' in item.keywords:
                item.add_marker(skip)


CARLA_HOST    = os.environ.get('CARLA_HOST', 'localhost')
CARLA_PORT    = int(os.environ.get('CARLA_PORT', 2000))
CARLA_TIMEOUT = 15.0
CARLA_MAP     = 'Town03'

@pytest.fixture(scope='session')
def carla_client():
    """
    Single carla.Client shared across the entire test session.
    Connecting to CARLA takes ~2-3 seconds so we do it once.
    """
    import carla
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    # Verify the server is reachable before any test runs.
    try:
        client.get_server_version()
    except RuntimeError as e:
        pytest.exit(
            f'Cannot reach CARLA server at {CARLA_HOST}:{CARLA_PORT}. '
            f'Start the server first.\n  Error: {e}',
            returncode=1,
        )
    yield client

@pytest.fixture(scope='function')
def carla_world(carla_client):
    """
    Load a clean world for each test in synchronous mode.

    Synchronous mode (fixed_delta_seconds=0.05 => 20 Hz sim tick) ensures
    that world.tick() advances the simulation by exactly one step, making
    spawning and state checks deterministic.

    Teardown destroys every pedestrian actor and restores async mode so the
    server is left in a clean state for the next test.
    """
    import carla

    world = carla_client.load_world(CARLA_MAP)

    # Switch to synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05   # 20 Hz
    world.apply_settings(settings)
    world.tick()   # settle

    yield world

    # --- Teardown ---
    # Destroy all pedestrians spawned during the test
    for actor in world.get_actors().filter('walker.pedestrian.*'):
        actor.destroy()
    world.tick()   # flush destroys

    # Restore async mode so the next test starts in a known state
    settings.synchronous_mode    = False
    settings.fixed_delta_seconds = 0.0
    world.apply_settings(settings)


@pytest.fixture(scope='function')
def adapter(carla_client, carla_world):
    """CARLAAdapter wrapping the test world."""
    from python_api.adapter import CARLAAdapter
    return CARLAAdapter(carla_client, carla_world)


@pytest.fixture(scope='function')
def registry():
    """New EthicalActorRegistry for each test."""
    from python_api.actor_registry import EthicalActorRegistry
    return EthicalActorRegistry()


@pytest.fixture(scope='function')
def spawner(adapter, registry):
    """EthicalWalkerSpawner wired to the test adapter and registry."""
    from python_api.spawn_walkers import EthicalWalkerSpawner
    return EthicalWalkerSpawner(adapter, registry)


@pytest.fixture(scope='function')
def spawn_points(carla_world):
    """All available spawn points in the test map."""
    return carla_world.get_map().get_spawn_points()
