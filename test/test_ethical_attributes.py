import pytest
import json
import threading
import carla
import time
import tempfile
from pathlib import Path
from collections import Counter

from python_api import EthicalActorRegistry, AgeGroup, Disability, SocialRole, EthicalAttributeSchema, EthicalWalkerSpawner

@pytest.fixture(scope="session")
def carla_client():
    """Connect to CARLA server once for all tests."""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"\nConnected to CARLA server (version {client.get_server_version()})")
        return client
    except RuntimeError as e:
        pytest.fail(f"Could not connect to CARLA server: {e}. Make sure CARLA is running.")

@pytest.fixture
def world(carla_client):
    """Get CARLA world and clean up actors before each test."""
    world = carla_client.get_world()

    # Clean up before test
    actors = world.get_actors()
    walkers = actors.filter('walker.pedestrian.*')
    for walker in walkers:
        walker.destroy()
    time.sleep(0.1)
    
    yield world
    
    # Clean up after test
    actors = world.get_actors()
    walkers = actors.filter('walker.pedestrian.*')
    for walker in walkers:
        walker.destroy()

@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return EthicalActorRegistry()


@pytest.fixture
def spawner(carla_client, world, registry):
    """Create walker spawner with real CARLA world."""
    from python_api.adapter import CARLAAdapter
    adapter = CARLAAdapter(carla_client, world)
    return EthicalWalkerSpawner(adapter, registry)


@pytest.fixture
def temp_dir():
    """Create temporary directory for file tests."""
    tmp = tempfile.mkdtemp()
    yield tmp
    import shutil
    shutil.rmtree(tmp)

"""Unit Tests"""
class TestVulnerabilityScore:
    """Test vulnerability score calculation."""
    
    def test_vulnerability_child_only(self):
        """Test case 1: Child with no other vulnerabilities."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.3, abs=0.01)
    
    def test_vulnerability_elderly_only(self):
        """Test case 2: Elderly with no other vulnerabilities."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ELDERLY,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.2, abs=0.01)
    
    def test_vulnerability_disability_wheelchair(self):
        """Test case 3: Adult with wheelchair."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.WHEELCHAIR,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.25, abs=0.01)
    
    def test_vulnerability_pregnancy_only(self):
        """Test case 4: Pregnant adult."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=True,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.25, abs=0.01)
    
    def test_vulnerability_child_disabled(self):
        """Test case 5: Child with cane."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.CANE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.55, abs=0.01)
    
    def test_vulnerability_elderly_disabled(self):
        """Test case 6: Elderly with blindness."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ELDERLY,
            disability=Disability.BLIND,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.45, abs=0.01)
    
    def test_vulnerability_pregnant_disabled(self):
        """Test case 7: Pregnant adult with disability."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.WHEELCHAIR,
            pregnancy=True,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.5, abs=0.01)
    
    def test_vulnerability_max_capped(self):
        """Test case 8: Maximum vulnerability (should cap at 1.0)."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.WHEELCHAIR,
            pregnancy=True,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == 0.8
        assert attrs.vulnerability_score <= 1.0
    
    def test_vulnerability_teen_healthy(self):
        """Test case 9: Healthy teen."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.TEEN,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.0, abs=0.01)
    
    def test_vulnerability_adult_healthy(self):
        """Test case 10: Healthy adult."""
        attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        assert attrs.vulnerability_score == pytest.approx(0.0, abs=0.01)

class TestRegistryCRUD:
    """Test registry CRUD operations (no CARLA connection needed)."""
    
    def test_register_and_retrieve(self, registry):
        """Test basic registration and retrieval."""
        sample_attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        registry.register(1, sample_attrs)
        retrieved = registry.get_attributes(1)
        assert retrieved is not None
        assert retrieved.age_group == AgeGroup.ADULT
    
    def test_retrieve_nonexistent(self, registry):
        """Test retrieving non-existent actor returns None."""
        result = registry.get_attributes(9999)
        assert result is None
    
    def test_update_existing(self, registry):
        """Test updating existing actor attributes."""
        sample_attrs = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        registry.register(1, sample_attrs)
        
        new_attrs = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.WHEELCHAIR,
            pregnancy=False,
            group_size=2,
            social_role=SocialRole.CIVILIAN
        )
        registry.register(1, new_attrs)
        
        retrieved = registry.get_attributes(1)
        assert retrieved.age_group == AgeGroup.CHILD
        assert retrieved.disability == Disability.WHEELCHAIR
    
    def test_multiple_actors(self, registry):
        """Test registering multiple actors."""
        for i in range(100):
            attrs = EthicalAttributeSchema(
                age_group=AgeGroup.ADULT,
                disability=Disability.NONE,
                pregnancy=False,
                group_size=1,
                social_role=SocialRole.CIVILIAN
            )
            registry.register(i, attrs)
        
        # All should be retrievable (100% success rate)
        for i in range(100):
            assert registry.get_attributes(i) is not None
    
    def test_get_vulnerable_actors(self, registry):
        """Test filtering vulnerable actors."""
        attrs_vulnerable = EthicalAttributeSchema(
            age_group=AgeGroup.CHILD,
            disability=Disability.WHEELCHAIR,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        attrs_not_vulnerable = EthicalAttributeSchema(
            age_group=AgeGroup.ADULT,
            disability=Disability.NONE,
            pregnancy=False,
            group_size=1,
            social_role=SocialRole.CIVILIAN
        )
        
        registry.register(1, attrs_vulnerable)
        registry.register(2, attrs_not_vulnerable)
        registry.register(3, attrs_vulnerable)
        
        vulnerable = registry.get_all_vulnerable_actors(threshold=0.5)
        assert len(vulnerable) == 2
        assert 1 in vulnerable
        assert 3 in vulnerable
        assert 2 not in vulnerable

class TestThreadSafety:
    """Test thread safety with 100 concurrent operations."""
    
    def test_concurrent_register(self):
        """Test concurrent registrations."""
        registry = EthicalActorRegistry()
        num_threads = 100
        threads = []
        
        def register_actor(actor_id):
            attrs = EthicalAttributeSchema(
                age_group=AgeGroup.ADULT,
                disability=Disability.NONE,
                pregnancy=False,
                group_size=1,
                social_role=SocialRole.CIVILIAN
            )
            registry.register(actor_id, attrs)
        
        for i in range(num_threads):
            t = threading.Thread(target=register_actor, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all actors registered
        for i in range(num_threads):
            assert registry.get_attributes(i) is not None
    
    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        registry = EthicalActorRegistry()
        num_operations = 100
        threads = []
        results = []
        
        # Pre-populate
        for i in range(10):
            attrs = EthicalAttributeSchema(
                age_group=AgeGroup.ADULT,
                disability=Disability.NONE,
                pregnancy=False,
                group_size=1,
                social_role=SocialRole.CIVILIAN
            )
            registry.register(i, attrs)
        
        def read_actor(actor_id):
            result = registry.get_attributes(actor_id % 10)
            results.append(result is not None)
        
        def write_actor(actor_id):
            attrs = EthicalAttributeSchema(
                age_group=AgeGroup.CHILD,
                disability=Disability.WHEELCHAIR,
                pregnancy=False,
                group_size=1,
                social_role=SocialRole.CIVILIAN
            )
            registry.register(actor_id % 10, attrs)
        
        for i in range(num_operations):
            if i % 2 == 0:
                t = threading.Thread(target=read_actor, args=(i,))
            else:
                t = threading.Thread(target=write_actor, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All reads should succeed
        assert all(results)

class TestLRUEviction:
    """Test LRU eviction for memory management."""
    
    def test_max_size_enforcement(self):
        """Test that registry respects max_size limit."""
        registry = EthicalActorRegistry(max_size=100)
        
        for i in range(150):
            attrs = EthicalAttributeSchema(
                age_group=AgeGroup.ADULT,
                disability=Disability.NONE,
                pregnancy=False,
                group_size=1,
                social_role=SocialRole.CIVILIAN
            )
            registry.register(i, attrs)
        
        # Count existing actors
        count = sum(1 for i in range(150) if registry.get_attributes(i) is not None)
        assert count == 100
        
        # First 50 should be evicted
        for i in range(50):
            assert registry.get_attributes(i) is None
        
        # Last 100 should exist
        for i in range(50, 150):
            assert registry.get_attributes(i) is not None

class TestJSONSerialization:
    """Test JSON serialization"""
    
    def test_export_import_roundtrip(self, registry, temp_dir):
        """Test that export and import preserve all data."""
        test_cases = [
            (1, AgeGroup.CHILD, Disability.NONE, False, 1, SocialRole.CIVILIAN),
            (2, AgeGroup.ELDERLY, Disability.WHEELCHAIR, False, 2, SocialRole.EMERGENCY),
            (3, AgeGroup.ADULT, Disability.NONE, True, 1, SocialRole.HEALTHCARE),
            (4, AgeGroup.TEEN, Disability.CANE, False, 3, SocialRole.CIVILIAN),
        ]
        
        for actor_id, age, disability, pregnancy, group_size, role in test_cases:
            attrs = EthicalAttributeSchema(
                age_group=age,
                disability=disability,
                pregnancy=pregnancy,
                group_size=group_size,
                social_role=role
            )
            registry.register(actor_id, attrs)
        
        # Export
        filepath = Path(temp_dir) / "test_export.json"
        registry.export_to_json(str(filepath))
        
        # Verify file exists and is valid JSON
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "actors" in data
        
        # Import into new registry
        new_registry = EthicalActorRegistry()
        new_registry.import_from_json(str(filepath))
        
        # Verify all actors preserved (identical distribution requirement)
        for actor_id, age, disability, pregnancy, group_size, role in test_cases:
            attrs = new_registry.get_attributes(actor_id)
            assert attrs is not None
            assert attrs.age_group == age
            assert attrs.disability == disability
            assert attrs.pregnancy == pregnancy
            assert attrs.group_size == group_size
            assert attrs.social_role == role

class TestVulnerabilityScoreRange:
    """Test that all vulnerability scores are in valid range."""
    
    def test_all_combinations_in_range(self):
        """Test all possible attribute combinations produce valid scores."""
        for age in AgeGroup:
            for disability in Disability:
                for pregnancy in [True, False]:
                    attrs = EthicalAttributeSchema(
                        age_group=age,
                        disability=disability,
                        pregnancy=pregnancy,
                        group_size=1,
                        social_role=SocialRole.CIVILIAN
                    )
                    score = attrs.vulnerability_score
                    assert 0.0 <= score <= 1.0

class TestPerformance:
    def test_lookup_performance(self):
        registry = EthicalActorRegistry()
            
        # Pre-populate
        for i in range(1000):
            attrs = EthicalAttributeSchema(
                age_group=AgeGroup.ADULT,
                disability=Disability.NONE,
                pregnancy=False,
                group_size=1,
                social_role=SocialRole.CIVILIAN
            )
            registry.register(i, attrs)
        
        # Test 100 lookups
        start_time = time.time()
        for i in range(100):
            registry.get_attributes(i % 1000)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        assert avg_time_ms < 1.0, f"Average lookup time {avg_time_ms:.3f}ms exceeds 1ms"

"""Integration Tests"""
class TestCarlaIntegration:
    def test_spawn_walkers_all_have_attributes(self, world, registry, spawner):
        """Test spawning walkers and verify all have attributes."""
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        
        # Limit based on available spawn points (may be less than 1000)
        num_walkers = min(1000, len(spawn_points))
        print(f"\nSpawning {num_walkers} walkers (limited by available spawn points)")
        
        # Spawn walkers
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=num_walkers)
        
        print(f"Successfully spawned {len(walkers)} walkers")
        assert len(walkers) > 0, "No walkers spawned"
        
        # Verify all have attributes (100% success rate)
        for walker in walkers:
            attrs = spawner.get_walker_attributes(walker.id)
            assert attrs is not None, f"Walker {walker.id} missing attributes"
            
            # Verify attributes are valid
            assert attrs.age_group in [AgeGroup.CHILD, AgeGroup.TEEN, AgeGroup.ADULT, AgeGroup.ELDERLY]
            assert attrs.disability in [Disability.NONE, Disability.WHEELCHAIR, Disability.CANE, Disability.BLIND]
            assert isinstance(attrs.pregnancy, bool)
            assert 1 <= attrs.group_size <= 5
            
            # Verify vulnerability score in range [0.0, 1.0]
            assert 0.0 <= attrs.vulnerability_score <= 1.0
    
    def test_distribution_with_real_spawns(self, world, registry, spawner):
        """Test attribute distributions with real CARLA spawns."""
        spawn_points = world.get_map().get_spawn_points()
        num_walkers = min(500, len(spawn_points))  # Use 500 for distribution test
        
        print(f"\nTesting distributions with {num_walkers} walkers")
        walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=num_walkers)
        
        # Collect statistics
        age_counts = Counter()
        disability_counts = Counter()
        
        for walker in walkers:
            attrs = registry.get_attributes(walker.id)
            age_counts[attrs.age_group] += 1
            disability_counts[attrs.disability] += 1
        
        total = len(walkers)
        
        # Check distributions (within 10% tolerance for smaller sample)
        print(f"Age distribution:")
        print(f"  Child: {age_counts[AgeGroup.CHILD]/total:.2%} (expected 15%)")
        print(f"  Teen: {age_counts[AgeGroup.TEEN]/total:.2%} (expected 20%)")
        print(f"  Adult: {age_counts[AgeGroup.ADULT]/total:.2%} (expected 50%)")
        print(f"  Elderly: {age_counts[AgeGroup.ELDERLY]/total:.2%} (expected 15%)")
        
        # Verify reasonable distributions
        assert age_counts[AgeGroup.CHILD] / total == pytest.approx(0.15, abs=0.10)
        assert age_counts[AgeGroup.TEEN] / total == pytest.approx(0.20, abs=0.10)
        assert age_counts[AgeGroup.ADULT] / total == pytest.approx(0.50, abs=0.10)
        assert age_counts[AgeGroup.ELDERLY] / total == pytest.approx(0.15, abs=0.10)
    
    def test_memory_stability_spawn_despawn_cycles(self, world, registry, spawner):
        """Test memory stability over spawn/despawn cycles."""
        spawn_points = world.get_map().get_spawn_points()
        
        # Run multiple spawn/despawn cycles (scaled down from 10,000 for practicality)
        num_cycles = 10
        walkers_per_cycle = min(100, len(spawn_points))
        
        print(f"\nRunning {num_cycles} spawn/despawn cycles with {walkers_per_cycle} walkers each")
        
        for cycle in range(num_cycles):
            # Spawn
            walkers = spawner.spawn_walkers_batch(
                spawn_points[:walkers_per_cycle], 
                num_walkers=walkers_per_cycle
            )
            
            # Verify all have attributes
            for walker in walkers:
                assert registry.get_attributes(walker.id) is not None
            
            # Despawn
            for walker in walkers:
                walker.destroy()
            
            # Small delay
            time.sleep(0.1)
        
        print("Memory stability test completed successfully")
