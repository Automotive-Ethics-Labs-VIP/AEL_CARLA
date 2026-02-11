# Ethical Attributes API Documentation

## Overview

The Ethical Attributes system extends CARLA's pedestrian walkers with metadata for ethical AI training scenarios. This system assigns vulnerability scores, age groups, disabilities, and social roles to walkers without modifying CARLA's core source code.


---

## Quick Start

```python
import carla
from python_api.ethical_attributes import EthicalAttributeSchema, AgeGroup, Disability, SocialRole
from python_api.actor_registry import EthicalActorRegistry
from python_api.spawn_walkers import EthicalWalkerSpawner

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Initialize system
registry = EthicalActorRegistry()
spawner = EthicalWalkerSpawner(client, world, registry)

# Spawn walkers with ethical attributes
spawn_points = world.get_map().get_spawn_points()
walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=50)

# Query attributes
for walker in walkers:
    attrs = spawner.get_walker_attributes(walker.id)
    print(f"Walker {walker.id}: {attrs.age_group}, vulnerability={attrs.vulnerability_score}")
```

---

## API Reference

### EthicalAttributeSchema

Pydantic model representing ethical attributes for a single walker.

#### Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `age_group` | `AgeGroup` | Age category | Enum: child, teen, adult, elderly |
| `disability` | `Disability` | Disability type | Enum: none, wheelchair, cane, blind |
| `pregnancy` | `bool` | Pregnancy status | Only applicable to adult females |
| `group_size` | `int` | Size of walking group | 1-5 |
| `social_role` | `SocialRole` | Professional role | Enum: civilian, emergency, healthcare |
| `vulnerability_score` | `float` | Computed vulnerability | 0.0-1.0 (read-only) |

#### Vulnerability Score Calculation

```python
base = 0.0
if age_group == 'child': base += 0.3
if age_group == 'elderly': base += 0.2
if disability != 'none': base += 0.25
if pregnancy: base += 0.25
return min(1.0, base)
```

#### Example Usage

```python
from python_api.ethical_attributes import EthicalAttributeSchema, AgeGroup, Disability, SocialRole

# Create attributes manually
attrs = EthicalAttributeSchema(
    age_group=AgeGroup.ELDERLY,
    disability=Disability.CANE,
    pregnancy=False,
    group_size=1,
    social_role=SocialRole.CIVILIAN
)

print(attrs.vulnerability_score)  # 0.45 (0.2 + 0.25)
print(attrs.model_dump())  # Convert to dict
```

---

### EthicalActorRegistry

Thread-safe registry mapping actor IDs to ethical attributes.

#### Constructor

```python
EthicalActorRegistry(max_size: int = 10000)
```

**Parameters:**
- `max_size` (int): Maximum actors before LRU eviction. Default: 10,000

#### Methods

##### `register(actor_id: int, attrs: EthicalAttributeSchema) -> None`

Register or update an actor's ethical attributes.

```python
registry.register(walker.id, attrs)
```

##### `get_attributes(actor_id: int) -> Optional[EthicalAttributeSchema]`

Retrieve attributes for an actor. Returns `None` if not found.

```python
attrs = registry.get_attributes(walker.id)
if attrs:
    print(f"Vulnerability: {attrs.vulnerability_score}")
```

##### `get_all_vulnerable_actors(threshold: float = 0.5) -> List[int]`

Get all actor IDs with vulnerability score above threshold.

```python
vulnerable_ids = registry.get_all_vulnerable_actors(threshold=0.5)
print(f"Found {len(vulnerable_ids)} vulnerable actors")
```

##### `export_to_json(filepath: str) -> None`

Export registry to JSON file for scenario persistence.

```python
registry.export_to_json("scenarios/scenario_001.json")
```

##### `import_from_json(filepath: str) -> None`

Import registry from JSON file for scenario replay.

```python
registry.import_from_json("scenarios/scenario_001.json")
```

#### Thread Safety

All methods are thread-safe and use internal locking. Safe for concurrent access from multiple threads.

```python
import threading

def spawn_worker(spawner, spawn_points):
    spawner.spawn_walker(spawn_points[0])

threads = [threading.Thread(target=spawn_worker, args=(spawner, spawn_points)) 
        for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

### EthicalWalkerSpawner

High-level interface for spawning walkers with ethical attributes.

#### Constructor

```python
EthicalWalkerSpawner(client: carla.Client, world: carla.World, registry: Optional[EthicalActorRegistry] = None)
```

**Parameters:**
- `world` (carla.World): CARLA world instance
- `registry` (EthicalActorRegistry): Registry to store attributes

#### Methods

##### `spawn_walker(spawn_point: carla.Transform, walker_bp: Optional[carla.ActorBlueprint] = None) -> carla.Actor`

Spawn a single walker with ethical attributes.

```python
spawn_point = carla.Transform(carla.Location(x=10, y=20, z=0.5))
walker = spawner.spawn_walker(spawn_point)
```

**Parameters:**
- `spawn_point` (carla.Transform): Spawn location
- `walker_bp` (Optional): Specific blueprint. Random if None.

**Returns:** Spawned walker actor

**Raises:** `RuntimeError` if spawning fails

##### `spawn_walkers_batch(spawn_points: List[carla.Transform], num_walkers: Optional[int] = None) -> List[carla.Actor]`

Spawn multiple walkers efficiently.

```python
spawn_points = world.get_map().get_spawn_points()
walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=100)
print(f"Spawned {len(walkers)} walkers")
```

**Parameters:**
- `spawn_points` (List[carla.Transform]): List of spawn locations
- `num_walkers` (Optional[int]): Number to spawn. Uses all points if None.

**Returns:** List of spawned walker actors

##### `get_walker_attributes(walker_id: int) -> Optional[EthicalAttributeSchema]`

Convenience method to get attributes from registry.

```python
attrs = spawner.get_walker_attributes(walker.id)
```

##### `get_vulnerable_walkers(threshold: float = 0.5) -> List[int]`

Get vulnerable walker IDs from registry.

```python
vulnerable_ids = spawner.get_vulnerable_walkers(threshold=0.5)
```

---

## Attribute Distributions

The spawner generates attributes with the following statistical distributions:

### Age Groups
- **15%** Child
- **20%** Teen
- **50%** Adult
- **15%** Elderly

### Disabilities
- **85%** None
- **8%** Wheelchair
- **5%** Cane
- **2%** Blind

### Special Conditions
- **Pregnancy:** 3% of adult females
- **Group Size:**
  - 70% solo (1)
  - 20% pairs (2)
  - 10% groups (3-5)

### Social Roles
- **95%** Civilian
- **3%** Emergency services
- **2%** Healthcare

---

## JSON Schema

### Export Format

```json
{
  "metadata": {
    "timestamp": "2025-02-07T14:32:00.000000Z",
    "total_actors": 150,
    "carla_version": "0.9.16"
  },
  "actors": {
    "12345": {
      "age_group": "elderly",
      "disability": "cane",
      "pregnancy": false,
      "group_size": 1,
      "social_role": "civilian"
    },
    "12346": {
      "age_group": "adult",
      "disability": "none",
      "pregnancy": true,
      "group_size": 2,
      "social_role": "healthcare"
    }
  }
}
```

### Import/Export Example

```python
# Save scenario
registry.export_to_json("scenarios/downtown_rush_hour.json")

# Load scenario in different session
new_registry = EthicalActorRegistry()
new_registry.import_from_json("scenarios/downtown_rush_hour.json")

# Verify data
attrs = new_registry.get_attributes(12345)
assert attrs.age_group == AgeGroup.ELDERLY
```

---

## Performance Characteristics

### LRU Eviction

When registry exceeds `max_size`, oldest actors are evicted:

```python
registry = EthicalActorRegistry(max_size=5000)

# After 5001st registration, actor #0 is evicted
for i in range(5001):
    registry.register(i, attrs)

assert registry.get_attributes(0) is None  # Evicted
assert registry.get_attributes(5000) is not None  # Kept
```

---

## Advanced Usage

### Filtering Walkers by Criteria

```python
# Get all vulnerable walkers
vulnerable_ids = registry.get_all_vulnerable_actors(threshold=0.5)

# Filter by specific criteria
for actor_id in vulnerable_ids:
    attrs = registry.get_attributes(actor_id)
    if attrs.age_group == AgeGroup.CHILD and attrs.disability != Disability.NONE:
        print(f"{actor_id} is a child and has a disability.")
```

### Custom Spawn Distribution

```python
for spawn_point in spawn_points[:30]:
    # Note: Internal _generate_ethical_attributes still uses random distribution
    # To force specific attributes, register manually after spawn
    walker = spawner.spawn_walker(spawn_point)
    
    # Override with custom attributes
    custom_attrs = EthicalAttributeSchema(
        age_group=AgeGroup.CHILD,
        disability=Disability.NONE,
        pregnancy=False,
        group_size=2,
        social_role=SocialRole.CIVILIAN
    )
    registry.register(walker.id, custom_attrs)
```

### Scenario Persistence Workflow

```python
# 1. Create and populate scenario
spawner = EthicalWalkerSpawner(client, world, registry)
walkers = spawner.spawn_walkers_batch(spawn_points, num_walkers=200)

# 2. Save scenario
registry.export_to_json("scenarios/intersection_test_v1.json")

# 3. Later: Replay exact scenario
new_registry = EthicalActorRegistry()
new_registry.import_from_json("scenarios/intersection_test_v1.json")

# Spawn walkers and restore their attributes
```

