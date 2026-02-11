import carla
from .ethical_attributes import EthicalAttributeSchema, AgeGroup, Disability, SocialRole
from .actor_registry import EthicalActorRegistry
import random
from typing import List, Optional, Tuple


class EthicalWalkerSpawner:
    def __init__(self, client: carla.Client, world: carla.World, registry: Optional[EthicalActorRegistry] = None):
        self.client = client # store the carla client for batch operations
        self.world = world
        self.registry = registry if registry is not None else EthicalActorRegistry()
        self.blueprint_library = world.get_blueprint_library()
        self.spawned_walkers: List[carla.Actor] = []
     
    def _generate_ethical_attributes(self, walker_bp) -> EthicalAttributeSchema:
        """
        Generate ethical attributes with specified distributions:
        - age_group: 15% child, 20% teen, 50% adult, 15% elderly
        - disability: 85% none, 8% wheelchair, 5% cane, 2% blind
        - pregnancy: 3% of adult females
        - group_size: 70% solo, 20% pairs, 10% groups of 3+

        Args:
            walker_bp: CARLA walker blueprint to check gender attribute
        """
        age_group = random.choices(
            population=[AgeGroup.CHILD, AgeGroup.TEEN, AgeGroup.ADULT, AgeGroup.ELDERLY],
            weights=[0.15, 0.20, 0.50, 0.15],
            k=1
        )[0]

        disability = random.choices(
            population=[Disability.NONE, Disability.WHEELCHAIR, Disability.CANE, Disability.BLIND],
            weights=[0.85, 0.08, 0.05, 0.02],
            k=1
        )[0]

        # this relies on the walker blueprint having their gender attribute as female, not just our metadata
        # if we just use a single blueprint that has a gender attribute as male, none of them will have the pregnancy attribute
        pregnancy = False 
        if age_group == AgeGroup.ADULT:
            gender = walker_bp.get_attribute('gender').as_str()
            is_female = (gender == 'female')

            if is_female and random.random() < 0.03:
                pregnancy = True

        group_size = random.choices(
            population=[1, 2, 3, 4, 5],
            weights=[0.70, 0.20, 0.05, 0.03, 0.02],
            k=1
        )[0]

        social_role = random.choices(
            population=[SocialRole.CIVILIAN, SocialRole.EMERGENCY, SocialRole.HEALTHCARE],
            weights=[0.95, 0.03, 0.02],
            k=1
        )[0]

        return EthicalAttributeSchema(
            age_group=age_group,
            disability=disability,
            pregnancy=pregnancy,
            group_size=group_size,
            social_role=social_role
        )

    def spawn_walker(self, spawn_point: carla.Transform, walker_bp: Optional[carla.ActorBlueprint] = None) -> carla.Actor:
        """
        Spawn a single walker with ethical attributes.
        
        Args:
            spawn_point: Transform location for spawning
            walker_bp: Optional specific walker blueprint. Random if not provided.
            
        Returns:
            Spawned walker actor
            
        Raises:
            RuntimeError: If spawning fails
        """

        if walker_bp is None:
            walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
            walker_bp = random.choice(walker_blueprints)
        
        walker = self.world.try_spawn_actor(walker_bp, spawn_point)
        
        if walker is None:
            raise RuntimeError(f"Failed to spawn walker at {spawn_point.location}")
        
        attributes = self._generate_ethical_attributes(walker_bp)
        self.registry.register(walker.id, attributes)
        
        self.spawned_walkers.append(walker)
        
        return walker
        
    def spawn_walkers_batch(self, spawn_points: List[carla.Transform], num_walkers: Optional[int] = None) -> List[carla.Actor]:
        """
        Spawn multiple walkers with ethical attributes using CARLA's batch API.
        
        This method is much faster than individual spawns because it uses a single
        RPC call to spawn all walkers at once.
        
        Args:
            spawn_points: List of spawn locations
            num_walkers: Number of walkers to spawn. Uses all spawn points if None.
            
        Returns:
            List of spawned walker actors
        """
        if num_walkers is None:
            num_walkers = len(spawn_points)
        
        num_walkers = min(num_walkers, len(spawn_points))
        
        # Prepare blueprints and attributes before batch spawn
        walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
        spawn_commands = []
        blueprint_attr_pairs: List[Tuple[carla.ActorBlueprint, EthicalAttributeSchema]] = []
        
        for i in range(num_walkers):
            # Select random blueprint
            walker_bp = random.choice(walker_blueprints)
            
            # Generate attributes in advance (before spawning)
            attributes = self._generate_ethical_attributes(walker_bp)
            blueprint_attr_pairs.append((walker_bp, attributes))
            
            # Create spawn command
            spawn_commands.append(carla.command.SpawnActor(walker_bp, spawn_points[i]))
        
        # Execute batch spawn
        batch_results = self.client.apply_batch_sync(spawn_commands, do_tick=False)
        
        # Process results and register attributes
        walkers = []
        for i, response in enumerate(batch_results):
            if response.error:
                print(f"Warning: Could not spawn walker {i}: {response.error}")
                continue
            
            # Get the spawned actor
            actor_id = response.actor_id
            walker = self.world.get_actor(actor_id)
            
            if walker is not None:
                # Register the pre-generated attributes
                _, attributes = blueprint_attr_pairs[i]
                self.registry.register(actor_id, attributes)
                
                self.spawned_walkers.append(walker)
                walkers.append(walker)
        
        return walkers
    
    def get_walker_attributes(self, walker_id: int) -> Optional[EthicalAttributeSchema]:
        """
        Get ethical attributes for a walker.
        
        Args:
            walker_id: Actor ID of the walker
            
        Returns:
            EthicalAttributeSchema if found, None otherwise
        """
        return self.registry.get_attributes(walker_id)
    
    def get_vulnerable_walkers(self, threshold: float = 0.5) -> List[int]:
        """
        Get all walker IDs with vulnerability score above threshold.
        
        Args:
            threshold: Minimum vulnerability score (0.0 to 1.0)
            
        Returns:
            List of walker actor IDs
        """
        return self.registry.get_all_vulnerable_actors(threshold)