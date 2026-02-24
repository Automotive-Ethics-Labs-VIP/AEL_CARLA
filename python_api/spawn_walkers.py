from .adapter import SimulatorAdapter
from .ethical_attributes import EthicalAttributeSchema, AgeGroup, Disability, SocialRole
from .actor_registry import EthicalActorRegistry
import random
from typing import List, Optional


class EthicalWalkerSpawner:
    def __init__(
        self,
        adapter: SimulatorAdapter,
        registry: Optional[EthicalActorRegistry] = None,
    ) -> None:
        self._adapter  = adapter 
        self.registry  = registry if registry is not None else EthicalActorRegistry()
        self._bp_lib   = adapter.get_blueprint_library()
        self.spawned_walkers: List = []
     
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

    def spawn_walker(self, spawn_point, walker_bp=None):
        if walker_bp is None:
            walker_bp = random.choice(self._bp_lib.filter('walker.pedestrian.*'))

        actor_id = self._adapter.spawn_walker(spawn_point, walker_bp) 
        if actor_id is None:
            raise RuntimeError(f"Failed to spawn walker at {spawn_point}")

        attributes = self._generate_ethical_attributes(walker_bp)
        self.registry.register(actor_id, attributes)

        actor = self._adapter.get_actor(actor_id)
        if actor is not None:
            self.spawned_walkers.append(actor)
        return actor

    def spawn_walkers_batch(self, spawn_points, num_walkers=None):
        if num_walkers is None:
            num_walkers = len(spawn_points)
        num_walkers = min(num_walkers, len(spawn_points))

        walker_blueprints = self._bp_lib.filter('walker.pedestrian.*')
        commands = []
        bp_attr_pairs = []

        for i in range(num_walkers):
            walker_bp  = random.choice(walker_blueprints)
            attributes = self._generate_ethical_attributes(walker_bp)
            bp_attr_pairs.append((walker_bp, attributes))
            commands.append(self._adapter.make_spawn_command(walker_bp, spawn_points[i]))

        batch_results = self._adapter.spawn_walkers_batch(commands)

        walkers = []
        for i, response in enumerate(batch_results):
            if response.error:
                continue
            actor_id = response.actor_id
            _, attributes = bp_attr_pairs[i]
            self.registry.register(actor_id, attributes)
            actor = self._adapter.get_actor(actor_id)
            if actor is not None:
                self.spawned_walkers.append(actor)
                walkers.append(actor)
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