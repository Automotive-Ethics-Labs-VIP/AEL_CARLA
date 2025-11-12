"""
CARLA Training Data Generator
Generates ethical decision scenarios for RLHF training
"""

import carla
import numpy as np
import json
import time
from pathlib import Path
from state_vector_extractor import StateVectorExtractor

class TrainingDataGenerator:
    """Generates training scenarios in CARLA"""
    
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.extractor = StateVectorExtractor(self.world)
        self.scenarios_generated = 0
        
    def generate_scenario(self, scenario_type: str):
        """Generate a single ethical scenario"""
        
        # Spawn ego vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        
        try:
            # Setup scenario based on type
            if scenario_type == "unavoidable_collision":
                self._setup_unavoidable_collision()
            elif scenario_type == "trolley_problem":
                self._setup_trolley_problem()
            elif scenario_type == "protected_class":
                self._setup_protected_class_dilemma()
            # ... more scenario types
            
            # Let scenario develop
            time.sleep(2.0)
            
            # Extract state vector
            state_vector = self.extractor.extract_state_vector(ego_vehicle)
            
            # Record scenario
            scenario_data = {
                'scenario_type': scenario_type,
                'state_vector': state_vector.tolist(),
                'timestamp': time.time(),
                'scenario_id': self.scenarios_generated
            }
            
            self.scenarios_generated += 1
            return scenario_data
            
        finally:
            # Cleanup
            ego_vehicle.destroy()
    
    def _setup_unavoidable_collision(self):
        """Setup unavoidable collision scenario"""
        # TODO: Implement scenario
        pass
    
    def _setup_trolley_problem(self):
        """Setup trolley problem variant"""
        # TODO: Implement scenario
        pass
    
    def _setup_protected_class_dilemma(self):
        """Setup protected class scenario"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Spawn pedestrians with specific attributes
        walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]
        
        # Child
        child_bp = walker_bp
        child_bp.set_attribute('is_child', 'true')
        child = self.world.spawn_actor(child_bp, spawn_points[1])
        
        # Elderly
        elderly_bp = walker_bp
        elderly_bp.set_attribute('is_elderly', 'true')
        elderly = self.world.spawn_actor(elderly_bp, spawn_points[2])
        
        # TODO: Setup scenario dynamics
    
    def generate_dataset(self, num_scenarios: int, output_path: Path):
        """Generate full training dataset"""
        
        dataset = []
        
        scenario_types = [
            "unavoidable_collision",
            "trolley_problem", 
            "protected_class",
            "property_vs_life",
            "legal_vs_ethical"
        ]
        
        scenarios_per_type = num_scenarios // len(scenario_types)
        
        print(f"Generating {num_scenarios} scenarios...")
        
        for scenario_type in scenario_types:
            print(f"  Generating {scenarios_per_type} {scenario_type} scenarios...")
            
            for i in range(scenarios_per_type):
                try:
                    scenario = self.generate_scenario(scenario_type)
                    dataset.append(scenario)
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Progress: {i + 1}/{scenarios_per_type}")
                        
                except Exception as e:
                    print(f"    Error generating scenario: {e}")
                    continue
        
        # Save dataset
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"training_data_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nDataset saved to {output_file}")
        print(f"Total scenarios: {len(dataset)}")


if __name__ == "__main__":
    generator = TrainingDataGenerator(host='localhost', port=2000)
    
    # Generate 1000 scenarios
    output_path = Path("/gpfs/projects/$USER/training-data")
    generator.generate_dataset(num_scenarios=1000, output_path=output_path)
