"""
CARLA State Vector Extractor
Extracts 40D state vectors for RLHF ethical decision-making
"""

import carla
import numpy as np
from typing import List, Dict, Tuple

class StateVectorExtractor:
    """Extracts 40-dimensional state vectors from CARLA"""
    
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()
        
    def extract_state_vector(self, ego_vehicle: carla.Vehicle) -> np.ndarray:
        """
        Extract 40D state vector:
        [0]     velocity_ego
        [1]     num_passengers  
        [2]     lane_position
        [3]     velocity_delta
        [4-6]   num_ped_if_{straight, left, right}
        [7-39]  obstacle_type[3 actions][11 types]
        """
        state = np.zeros(40)
        
        # [0] Ego velocity (m/s)
        velocity = ego_vehicle.get_velocity()
        state[0] = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # [1] Number of passengers (from vehicle attributes)
        state[1] = self._get_num_passengers(ego_vehicle)
        
        # [2] Lane position (0=leftmost, 0.5=center, 1=rightmost)
        state[2] = self._get_lane_position(ego_vehicle)
        
        # [3] Velocity delta (ego vs. traffic average)
        state[3] = self._get_velocity_delta(ego_vehicle)
        
        # [4-6] Pedestrian counts per action
        ped_counts = self._count_pedestrians_per_action(ego_vehicle)
        state[4:7] = ped_counts
        
        # [7-39] Obstacle types per action (3 actions x 11 features)
        obstacles = self._classify_obstacles_per_action(ego_vehicle)
        state[7:40] = obstacles.flatten()
        
        return state
    
    def _get_num_passengers(self, vehicle: carla.Vehicle) -> int:
        """Get number of passengers from vehicle attributes"""
        attributes = vehicle.attributes
        return int(attributes.get('num_passengers', 1))
    
    def _get_lane_position(self, vehicle: carla.Vehicle) -> float:
        """Calculate normalized lane position (0=left, 0.5=center, 1=right)"""
        transform = vehicle.get_transform()
        waypoint = self.map.get_waypoint(transform.location)
        
        if waypoint is None:
            return 0.5
        
        # Get lane boundaries
        lane_width = waypoint.lane_width
        left_lane = waypoint.get_left_lane()
        right_lane = waypoint.get_right_lane()
        
        # Calculate lateral offset
        vehicle_location = transform.location
        lane_center = waypoint.transform.location
        
        lateral_offset = np.sqrt(
            (vehicle_location.x - lane_center.x)**2 +
            (vehicle_location.y - lane_center.y)**2
        )
        
        # Normalize to [0, 1]
        return 0.5 + (lateral_offset / lane_width) if lane_width > 0 else 0.5
    
    def _get_velocity_delta(self, ego_vehicle: carla.Vehicle) -> float:
        """Calculate ego velocity minus average traffic velocity"""
        ego_vel = ego_vehicle.get_velocity()
        ego_speed = np.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
        
        # Get nearby vehicles
        nearby_vehicles = self._get_nearby_actors(ego_vehicle, carla.Vehicle, radius=50.0)
        
        if not nearby_vehicles:
            return 0.0
        
        # Calculate average speed
        speeds = []
        for vehicle in nearby_vehicles:
            vel = vehicle.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            speeds.append(speed)
        
        avg_speed = np.mean(speeds)
        return ego_speed - avg_speed
    
    def _count_pedestrians_per_action(self, ego_vehicle: carla.Vehicle) -> np.ndarray:
        """Count pedestrians in paths for straight, left, right actions"""
        counts = np.zeros(3)
        
        # Project paths for each action
        paths = self._project_action_paths(ego_vehicle)
        
        # Get all pedestrians
        pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
        
        for i, path in enumerate(paths):
            for ped in pedestrians:
                if self._is_in_path(ped, path):
                    counts[i] += 1
        
        return counts
    
    def _classify_obstacles_per_action(self, ego_vehicle: carla.Vehicle) -> np.ndarray:
        """
        Classify obstacles for each action (3 directions x 11 features)
        Features per obstacle: [7 obstacle types + 4 pedestrian attributes]
        """
        obstacles = np.zeros((3, 11))
        
        paths = self._project_action_paths(ego_vehicle)
        
        # Get all actors
        all_actors = self.world.get_actors()
        
        for action_idx, path in enumerate(paths):
            # Check vehicles
            vehicles = all_actors.filter('vehicle.*')
            for vehicle in vehicles:
                if vehicle.id != ego_vehicle.id and self._is_in_path(vehicle, path):
                    obstacles[action_idx, 0] = 1  # Vehicle detected
            
            # Check pedestrians with attributes
            pedestrians = all_actors.filter('walker.pedestrian.*')
            for ped in pedestrians:
                if self._is_in_path(ped, path):
                    # Pedestrian detected (obstacle type 1)
                    obstacles[action_idx, 1] = 1
                    
                    # Extract pedestrian attributes
                    ped_attrs = self._get_pedestrian_attributes(ped)
                    obstacles[action_idx, 7:11] = ped_attrs
            
            # Additional obstacle types (barriers, animals, etc.)
            # TODO: Add detection for other obstacle types
        
        return obstacles
    
    def _get_pedestrian_attributes(self, pedestrian: carla.Walker) -> np.ndarray:
        """
        Extract custom pedestrian attributes: [is_child, is_elderly, is_pregnant, is_disabled]
        Requires custom CARLA build with modified walker blueprints
        """
        attrs = np.zeros(4)
        
        try:
            blueprint = pedestrian.type_id
            attributes = pedestrian.attributes
            
            # Check custom properties (requires modified CARLA)
            attrs[0] = float(attributes.get('is_child', False))
            attrs[1] = float(attributes.get('is_elderly', False))
            attrs[2] = float(attributes.get('is_pregnant', False))
            attrs[3] = float(attributes.get('is_disabled', False))
        except:
            pass
        
        return attrs
    
    def _project_action_paths(self, ego_vehicle: carla.Vehicle) -> List[List[carla.Location]]:
        """Project paths for straight, left, right maneuvers"""
        transform = ego_vehicle.get_transform()
        forward = transform.get_forward_vector()
        location = transform.location
        
        # Project path length (meters)
        path_length = 20.0
        path_width = 3.5
        
        paths = []
        
        # Straight path
        straight_path = [
            carla.Location(
                x=location.x + forward.x * i,
                y=location.y + forward.y * i,
                z=location.z
            ) for i in range(0, int(path_length), 2)
        ]
        paths.append(straight_path)
        
        # Left path (TODO: implement proper curve)
        left_path = straight_path  # Placeholder
        paths.append(left_path)
        
        # Right path (TODO: implement proper curve)
        right_path = straight_path  # Placeholder
        paths.append(right_path)
        
        return paths
    
    def _is_in_path(self, actor: carla.Actor, path: List[carla.Location]) -> bool:
        """Check if actor intersects with path"""
        actor_loc = actor.get_location()
        threshold = 2.0  # meters
        
        for path_point in path:
            distance = np.sqrt(
                (actor_loc.x - path_point.x)**2 +
                (actor_loc.y - path_point.y)**2
            )
            if distance < threshold:
                return True
        
        return False
    
    def _get_nearby_actors(self, reference: carla.Actor, actor_type, radius: float):
        """Get actors of specific type within radius"""
        ref_loc = reference.get_location()
        all_actors = self.world.get_actors().filter(f'{actor_type.__name__.lower()}.*')
        
        nearby = []
        for actor in all_actors:
            if actor.id == reference.id:
                continue
            
            loc = actor.get_location()
            distance = np.sqrt(
                (loc.x - ref_loc.x)**2 +
                (loc.y - ref_loc.y)**2
            )
            
            if distance < radius:
                nearby.append(actor)
        
        return nearby
