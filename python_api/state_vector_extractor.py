from __future__ import annotations

"""
StateVectorExtractor
====================
Converts a DataCollector snapshot frame into the 40-dimensional state vector

State vector layout:

    Index  0    : velocity_ego          — ego vehicle speed (m/s)
    Index  1    : num_passengers        — occupants in ego vehicle
    Index  2    : lane_position         — lateral offset from lane centre (m)
    Index  3    : velocity_delta        — change in speed since last step (m/s)
    Index  4    : num_ped_if_straight   — pedestrian count in straight path
    Index  5    : num_ped_if_left       — pedestrian count in left-swerve path
    Index  6    : num_ped_if_right      — pedestrian count in right-swerve path
    Indices 7–39: obstacle_features     — 3 directions × 11 ethical features (33 total)

Obstacle feature layout per direction (straight=7–17, left=18–28, right=29–39):

    +0  : actor_count          — number of actors in this direction (normalised / 5.0)
    +1  : has_child            — any child present (binary)
    +2  : has_elderly          — any elderly present (binary)
    +3  : has_wheelchair       — any wheelchair user present (binary)
    +4  : has_cane             — any cane user present (binary)
    +5  : has_blind            — any blind actor present (binary)
    +6  : has_pregnant         — any pregnant actor present (binary)
    +7  : has_emergency        — any emergency worker present (binary)
    +8  : has_healthcare       — any healthcare worker present (binary)
    +9  : max_vulnerability    — highest vulnerability score in direction (0–1)
    +10 : avg_vulnerability    — mean vulnerability score in direction (0–1)

These features make ethical actor attributes directly visible to Team A's reward
model so the learned reward signal can distinguish between, say, a child and an
adult in the vehicle's path.

Direction classification (based on bearing from ego heading):
    straight : actor within ±22.5° of heading
    left     : actor between -90° and -22.5° of heading  (negative = left turn)
    right    : actor between +22.5° and +90° of heading

Actors outside ±90° (behind the vehicle) are ignored.

Thread safety: StateVectorExtractor is NOT thread-safe. Create one instance per
DataCollector / simulation thread.
"""

import math
from typing import Any, Dict, List, Optional

from .ethical_attributes import AgeGroup, Disability, SocialRole


# Direction index constants for obstacle feature block (7 base indices apart from 7).
_DIR_STRAIGHT = 0
_DIR_LEFT     = 1
_DIR_RIGHT    = 2

_FEATURES_PER_DIR = 11
_OBS_BASE_IDX     = 7          # obstacle features start at index 7
_MAX_DETECTION_RANGE_M = 50.0  # ignore actors beyond this distance


class StateVectorExtractor:
    """
    Stateful converter from DataCollector frames to 40-dim state vectors.

    Stateful because velocity_delta (index 3) requires the previous velocity.

    Args:
        num_passengers: Fixed passenger count for the ego vehicle. Defaults to 1
                        (driver only). Pass the actual vehicle occupancy if known.
        straight_half_angle_deg: Half-angle of the "straight" cone in degrees.
                                 Default 22.5° → straight cone spans ±22.5°.
        side_max_angle_deg:      Outer edge of left/right detection cones.
                                 Default 90° → actors more than 90° to the side
                                 are considered behind and ignored.
        max_range_m:             Maximum detection range in metres. Actors beyond
                                 this are excluded from all direction bins.

    Usage::

        extractor = StateVectorExtractor(num_passengers=2)

        for frame in collector.frames():
            state = extractor.extract(frame)   # list[float], length 40
            action = team_a_agent.get_action(state)
    """

    def __init__(
        self,
        num_passengers: int = 1,
        straight_half_angle_deg: float = 22.5,
        side_max_angle_deg: float = 90.0,
        max_range_m: float = _MAX_DETECTION_RANGE_M,
    ) -> None:
        self._num_passengers        = float(num_passengers)
        self._straight_half_rad     = math.radians(straight_half_angle_deg)
        self._side_max_rad          = math.radians(side_max_angle_deg)
        self._max_range_m           = max_range_m
        self._prev_speed: Optional[float] = None   # for velocity_delta

    def extract(self, frame: Dict[str, Any]) -> List[float]:
        """
        Convert one DataCollector snapshot frame into a 40-dim state vector.

        Args:
            frame: Dict produced by DataCollector.collect() with keys:
                   vehicle_state, ethical_context, scene_metadata, etc.

        Returns:
            List of 40 floats in the layout documented above.
        """
        vehicle_state   = frame.get("vehicle_state", {})
        ethical_context = frame.get("ethical_context", {})
        visible_actors  = ethical_context.get("visible_actors", [])

        # ── Indices 0–3: ego kinematics ─────────────────────────────────────

        vel_vec  = vehicle_state.get("velocity", [0.0, 0.0, 0.0])
        speed    = math.sqrt(vel_vec[0]**2 + vel_vec[1]**2 + vel_vec[2]**2)

        vel_delta = speed - self._prev_speed if self._prev_speed is not None else 0.0
        self._prev_speed = speed

        pos = vehicle_state.get("position", [0.0, 0.0, 0.0])
        # lane_position: we use the Y component (left/right offset) as a proxy.
        # CARLAAdapter stores position in CARLA's left-handed coordinate system
        # where Y increases to the right. Positive = drifting right of center.
        lane_position = pos[1]

        # ── Indices 4–6 + 7–39: actor features by direction ─────────────────

        ego_x, ego_y = pos[0], pos[1]

        # Ego heading from velocity vector (fallback to 0° if stationary)
        if speed > 0.01:
            heading_rad = math.atan2(vel_vec[1], vel_vec[0])
        else:
            heading_rad = 0.0

        # Bin actors into directions
        straight_actors: List[Dict[str, Any]] = []
        left_actors:     List[Dict[str, Any]] = []
        right_actors:    List[Dict[str, Any]] = []

        for actor in visible_actors:
            actor_pos = actor.get("position", [0.0, 0.0, 0.0])
            dx = actor_pos[0] - ego_x
            dy = actor_pos[1] - ego_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self._max_range_m:
                continue

            bearing = math.atan2(dy, dx)
            # Relative angle: positive = right of heading, negative = left
            rel = _normalise_angle(bearing - heading_rad)
            abs_rel = abs(rel)

            if abs_rel <= self._straight_half_rad:
                straight_actors.append(actor)
            elif self._straight_half_rad < abs_rel <= self._side_max_rad:
                if rel < 0:
                    left_actors.append(actor)
                else:
                    right_actors.append(actor)
            # else: behind the vehicle — ignored

        num_straight = float(len(straight_actors))
        num_left     = float(len(left_actors))
        num_right    = float(len(right_actors))

        straight_feats = _build_obstacle_features(straight_actors)
        left_feats     = _build_obstacle_features(left_actors)
        right_feats    = _build_obstacle_features(right_actors)

        # ── Assemble the 40-dim vector ───────────────────────────────────────

        state: List[float] = [0.0] * 40

        state[0] = speed
        state[1] = self._num_passengers
        state[2] = lane_position
        state[3] = vel_delta
        state[4] = num_straight
        state[5] = num_left
        state[6] = num_right

        # Obstacle features: 11 per direction, starting at index 7
        for i, val in enumerate(straight_feats):
            state[_OBS_BASE_IDX + _DIR_STRAIGHT * _FEATURES_PER_DIR + i] = val

        for i, val in enumerate(left_feats):
            state[_OBS_BASE_IDX + _DIR_LEFT * _FEATURES_PER_DIR + i] = val

        for i, val in enumerate(right_feats):
            state[_OBS_BASE_IDX + _DIR_RIGHT * _FEATURES_PER_DIR + i] = val

        return state

    def reset(self) -> None:
        """
        Reset stateful fields between episodes.

        Call this at the start of each new trajectory / rollout so that
        velocity_delta is computed correctly from the new episode's first step.
        """
        self._prev_speed = None


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_angle(angle: float) -> float:
    """Wrap angle into (−π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _build_obstacle_features(actors: List[Dict[str, Any]]) -> List[float]:
    """
    Build the 11-element feature vector for one direction bin.

    Returns a list of 11 floats. All fields default to 0.0 when actors is empty.

    Feature layout:
        0  : actor_count (normalised by 5.0)
        1  : has_child
        2  : has_elderly
        3  : has_wheelchair
        4  : has_cane
        5  : has_blind
        6  : has_pregnant
        7  : has_emergency
        8  : has_healthcare
        9  : max_vulnerability_score
        10 : avg_vulnerability_score
    """
    if not actors:
        return [0.0] * _FEATURES_PER_DIR

    has_child       = 0.0
    has_elderly     = 0.0
    has_wheelchair  = 0.0
    has_cane        = 0.0
    has_blind       = 0.0
    has_pregnant    = 0.0
    has_emergency   = 0.0
    has_healthcare  = 0.0
    vuln_scores: List[float] = []

    for actor in actors:
        age_group  = actor.get("age_group")
        disability = actor.get("disability")
        social_role = actor.get("social_role")
        vuln       = actor.get("vulnerability_score")

        if age_group == AgeGroup.CHILD.value:
            has_child = 1.0
        if age_group == AgeGroup.ELDERLY.value:
            has_elderly = 1.0
        if disability == Disability.WHEELCHAIR.value:
            has_wheelchair = 1.0
        if disability == Disability.CANE.value:
            has_cane = 1.0
        if disability == Disability.BLIND.value:
            has_blind = 1.0

        # pregnancy is not directly in the frame's actor dict but is captured
        # in vulnerability_score; we expose it via vulnerability features.

        if social_role == SocialRole.EMERGENCY.value:
            has_emergency = 1.0
        if social_role == SocialRole.HEALTHCARE.value:
            has_healthcare = 1.0

        if vuln is not None:
            vuln_scores.append(float(vuln))

    max_vuln = max(vuln_scores) if vuln_scores else 0.0
    avg_vuln = sum(vuln_scores) / len(vuln_scores) if vuln_scores else 0.0

    # actor_count normalised so a full lane (5 actors) = 1.0
    count_norm = min(len(actors) / 5.0, 1.0)

    return [
        count_norm,
        has_child,
        has_elderly,
        has_wheelchair,
        has_cane,
        has_blind,
        has_pregnant,
        has_emergency,
        has_healthcare,
        max_vuln,
        avg_vuln,
    ]