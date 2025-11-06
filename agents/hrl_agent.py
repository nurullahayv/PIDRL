"""
Hierarchical Reinforcement Learning Agent

Phase 5: HRL agent with multiple levels of decision making:
- High-level: Strategy (attack, evade, patrol, reposition)
- Mid-level: Tactics (target selection, maneuver type)
- Low-level: Motor control (pursuit/evasion skills)

This enables complex strategic behavior in adversarial scenarios.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from enum import Enum


class Strategy(Enum):
    """High-level strategic behaviors."""
    ATTACK = 0
    EVADE = 1
    PATROL = 2
    REPOSITION = 3


class Maneuver(Enum):
    """Mid-level tactical maneuvers."""
    INTERCEPT = 0
    PURSUE = 1
    FLANK = 2
    VERTICAL_LOOP = 3
    SPLIT_S = 4
    BOOM_AND_ZOOM = 5


class HRLAgent:
    """
    Hierarchical RL agent with three levels:

    Level 1 (High): Strategy selection
    Level 2 (Mid): Tactic selection given strategy
    Level 3 (Low): Motor control given tactic

    Each level can be:
    - Rule-based (hand-crafted)
    - RL-based (learned policy)
    - Hybrid (combination)
    """

    def __init__(
        self,
        high_level_policy=None,
        mid_level_policy=None,
        low_level_policies: Optional[Dict] = None,
    ):
        """
        Initialize HRL agent.

        Args:
            high_level_policy: Strategy selection policy
            mid_level_policy: Tactic selection policy
            low_level_policies: Dict of low-level policies for each tactic
        """
        self.high_level_policy = high_level_policy or self._default_high_level
        self.mid_level_policy = mid_level_policy or self._default_mid_level
        self.low_level_policies = low_level_policies or self._default_low_level_policies()

        # State tracking
        self.current_strategy = Strategy.PATROL
        self.current_maneuver = Maneuver.INTERCEPT
        self.strategy_duration = 0
        self.maneuver_duration = 0

    def _default_high_level(self, observation: Dict) -> Strategy:
        """
        Default high-level policy (rule-based).

        Args:
            observation: Environment observation with tactical info

        Returns:
            Selected strategy
        """
        # Simple rules for strategy selection
        enemy_distance = observation.get("enemy_distance", float('inf'))
        own_health = observation.get("own_health", 100.0)
        enemy_health = observation.get("enemy_health", 100.0)
        in_no_fly_zone = observation.get("in_no_fly_zone", False)

        # If in no-fly zone, evade immediately
        if in_no_fly_zone:
            return Strategy.EVADE

        # If low health, evade
        if own_health < 30.0:
            return Strategy.EVADE

        # If enemy close and we have health advantage, attack
        if enemy_distance < 200.0 and own_health > enemy_health:
            return Strategy.ATTACK

        # If enemy far, patrol or reposition
        if enemy_distance > 500.0:
            return Strategy.PATROL

        # Default: Attack
        return Strategy.ATTACK

    def _default_mid_level(self, strategy: Strategy, observation: Dict) -> Maneuver:
        """
        Default mid-level policy (rule-based).

        Args:
            strategy: Current high-level strategy
            observation: Environment observation

        Returns:
            Selected maneuver
        """
        enemy_distance = observation.get("enemy_distance", float('inf'))
        relative_altitude = observation.get("relative_altitude", 0.0)

        if strategy == Strategy.ATTACK:
            # Choose maneuver based on geometry
            if enemy_distance > 300.0:
                return Maneuver.INTERCEPT
            elif relative_altitude > 50.0:
                # We're above enemy: boom and zoom
                return Maneuver.BOOM_AND_ZOOM
            else:
                return Maneuver.PURSUE

        elif strategy == Strategy.EVADE:
            # Evasive maneuvers
            if relative_altitude < -50.0:
                # Enemy above: split-s to evade
                return Maneuver.SPLIT_S
            else:
                return Maneuver.VERTICAL_LOOP

        elif strategy == Strategy.PATROL:
            return Maneuver.INTERCEPT

        elif strategy == Strategy.REPOSITION:
            return Maneuver.FLANK

        return Maneuver.PURSUE

    def _default_low_level_policies(self) -> Dict:
        """
        Default low-level policies for each maneuver.

        Returns:
            Dictionary mapping maneuvers to control policies
        """
        return {
            Maneuver.INTERCEPT: self._intercept_control,
            Maneuver.PURSUE: self._pursue_control,
            Maneuver.FLANK: self._flank_control,
            Maneuver.VERTICAL_LOOP: self._vertical_loop_control,
            Maneuver.SPLIT_S: self._split_s_control,
            Maneuver.BOOM_AND_ZOOM: self._boom_zoom_control,
        }

    def _intercept_control(self, observation: Dict) -> np.ndarray:
        """Intercept maneuver: Move directly towards enemy."""
        to_enemy = observation.get("to_enemy", np.array([1.0, 0.0, 0.0]))
        distance = np.linalg.norm(to_enemy)

        if distance > 0.1:
            direction = to_enemy / distance
            return direction  # Acceleration in enemy direction

        return np.zeros(3)

    def _pursue_control(self, observation: Dict) -> np.ndarray:
        """Pursuit maneuver: Follow enemy with lead prediction."""
        to_enemy = observation.get("to_enemy", np.array([1.0, 0.0, 0.0]))
        enemy_velocity = observation.get("enemy_velocity", np.zeros(3))

        # Lead pursuit: Aim ahead of enemy
        lead_point = to_enemy + enemy_velocity * 0.5
        distance = np.linalg.norm(lead_point)

        if distance > 0.1:
            direction = lead_point / distance
            return direction

        return np.zeros(3)

    def _flank_control(self, observation: Dict) -> np.ndarray:
        """Flank maneuver: Move perpendicular to enemy."""
        to_enemy = observation.get("to_enemy", np.array([1.0, 0.0, 0.0]))
        distance = np.linalg.norm(to_enemy)

        if distance > 0.1:
            forward = to_enemy / distance
            # Create perpendicular vector
            perpendicular = np.cross(forward, [0, 0, 1])
            if np.linalg.norm(perpendicular) < 0.1:
                perpendicular = np.cross(forward, [0, 1, 0])

            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
            return perpendicular

        return np.zeros(3)

    def _vertical_loop_control(self, observation: Dict) -> np.ndarray:
        """Vertical loop maneuver: Pull up."""
        return np.array([0.0, 0.0, 1.0])  # Accelerate upward

    def _split_s_control(self, observation: Dict) -> np.ndarray:
        """Split-S maneuver: Roll and pull down."""
        return np.array([0.0, 0.0, -1.0])  # Accelerate downward

    def _boom_zoom_control(self, observation: Dict) -> np.ndarray:
        """Boom and zoom: Dive on enemy."""
        to_enemy = observation.get("to_enemy", np.array([1.0, 0.0, 0.0]))
        # Dive vector (towards enemy + downward)
        dive = to_enemy.copy()
        dive[2] -= 0.5  # Add downward component
        distance = np.linalg.norm(dive)

        if distance > 0.1:
            return dive / distance

        return np.zeros(3)

    def predict(
        self,
        observation: Dict,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action using hierarchical decision making.

        Args:
            observation: Environment observation (should be Dict with tactical info)
            deterministic: Whether to use deterministic policies

        Returns:
            action: Acceleration command
            state: None
        """
        # Level 1: Strategy selection (executed every N steps)
        if self.strategy_duration == 0 or self.strategy_duration > 50:
            self.current_strategy = self.high_level_policy(observation)
            self.strategy_duration = 0

        self.strategy_duration += 1

        # Level 2: Tactic selection (executed every M steps)
        if self.maneuver_duration == 0 or self.maneuver_duration > 20:
            self.current_maneuver = self.mid_level_policy(self.current_strategy, observation)
            self.maneuver_duration = 0

        self.maneuver_duration += 1

        # Level 3: Motor control (executed every step)
        low_level_policy = self.low_level_policies.get(self.current_maneuver, self._pursue_control)
        action = low_level_policy(observation)

        # Normalize to [-1, 1]
        action_mag = np.linalg.norm(action)
        if action_mag > 1e-6:
            action = action / action_mag

        return action, None

    def reset(self):
        """Reset agent state."""
        self.current_strategy = Strategy.PATROL
        self.current_maneuver = Maneuver.INTERCEPT
        self.strategy_duration = 0
        self.maneuver_duration = 0
