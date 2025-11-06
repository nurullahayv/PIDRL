"""
Strategic Dogfight Environment with No-Fly Zones

Phase 5: Complete strategic environment with:
- HRL agents
- No-fly zones (SAM sites)
- Strategic objectives
- Large map with terrain
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional


class NoFlyZone:
    """
    Represents a no-fly zone (SAM site).

    Hemispherical danger zone that penalizes agents entering it.
    """

    def __init__(self, center: np.ndarray, radius: float, penalty: float = -10.0):
        self.center = center.astype(np.float32)
        self.radius = radius
        self.penalty = penalty

    def is_inside(self, position: np.ndarray) -> bool:
        """Check if position is inside no-fly zone."""
        distance = np.linalg.norm(position - self.center)
        # Only check if above ground and within radius
        if position[2] < self.center[2]:
            return False
        return distance < self.radius

    def get_penalty(self, position: np.ndarray) -> float:
        """Get penalty for being at position."""
        if self.is_inside(position):
            # Penalty increases as you go deeper
            distance = np.linalg.norm(position - self.center)
            depth = 1.0 - (distance / self.radius)
            return self.penalty * depth
        return 0.0


class StrategicDogfightEnv(gym.Env):
    """
    Strategic dogfight environment with HRL and no-fly zones.

    Features:
    - Large map (2000x2000x1000)
    - Multiple no-fly zones
    - Strategic waypoints
    - Complex reward structure
    - Support for HRL agents
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_size: float = 2000.0,
        num_no_fly_zones: int = 5,
        max_velocity: float = 50.0,
        max_acceleration: float = 5.0,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.map_size = map_size
        self.num_no_fly_zones = num_no_fly_zones
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.render_mode = render_mode

        # Observation: Tactical information for HRL
        # [own_pos (3), own_vel (3), own_orient (3),
        #  enemy_pos (3), enemy_vel (3), enemy_orient (3),
        #  nearest_no_fly_zone (4), tactical_info (6)]
        # Total: 28 dimensions
        self.observation_space = spaces.Box(
            low=-map_size,
            high=map_size,
            shape=(28,),
            dtype=np.float32
        )

        # Action: [ax, ay, az]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Agent and enemy states
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.agent_orient = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.agent_health = 100.0

        self.enemy_pos = np.zeros(3, dtype=np.float32)
        self.enemy_vel = np.zeros(3, dtype=np.float32)
        self.enemy_orient = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.enemy_health = 100.0

        # No-fly zones
        self.no_fly_zones: List[NoFlyZone] = []

        self.step_count = 0
        self.max_steps = 5000

    def _create_no_fly_zones(self):
        """Create no-fly zones randomly across the map."""
        self.no_fly_zones = []

        for i in range(self.num_no_fly_zones):
            # Random position (avoid edges)
            center = self.np_random.uniform(
                -self.map_size / 3,
                self.map_size / 3,
                size=3
            )
            center[2] = 0.0  # Ground level

            # Random radius
            radius = self.np_random.uniform(100.0, 300.0)

            # Varying penalties
            penalty = self.np_random.uniform(-5.0, -15.0)

            zone = NoFlyZone(center, radius, penalty)
            self.no_fly_zones.append(zone)

    def _get_tactical_observation(self) -> Dict:
        """
        Get tactical observation for HRL agent.

        Returns dictionary with tactical information.
        """
        # Vector to enemy
        to_enemy = self.enemy_pos - self.agent_pos
        enemy_distance = np.linalg.norm(to_enemy)

        # Relative altitude
        relative_altitude = self.agent_pos[2] - self.enemy_pos[2]

        # Find nearest no-fly zone
        nearest_nfz_dist = float('inf')
        nearest_nfz_pos = np.zeros(3)
        in_no_fly_zone = False

        for zone in self.no_fly_zones:
            dist = np.linalg.norm(zone.center - self.agent_pos)
            if dist < nearest_nfz_dist:
                nearest_nfz_dist = dist
                nearest_nfz_pos = zone.center

            if zone.is_inside(self.agent_pos):
                in_no_fly_zone = True

        return {
            "own_pos": self.agent_pos,
            "own_vel": self.agent_vel,
            "own_orient": self.agent_orient,
            "own_health": self.agent_health,
            "enemy_pos": self.enemy_pos,
            "enemy_velocity": self.enemy_vel,
            "enemy_health": self.enemy_health,
            "to_enemy": to_enemy,
            "enemy_distance": enemy_distance,
            "relative_altitude": relative_altitude,
            "nearest_nfz_dist": nearest_nfz_dist,
            "in_no_fly_zone": in_no_fly_zone,
        }

    def _get_observation_vector(self) -> np.ndarray:
        """Convert tactical observation to vector for RL."""
        tactical = self._get_tactical_observation()

        # Encode tactical info
        nearest_nfz = np.array([
            tactical["nearest_nfz_dist"],
            1.0 if tactical["in_no_fly_zone"] else 0.0,
            0.0,  # Reserved
            0.0,  # Reserved
        ])

        tactical_info = np.array([
            tactical["enemy_distance"],
            tactical["relative_altitude"],
            tactical["own_health"] / 100.0,
            tactical["enemy_health"] / 100.0,
            0.0,  # Reserved
            0.0,  # Reserved
        ])

        obs = np.concatenate([
            self.agent_pos,
            self.agent_vel,
            self.agent_orient,
            self.enemy_pos,
            self.enemy_vel,
            self.enemy_orient,
            nearest_nfz,
            tactical_info,
        ])

        return obs.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Random starting positions (far apart)
        self.agent_pos = self.np_random.uniform(
            [-self.map_size / 4, -self.map_size / 4, 100],
            [-self.map_size / 8, -self.map_size / 8, 300],
            size=3
        ).astype(np.float32)

        self.enemy_pos = self.np_random.uniform(
            [self.map_size / 8, self.map_size / 8, 100],
            [self.map_size / 4, self.map_size / 4, 300],
            size=3
        ).astype(np.float32)

        # Random velocities
        self.agent_vel = self.np_random.uniform(-10, 10, size=3).astype(np.float32)
        self.enemy_vel = self.np_random.uniform(-10, 10, size=3).astype(np.float32)

        # Reset orientations
        self.agent_orient = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.enemy_orient = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Reset health
        self.agent_health = 100.0
        self.enemy_health = 100.0

        # Create no-fly zones
        self._create_no_fly_zones()

        self.step_count = 0

        return self._get_observation_vector(), self._get_tactical_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        self.step_count += 1

        # Agent action
        action = np.clip(action, -1.0, 1.0)
        agent_acc = action * self.max_acceleration

        # Update agent
        self.agent_vel += agent_acc * self.dt
        vel_mag = np.linalg.norm(self.agent_vel)
        if vel_mag > self.max_velocity:
            self.agent_vel = self.agent_vel / vel_mag * self.max_velocity

        self.agent_pos += self.agent_vel * self.dt

        if vel_mag > 1.0:
            self.agent_orient = self.agent_vel / vel_mag

        # Enemy behavior (simple evasion)
        to_agent = self.agent_pos - self.enemy_pos
        distance = np.linalg.norm(to_agent)

        if distance > 0.1:
            evade_dir = -to_agent / distance
            enemy_acc = evade_dir * self.max_acceleration * 0.6
        else:
            enemy_acc = np.zeros(3)

        self.enemy_vel += enemy_acc * self.dt
        enemy_vel_mag = np.linalg.norm(self.enemy_vel)
        if enemy_vel_mag > self.max_velocity * 0.8:
            self.enemy_vel = self.enemy_vel / enemy_vel_mag * (self.max_velocity * 0.8)

        self.enemy_pos += self.enemy_vel * self.dt

        # Keep in bounds
        self.agent_pos = np.clip(self.agent_pos, -self.map_size / 2, self.map_size / 2)
        self.enemy_pos = np.clip(self.enemy_pos, -self.map_size / 2, self.map_size / 2)

        # Compute reward
        reward = 0.0

        # Distance-based reward
        reward += 1.0 - (distance / (self.map_size / 2))

        # No-fly zone penalties
        nfz_penalty = 0.0
        for zone in self.no_fly_zones:
            nfz_penalty += zone.get_penalty(self.agent_pos)

        reward += nfz_penalty

        # Health-based reward
        if distance < 50.0:
            # Close engagement: Damage enemy
            self.enemy_health -= 0.1
            reward += 0.5

        # Termination
        terminated = self.enemy_health <= 0 or self.agent_health <= 0
        truncated = self.step_count >= self.max_steps

        return self._get_observation_vector(), reward, terminated, truncated, self._get_tactical_observation()

    def render(self):
        """Render environment (TODO: implement 3D visualization)."""
        pass

    def close(self):
        """Close environment."""
        pass
