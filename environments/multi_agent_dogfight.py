"""
Multi-Agent Dogfight Environment

Phase 4: N vs N dogfight where multiple agents engage each other.

Features:
- Multiple agents (N vs N or free-for-all)
- Target selection (hierarchical decision)
- Collision detection
- Multi-agent observations
- Team coordination (optional)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional


class Aircraft:
    """Represents a single aircraft in the simulation."""

    def __init__(self, id: int, team: int, position: np.ndarray, velocity: np.ndarray):
        self.id = id
        self.team = team  # 0 or 1 for team-based, -1 for free-for-all
        self.position = position.astype(np.float32)
        self.velocity = velocity.astype(np.float32)
        self.orientation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.alive = True
        self.target_locked = None  # ID of locked target
        self.health = 100.0


class MultiAgentDogfightEnv(gym.Env):
    """
    Multi-agent dogfight environment.

    Each agent:
    - Observes: Local state + nearby agents
    - Acts: [acceleration, target_selection]
    - Decides: Which enemy to engage (high-level)
              How to engage (low-level)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        num_agents: int = 4,
        num_teams: int = 2,  # 2 teams or 0 for free-for-all
        arena_size: float = 1000.0,
        max_velocity: float = 50.0,
        max_acceleration: float = 5.0,
        fov_range: float = 300.0,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.num_teams = num_teams
        self.arena_size = arena_size
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.fov_range = fov_range
        self.dt = dt
        self.render_mode = render_mode

        # Observation: [own_state (9), nearby_agents (N*9)]
        # For simplicity: Fixed size observation with max nearby agents
        max_nearby = 5
        obs_dim = 9 + max_nearby * 9  # Own state + up to 5 nearby agents
        self.observation_space = spaces.Box(
            low=-arena_size,
            high=arena_size,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: [ax, ay, az] for acceleration
        # Target selection is handled separately (high-level decision)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Aircraft list
        self.aircraft: List[Aircraft] = []
        self.step_count = 0
        self.max_steps = 3000

    def _create_aircraft(self):
        """Initialize all aircraft."""
        self.aircraft = []

        agents_per_team = self.num_agents // self.num_teams if self.num_teams > 0 else self.num_agents

        for i in range(self.num_agents):
            team = i // agents_per_team if self.num_teams > 0 else -1

            # Spawn position based on team
            if team == 0:
                pos = self.np_random.uniform([-self.arena_size / 4, -self.arena_size / 4, 0],
                                               [-self.arena_size / 8, -self.arena_size / 8, 100],
                                               size=3)
            elif team == 1:
                pos = self.np_random.uniform([self.arena_size / 8, self.arena_size / 8, 0],
                                               [self.arena_size / 4, self.arena_size / 4, 100],
                                               size=3)
            else:
                pos = self.np_random.uniform(-self.arena_size / 4, self.arena_size / 4, size=3)

            vel = self.np_random.uniform(-10, 10, size=3)

            aircraft = Aircraft(i, team, pos, vel)
            self.aircraft.append(aircraft)

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for specific agent.

        Observation includes:
        - Own state: [pos (3), vel (3), orient (3)]
        - Nearby agents: Up to 5 nearest [relative_pos (3), relative_vel (3), team (1), health (1), alive (1)]
        """
        agent = self.aircraft[agent_id]

        # Own state
        own_state = np.concatenate([agent.position, agent.velocity, agent.orientation])

        # Find nearby agents
        nearby = []
        for other in self.aircraft:
            if other.id == agent_id or not other.alive:
                continue

            distance = np.linalg.norm(other.position - agent.position)
            if distance < self.fov_range:
                relative_pos = other.position - agent.position
                relative_vel = other.velocity - agent.velocity
                other_info = np.array([other.team, other.health / 100.0, 1.0 if other.alive else 0.0])
                nearby.append((distance, np.concatenate([relative_pos, relative_vel, other_info])))

        # Sort by distance and take top 5
        nearby.sort(key=lambda x: x[0])
        nearby = [x[1] for x in nearby[:5]]

        # Pad to 5 agents
        while len(nearby) < 5:
            nearby.append(np.zeros(9))

        nearby_state = np.concatenate(nearby)

        return np.concatenate([own_state, nearby_state])

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self._create_aircraft()
        self.step_count = 0

        # Return observation for first agent (in multi-agent, we'd return dict)
        obs = self._get_observation(0)
        info = {"num_alive": sum(1 for a in self.aircraft if a.alive)}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Note: This is simplified single-agent interface.
        Full multi-agent would use MultiAgentEnv with action dict.
        """
        self.step_count += 1

        # Update agent 0 (controlled agent)
        agent = self.aircraft[0]
        if agent.alive:
            action = np.clip(action, -1.0, 1.0)
            acc = action * self.max_acceleration

            agent.velocity += acc * self.dt
            vel_mag = np.linalg.norm(agent.velocity)
            if vel_mag > self.max_velocity:
                agent.velocity = agent.velocity / vel_mag * self.max_velocity

            agent.position += agent.velocity * self.dt

            if vel_mag > 1.0:
                agent.orientation = agent.velocity / vel_mag

        # Update other agents (simple behavior)
        for other in self.aircraft[1:]:
            if not other.alive:
                continue

            # Simple AI: Move towards nearest enemy
            nearest_enemy = None
            min_dist = float('inf')

            for target in self.aircraft:
                if target.id == other.id or not target.alive:
                    continue
                if self.num_teams > 0 and target.team == other.team:
                    continue

                dist = np.linalg.norm(target.position - other.position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy = target

            if nearest_enemy:
                to_enemy = nearest_enemy.position - other.position
                if min_dist > 0.1:
                    to_enemy = to_enemy / min_dist

                acc = to_enemy * self.max_acceleration * 0.8
                other.velocity += acc * self.dt

                vel_mag = np.linalg.norm(other.velocity)
                if vel_mag > self.max_velocity * 0.9:
                    other.velocity = other.velocity / vel_mag * (self.max_velocity * 0.9)

                other.position += other.velocity * self.dt

        # Keep all in bounds
        for aircraft in self.aircraft:
            aircraft.position = np.clip(aircraft.position, -self.arena_size / 2, self.arena_size / 2)

        # Simple reward: Distance to nearest enemy
        if agent.alive:
            nearest_dist = float('inf')
            for other in self.aircraft:
                if other.id == agent.id or not other.alive:
                    continue
                if self.num_teams > 0 and other.team == agent.team:
                    continue

                dist = np.linalg.norm(other.position - agent.position)
                nearest_dist = min(nearest_dist, dist)

            reward = 1.0 - (nearest_dist / self.fov_range) if nearest_dist < float('inf') else -1.0
        else:
            reward = -10.0

        # Termination
        num_alive = sum(1 for a in self.aircraft if a.alive)
        terminated = num_alive <= 1 or not agent.alive
        truncated = self.step_count >= self.max_steps

        obs = self._get_observation(0)
        info = {"num_alive": num_alive, "agent_alive": agent.alive}

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render environment (TODO: implement visualization)."""
        pass

    def close(self):
        """Close environment."""
        pass
