"""
3D Arena Environment with Search and Pursuit Modes

Phase 3: Large 3D arena where agents must:
1. SEARCH mode: Find each other when not in FOV
2. PURSUIT mode: Engage in dogfight when in FOV

Features:
- Large 3D space (1000x1000x1000)
- Cone-based FOV (limited range and angle)
- Mode switching (search ↔ pursuit)
- Aircraft dynamics (yaw, pitch, roll)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import pygame


class Arena3DEnv(gym.Env):
    """
    Large 3D arena environment with search and pursuit modes.

    State machine:
    - SEARCH mode: Agent searches for target (random search or intelligent)
    - PURSUIT mode: Agent engages target in dogfight
    - Transitions: Based on FOV cone visibility
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        arena_size: float = 1000.0,
        fov_angle: float = 60.0,  # degrees (cone angle)
        fov_range: float = 300.0,  # max detection range
        max_velocity: float = 50.0,
        max_acceleration: float = 5.0,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.arena_size = arena_size
        self.fov_angle = np.deg2rad(fov_angle)
        self.fov_range = fov_range
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.render_mode = render_mode

        # State: [agent_pos (3), agent_vel (3), agent_orient (3),
        #         target_pos (3), target_vel (3), target_orient (3)]
        # Total: 18 dimensions
        self.observation_space = spaces.Box(
            low=-arena_size,
            high=arena_size,
            shape=(18,),
            dtype=np.float32
        )

        # Action: [ax, ay, az] acceleration commands
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Agent and target states
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.agent_orient = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Forward direction

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self.target_orient = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Mode tracking
        self.mode = "SEARCH"  # or "PURSUIT"
        self.step_count = 0
        self.max_steps = 5000

        # Pygame for rendering
        self.screen = None
        self.clock = None

    def _check_fov_visibility(self) -> bool:
        """
        Check if target is in agent's FOV cone.

        Returns:
            True if target is visible, False otherwise
        """
        # Vector from agent to target
        to_target = self.target_pos - self.agent_pos
        distance = np.linalg.norm(to_target)

        # Check range
        if distance > self.fov_range or distance < 0.1:
            return False

        # Check angle (cone)
        to_target_norm = to_target / (distance + 1e-8)
        cos_angle = np.dot(self.agent_orient, to_target_norm)

        # FOV cone: check if angle is within fov_angle
        cos_fov = np.cos(self.fov_angle / 2)

        return cos_angle >= cos_fov

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([
            self.agent_pos,
            self.agent_vel,
            self.agent_orient,
            self.target_pos,
            self.target_vel,
            self.target_orient,
        ])

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        in_fov = self._check_fov_visibility()

        return {
            "mode": self.mode,
            "distance": distance,
            "in_fov": in_fov,
            "agent_pos": self.agent_pos.copy(),
            "target_pos": self.target_pos.copy(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Random positions in arena
        self.agent_pos = self.np_random.uniform(
            -self.arena_size / 4, self.arena_size / 4, size=3
        ).astype(np.float32)

        self.target_pos = self.np_random.uniform(
            -self.arena_size / 4, self.arena_size / 4, size=3
        ).astype(np.float32)

        # Random velocities
        self.agent_vel = self.np_random.uniform(-10, 10, size=3).astype(np.float32)
        self.target_vel = self.np_random.uniform(-10, 10, size=3).astype(np.float32)

        # Random orientations
        self.agent_orient = self.np_random.uniform(-1, 1, size=3).astype(np.float32)
        self.agent_orient = self.agent_orient / (np.linalg.norm(self.agent_orient) + 1e-8)

        self.target_orient = self.np_random.uniform(-1, 1, size=3).astype(np.float32)
        self.target_orient = self.target_orient / (np.linalg.norm(self.target_orient) + 1e-8)

        # Reset mode
        self.mode = "SEARCH"
        self.step_count = 0

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        self.step_count += 1

        # Agent action
        action = np.clip(action, -1.0, 1.0)
        agent_acc = action * self.max_acceleration

        # Update agent velocity and position
        self.agent_vel += agent_acc * self.dt
        vel_mag = np.linalg.norm(self.agent_vel)
        if vel_mag > self.max_velocity:
            self.agent_vel = self.agent_vel / vel_mag * self.max_velocity

        self.agent_pos += self.agent_vel * self.dt

        # Update agent orientation (from velocity)
        if vel_mag > 1.0:
            self.agent_orient = self.agent_vel / vel_mag

        # Target behavior (simple evasion for now)
        to_agent = self.agent_pos - self.target_pos
        distance = np.linalg.norm(to_agent)

        if distance > 0.1:
            # Evade
            evade_dir = -to_agent / distance
            target_acc = evade_dir * self.max_acceleration * 0.5
        else:
            target_acc = np.zeros(3)

        # Add random component
        target_acc += self.np_random.normal(0, 1.0, size=3)

        self.target_vel += target_acc * self.dt
        target_vel_mag = np.linalg.norm(self.target_vel)
        if target_vel_mag > self.max_velocity * 0.8:
            self.target_vel = self.target_vel / target_vel_mag * (self.max_velocity * 0.8)

        self.target_pos += self.target_vel * self.dt

        # Update target orientation
        if target_vel_mag > 1.0:
            self.target_orient = self.target_vel / target_vel_mag

        # Keep in bounds
        self.agent_pos = np.clip(self.agent_pos, -self.arena_size / 2, self.arena_size / 2)
        self.target_pos = np.clip(self.target_pos, -self.arena_size / 2, self.arena_size / 2)

        # Check FOV and update mode
        in_fov = self._check_fov_visibility()

        if in_fov:
            self.mode = "PURSUIT"
        else:
            self.mode = "SEARCH"

        # Compute reward
        if self.mode == "PURSUIT":
            # Reward for keeping target in FOV and close
            reward = 1.0 - (distance / self.fov_range)
        else:
            # Penalty for not finding target
            reward = -0.1

        # Termination
        terminated = distance < 5.0  # Success: very close
        truncated = self.step_count >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render environment."""
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((800, 800))
                pygame.display.set_caption("3D Arena - Search & Pursuit")
            else:
                self.screen = pygame.Surface((800, 800))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((20, 20, 40))

        # Draw arena bounds (2D projection)
        center_x, center_y = 400, 400
        scale = 400 / (self.arena_size / 2)

        # Project 3D positions to 2D (top-down view)
        agent_x = center_x + self.agent_pos[0] * scale
        agent_y = center_y - self.agent_pos[1] * scale

        target_x = center_x + self.target_pos[0] * scale
        target_y = center_y - self.target_pos[1] * scale

        # Draw FOV cone (simplified as circle)
        fov_radius = int(self.fov_range * scale)
        pygame.draw.circle(
            self.screen,
            (40, 40, 60),
            (int(agent_x), int(agent_y)),
            fov_radius,
            1
        )

        # Draw agent (green)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (int(agent_x), int(agent_y)),
            10
        )

        # Draw target (red if in FOV, yellow otherwise)
        color = (255, 0, 0) if self.mode == "PURSUIT" else (255, 255, 0)
        pygame.draw.circle(
            self.screen,
            color,
            (int(target_x), int(target_y)),
            8
        )

        # Draw connection line if in FOV
        if self.mode == "PURSUIT":
            pygame.draw.line(
                self.screen,
                (0, 255, 255),
                (int(agent_x), int(agent_y)),
                (int(target_x), int(target_y)),
                2
            )

        # Draw mode text
        font = pygame.font.Font(None, 36)
        mode_text = font.render(f"Mode: {self.mode}", True, (255, 255, 255))
        self.screen.blit(mode_text, (10, 10))

        distance_text = font.render(f"Distance: {np.linalg.norm(self.target_pos - self.agent_pos):.1f}", True, (255, 255, 255))
        self.screen.blit(distance_text, (10, 50))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        """Close environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
