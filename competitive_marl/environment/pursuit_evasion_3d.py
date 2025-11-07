"""
3D Competitive Pursuit-Evasion Environment

Clean implementation for competitive MARL training.
Both pursuer and evader are RL-controlled agents.

Features:
- 3D egocentric observation (agent always at origin)
- Relative motion dynamics
- Focus-based rewards (opposing for each agent)
- Optimized for Kaggle training (optional rendering)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.hud_renderer import HUDRenderer3D


class Target3D:
    """Target (Evader) in 3D space."""

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        base_size: float = 4.0,
    ):
        self.position = position.astype(np.float32)
        self.velocity = velocity.astype(np.float32)
        self.base_size = base_size


class CompetitivePursuitEvasion3D(gym.Env):
    """
    3D Pursuit-Evasion environment for competitive MARL.

    Observation Space:
        - Box(64, 64): Grayscale egocentric view with depth-based target sizing

    Action Space:
        - Box(-1, 1, (3,)): 3D acceleration vector [ax, ay, az]

    Rewards:
        - Pursuer: Positive for keeping target in focus, bonus for sustained focus
        - Evader: Opposite rewards (wants to escape focus)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        view_size: float = 30.0,
        frame_size: int = 64,
        depth_range: Tuple[float, float] = (10.0, 50.0),
        success_threshold: float = 9.0,
        target_size: float = 4.0,
        max_velocity: float = 50.0,
        max_acceleration: float = 5.0,
        max_angular_velocity: float = 2.0,
        dt: float = 0.1,
        max_steps: int = 1000,
        focus_reward: float = 0.1,
        focus_bonus: float = 10.0,
        escape_penalty: float = -2.0,
        outside_penalty_scale: float = 0.01,
        focus_time_threshold: int = 50,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Environment parameters
        self.view_size = view_size
        self.frame_size = frame_size
        self.min_depth, self.max_depth = depth_range
        self.success_threshold = success_threshold
        self.target_base_size = target_size
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Reward parameters
        self.focus_reward = focus_reward
        self.focus_bonus = focus_bonus
        self.escape_penalty = escape_penalty
        self.outside_penalty_scale = outside_penalty_scale
        self.focus_time_threshold = focus_time_threshold

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_size, frame_size),
            dtype=np.uint8
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Agent state (always at origin in egocentric frame)
        self.agent_velocity = np.zeros(3, dtype=np.float32)

        # Target (evader)
        self.target = None

        # Episode tracking
        self.step_count = 0
        self.steps_in_focus = 0
        self.was_near_completion = False
        self.cumulative_reward = 0.0

        # Renderer (only for local testing)
        self.renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset agent (always at origin in egocentric frame)
        self.agent_velocity = self.np_random.uniform(-5, 5, size=3).astype(np.float32)

        # Reset target at random position in view
        target_distance = self.np_random.uniform(self.view_size * 0.3, self.view_size * 0.8)
        target_angle_xy = self.np_random.uniform(-np.pi, np.pi)
        target_angle_z = self.np_random.uniform(-0.3, 0.3)  # Slightly up/down

        target_x = target_distance * np.cos(target_angle_xy) * np.cos(target_angle_z)
        target_y = target_distance * np.sin(target_angle_xy) * np.cos(target_angle_z)
        target_z = self.np_random.uniform(self.min_depth * 1.5, self.max_depth * 0.8)

        target_position = np.array([target_x, target_y, target_z], dtype=np.float32)
        target_velocity = self.np_random.uniform(-10, 10, size=3).astype(np.float32)

        self.target = Target3D(target_position, target_velocity, self.target_base_size)

        # Reset episode tracking
        self.step_count = 0
        self.steps_in_focus = 0
        self.was_near_completion = False
        self.cumulative_reward = 0.0

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Args:
            action: 3D acceleration vector [-1, 1]³

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        acceleration = action * self.max_acceleration

        # Update agent velocity
        self.agent_velocity += acceleration * self.dt
        vel_mag = np.linalg.norm(self.agent_velocity)
        if vel_mag > self.max_velocity:
            self.agent_velocity = self.agent_velocity / vel_mag * self.max_velocity

        # Target motion is handled by external controller (evader agent)
        # For now, target velocity updates via set_target_action()

        # RELATIVE MOTION: Since agent is at origin in egocentric frame,
        # target moves relative to agent
        relative_movement = (self.target.velocity - self.agent_velocity) * self.dt
        self.target.position += relative_movement

        # Keep target within depth bounds
        if self.target.position[2] < self.min_depth:
            self.target.position[2] = self.min_depth
            self.target.velocity[2] = abs(self.target.velocity[2])
        elif self.target.position[2] > self.max_depth:
            self.target.position[2] = self.max_depth
            self.target.velocity[2] = -abs(self.target.velocity[2])

        # Calculate distance and focus status
        distance_3d = np.linalg.norm(self.target.position)
        in_focus = distance_3d < self.success_threshold

        # COMPETITIVE REWARDS
        # Pursuer reward
        if in_focus:
            self.steps_in_focus += 1
            agent_reward = self.focus_reward

            # Check for focus bonus
            if self.steps_in_focus >= self.focus_time_threshold:
                agent_reward += self.focus_bonus
                self.steps_in_focus = 0
                self.was_near_completion = False

            # Track near completion
            if self.steps_in_focus >= self.focus_time_threshold * 0.8:
                self.was_near_completion = True
        else:
            # Outside focus
            if self.was_near_completion:
                agent_reward = self.escape_penalty
                self.was_near_completion = False
            else:
                agent_reward = -self.outside_penalty_scale * distance_3d

            self.steps_in_focus = 0

        # Evader reward (opposite)
        if in_focus:
            target_reward = -0.1
            if self.steps_in_focus >= self.focus_time_threshold:
                target_reward -= 10.0
        else:
            if self.was_near_completion:
                target_reward = 2.0
            else:
                target_reward = 0.05 * distance_3d

        # Update tracking
        self.step_count += 1
        self.cumulative_reward += agent_reward

        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Check if target escaped view
        max_view_distance = self.view_size * 1.5
        if distance_3d > max_view_distance:
            truncated = True
            agent_reward -= 100.0  # Large penalty for losing target
            target_reward += 100.0  # Large reward for evader

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["distance_3d"] = distance_3d
        info["distance_xy"] = np.linalg.norm(self.target.position[:2])
        info["in_focus"] = in_focus
        info["steps_in_focus"] = self.steps_in_focus
        info["focus_progress"] = self.steps_in_focus / self.focus_time_threshold
        info["agent_reward"] = agent_reward
        info["target_reward"] = target_reward

        return obs, agent_reward, terminated, truncated, info

    def set_target_action(self, target_action: np.ndarray):
        """
        Set target (evader) action.

        Called externally by evader agent during competitive training.

        Args:
            target_action: 3D acceleration vector [-1, 1]³
        """
        target_action = np.clip(target_action, -1.0, 1.0)
        acceleration = target_action * self.max_acceleration

        # Update target velocity
        self.target.velocity += acceleration * self.dt
        vel_mag = np.linalg.norm(self.target.velocity)
        if vel_mag > self.max_velocity:
            self.target.velocity = self.target.velocity / vel_mag * self.max_velocity

    def _get_observation(self) -> np.ndarray:
        """
        Generate egocentric 2.5D observation.

        Agent is always at center (origin), target rendered with depth-based sizing.
        """
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

        # Target position in egocentric frame
        relative_pos_3d = self.target.position
        relative_pos_2d = relative_pos_3d[:2]
        depth = relative_pos_3d[2]

        if depth <= 0:
            return frame

        # Depth-based sizing
        depth_factor = self.min_depth / max(depth, self.min_depth)
        apparent_size = self.target.base_size * depth_factor

        # Convert to pixel coordinates
        pixel_scale = self.frame_size / (2 * self.view_size)
        pixel_x = int(self.frame_size / 2 + relative_pos_2d[0] * pixel_scale)
        pixel_y = int(self.frame_size / 2 - relative_pos_2d[1] * pixel_scale)

        # Draw target
        if 0 <= pixel_x < self.frame_size and 0 <= pixel_y < self.frame_size:
            target_radius_pixels = max(int(apparent_size * pixel_scale), 2)

            for dy in range(-target_radius_pixels, target_radius_pixels + 1):
                for dx in range(-target_radius_pixels, target_radius_pixels + 1):
                    if dx * dx + dy * dy <= target_radius_pixels * target_radius_pixels:
                        px = pixel_x + dx
                        py = pixel_y + dy
                        if 0 <= px < self.frame_size and 0 <= py < self.frame_size:
                            frame[py, px] = 255

        return frame

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "step": self.step_count,
            "agent_velocity": self.agent_velocity.copy(),
            "target_position": self.target.position.copy(),
            "target_velocity": self.target.velocity.copy(),
            "cumulative_reward": self.cumulative_reward,
        }

    def render(self):
        """Render environment (only for local testing)."""
        if self.render_mode is None:
            return

        # Initialize renderer if needed
        if self.renderer is None:
            self.renderer = HUDRenderer3D(
                view_size=self.view_size,
                success_threshold=self.success_threshold,
                target_size=self.target_base_size,
                screen_size=(800, 800),
                fps=self.metadata["render_fps"],
            )

        # Calculate current state
        distance = np.linalg.norm(self.target.position)
        in_focus = distance < self.success_threshold

        # Render
        should_continue = self.renderer.render(
            target_position=self.target.position,
            target_velocity=self.target.velocity,
            agent_velocity=self.agent_velocity,
            in_focus=in_focus,
            distance=distance,
            steps_in_focus=self.steps_in_focus,
            focus_threshold=self.focus_time_threshold,
            episode_step=self.step_count,
            episode_reward=self.cumulative_reward,
        )

        return should_continue

    def close(self):
        """Close environment."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def get_target_observation(env: CompetitivePursuitEvasion3D) -> np.ndarray:
    """
    Get observation from evader's perspective.

    This inverts the perspective: evader at origin, pursuer (agent) is the target.

    Args:
        env: The environment

    Returns:
        Observation for evader agent
    """
    frame = np.zeros((env.frame_size, env.frame_size), dtype=np.uint8)

    # Invert: agent (pursuer) is now the "target" from evader's perspective
    # Agent is at origin in env frame, so relative position is -target.position
    relative_pos_3d = -env.target.position
    relative_pos_2d = relative_pos_3d[:2]
    depth = relative_pos_3d[2]

    if depth <= 0:
        return frame

    # Depth-based sizing
    depth_factor = env.min_depth / max(depth, env.min_depth)
    apparent_size = 1.5 * depth_factor  # Agent size

    # Convert to pixel coordinates
    pixel_scale = env.frame_size / (2 * env.view_size)
    pixel_x = int(env.frame_size / 2 + relative_pos_2d[0] * pixel_scale)
    pixel_y = int(env.frame_size / 2 - relative_pos_2d[1] * pixel_scale)

    # Draw agent (pursuer)
    if 0 <= pixel_x < env.frame_size and 0 <= pixel_y < env.frame_size:
        agent_radius_pixels = max(int(apparent_size * pixel_scale), 2)

        for dy in range(-agent_radius_pixels, agent_radius_pixels + 1):
            for dx in range(-agent_radius_pixels, agent_radius_pixels + 1):
                if dx * dx + dy * dy <= agent_radius_pixels * agent_radius_pixels:
                    px = pixel_x + dx
                    py = pixel_y + dy
                    if 0 <= px < env.frame_size and 0 <= py < env.frame_size:
                        frame[py, px] = 255

    return frame
