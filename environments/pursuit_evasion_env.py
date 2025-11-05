"""
Custom Gymnasium Environment for Vision-Based Pursuit-Evasion Control

This environment implements a 2D pursuit-evasion simulation where:
- The agent (chaser) controls acceleration, not position
- Observations are egocentric (first-person) 64x64 grayscale frames
- The target moves with Brownian motion
- Reward is negative squared distance to target
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Optional, Tuple, Dict, Any


class PursuitEvasionEnv(gym.Env):
    """
    A 2D pursuit-evasion environment with dynamic physics and egocentric vision.

    Observation Space:
        Box(0, 255, (64, 64), uint8) - Single frame egocentric view
        Note: Frame stacking wrapper will be applied externally

    Action Space:
        Box(-1, 1, (2,), float32) - Continuous 2D acceleration [a_x, a_y]

    Dynamics:
        v_{t+1} = friction * v_t + a_t * dt
        p_{t+1} = p_t + v_t * dt

    Reward:
        r_t = -reward_scale * ||p_target - p_agent||^2
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        frame_size: int = 64,
        dt: float = 0.1,
        max_velocity: float = 10.0,
        max_acceleration: float = 1.0,
        friction: float = 0.95,
        world_size: float = 100.0,
        view_radius: float = 30.0,
        target_brownian_std: float = 2.0,
        target_size: float = 2.0,
        agent_size: float = 1.5,
        max_steps: int = 500,
        success_threshold: float = 5.0,
        reward_scale: float = 0.01,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Environment parameters
        self.frame_size = frame_size
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.friction = friction
        self.world_size = world_size
        self.view_radius = view_radius
        self.target_brownian_std = target_brownian_std
        self.target_size = target_size
        self.agent_size = agent_size
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.reward_scale = reward_scale
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_size, frame_size), dtype=np.uint8
        )

        # State variables
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        # Rendering
        self.window = None
        self.clock = None
        self.render_scale = 5  # Pixels per world unit

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize agent at origin
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)

        # Initialize target at random position within view
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(5.0, self.view_radius * 0.8)
        self.target_pos = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ], dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)

        self.step_count = 0

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Clip and scale action to acceleration
        action = np.clip(action, -1.0, 1.0)
        acceleration = action * self.max_acceleration

        # Update agent dynamics
        self.agent_vel = self.friction * self.agent_vel + acceleration * self.dt

        # Clip velocity to maximum
        vel_magnitude = np.linalg.norm(self.agent_vel)
        if vel_magnitude > self.max_velocity:
            self.agent_vel = self.agent_vel / vel_magnitude * self.max_velocity

        self.agent_pos += self.agent_vel * self.dt

        # Update target with Brownian motion (in world coordinates)
        brownian_acceleration = self.np_random.normal(
            0, self.target_brownian_std, size=2
        )
        self.target_vel = self.friction * self.target_vel + brownian_acceleration * self.dt

        # Clip target velocity
        target_vel_magnitude = np.linalg.norm(self.target_vel)
        if target_vel_magnitude > self.max_velocity * 0.5:  # Target moves slower
            self.target_vel = self.target_vel / target_vel_magnitude * (self.max_velocity * 0.5)

        self.target_pos += self.target_vel * self.dt

        # Keep target within world bounds (soft constraint)
        target_distance_from_origin = np.linalg.norm(self.target_pos)
        if target_distance_from_origin > self.world_size * 0.8:
            # Apply force back toward origin
            self.target_vel -= 0.1 * self.target_pos / target_distance_from_origin

        # Calculate reward (negative squared distance)
        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        reward = -self.reward_scale * (distance ** 2)

        # Check termination conditions
        self.step_count += 1
        terminated = False  # No specific termination condition
        truncated = self.step_count >= self.max_steps

        # Check if target is out of view (additional failure condition)
        if distance > self.view_radius * 1.5:
            truncated = True
            reward -= 100.0  # Large penalty for losing target

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["distance"] = distance
        info["success"] = distance < self.success_threshold

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Generate egocentric observation.

        The agent is always at the center of the frame, and we render
        the target's relative position in the agent's coordinate frame.

        Returns:
            np.ndarray: (frame_size, frame_size) grayscale image
        """
        # Create blank frame
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

        # Calculate relative position of target in agent's frame
        relative_pos = self.target_pos - self.agent_pos

        # Convert to pixel coordinates (center of frame is origin)
        pixel_scale = self.frame_size / (2 * self.view_radius)
        pixel_x = int(self.frame_size / 2 + relative_pos[0] * pixel_scale)
        pixel_y = int(self.frame_size / 2 - relative_pos[1] * pixel_scale)  # Flip Y

        # Draw target as a white circle if within view
        if 0 <= pixel_x < self.frame_size and 0 <= pixel_y < self.frame_size:
            target_radius_pixels = max(int(self.target_size * pixel_scale), 2)

            # Draw filled circle using numpy (simple rasterization)
            y_coords, x_coords = np.ogrid[:self.frame_size, :self.frame_size]
            mask = (x_coords - pixel_x)**2 + (y_coords - pixel_y)**2 <= target_radius_pixels**2
            frame[mask] = 255

        return frame

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "agent_vel": self.agent_vel.copy(),
            "target_pos": self.target_pos.copy(),
            "target_vel": self.target_vel.copy(),
            "step": self.step_count,
        }

    def render(self):
        """
        Render the environment with HUD-style visualization.

        The rendering emphasizes the dogfight HUD concept:
        - Agent (green) is always at center with crosshair
        - Target (red) moves relative to agent
        - Error vector (cyan) shows the tracking error to be nullified
        - HUD displays error magnitude, velocity, and tracking status
        """
        if self.render_mode is None:
            return None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((800, 800))
            pygame.display.set_caption("Dogfight HUD - Pursuit-Evasion Control")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas with HUD-style dark background
        canvas = pygame.Surface((800, 800))
        canvas.fill((5, 10, 15))  # Dark blue-gray (HUD-like)

        # Define rendering area (centered view around agent)
        center_x, center_y = 400, 400

        # === HUD ELEMENTS ===

        # 1. Draw radar range rings
        for radius_factor in [0.33, 0.66, 1.0]:
            pygame.draw.circle(
                canvas,
                (30, 40, 50),  # Subtle grid lines
                (center_x, center_y),
                int(self.view_radius * self.render_scale * radius_factor),
                1
            )

        # 2. Draw cardinal direction lines (subtle grid)
        grid_color = (20, 30, 40)
        pygame.draw.line(canvas, grid_color,
                        (center_x - 150, center_y),
                        (center_x + 150, center_y), 1)  # Horizontal
        pygame.draw.line(canvas, grid_color,
                        (center_x, center_y - 150),
                        (center_x, center_y + 150), 1)  # Vertical

        # 3. Calculate error vector
        relative_pos = self.target_pos - self.agent_pos
        distance = np.linalg.norm(relative_pos)
        target_screen_x = center_x + int(relative_pos[0] * self.render_scale)
        target_screen_y = center_y - int(relative_pos[1] * self.render_scale)

        # 4. Draw ERROR VECTOR (the key concept!)
        # This is the vector the controller must nullify
        if distance > 0.1:
            error_color = (0, 255, 255)  # Cyan - highly visible
            # Draw thick error vector line from center to target
            pygame.draw.line(
                canvas,
                error_color,
                (center_x, center_y),
                (target_screen_x, target_screen_y),
                3  # Thick line to emphasize
            )
            # Draw arrowhead
            self._draw_arrow_head(canvas, center_x, center_y,
                                target_screen_x, target_screen_y, error_color)

        # 5. Draw target (the object to track)
        pygame.draw.circle(
            canvas,
            (255, 50, 50),  # Bright red
            (target_screen_x, target_screen_y),
            int(self.target_size * self.render_scale) + 2,
            2  # Outline
        )
        pygame.draw.circle(
            canvas,
            (255, 0, 0),  # Red fill
            (target_screen_x, target_screen_y),
            int(self.target_size * self.render_scale)
        )

        # 6. Draw target velocity vector (shows where target is heading)
        if np.linalg.norm(self.target_vel) > 0.1:
            vel_end = (
                target_screen_x + int(self.target_vel[0] * self.render_scale * 2),
                target_screen_y - int(self.target_vel[1] * self.render_scale * 2)
            )
            pygame.draw.line(
                canvas, (255, 100, 100),  # Light red
                (target_screen_x, target_screen_y),
                vel_end, 2
            )

        # 7. Draw agent CROSSHAIR at center (always centered)
        crosshair_size = 20
        crosshair_color = (0, 255, 0)  # Green
        # Horizontal crosshair
        pygame.draw.line(canvas, crosshair_color,
                        (center_x - crosshair_size, center_y),
                        (center_x - 5, center_y), 3)
        pygame.draw.line(canvas, crosshair_color,
                        (center_x + 5, center_y),
                        (center_x + crosshair_size, center_y), 3)
        # Vertical crosshair
        pygame.draw.line(canvas, crosshair_color,
                        (center_x, center_y - crosshair_size),
                        (center_x, center_y - 5), 3)
        pygame.draw.line(canvas, crosshair_color,
                        (center_x, center_y + 5),
                        (center_x, center_y + crosshair_size), 3)
        # Center dot
        pygame.draw.circle(canvas, crosshair_color, (center_x, center_y), 3)

        # 8. Draw agent (small circle at center)
        pygame.draw.circle(
            canvas,
            (0, 180, 0),  # Green
            (center_x, center_y),
            int(self.agent_size * self.render_scale),
            2  # Outline only
        )

        # 9. Draw agent velocity vector (shows our movement)
        if np.linalg.norm(self.agent_vel) > 0.1:
            vel_end = (
                center_x + int(self.agent_vel[0] * self.render_scale * 2),
                center_y - int(self.agent_vel[1] * self.render_scale * 2)
            )
            pygame.draw.line(canvas, (100, 255, 100),
                           (center_x, center_y), vel_end, 2)

        # === HUD TEXT DISPLAYS ===
        font_large = pygame.font.Font(None, 40)
        font_medium = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)

        # Top-left: Error magnitude (primary metric)
        error_mag = distance
        error_color_text = (0, 255, 255) if error_mag > self.success_threshold else (0, 255, 0)
        text = font_large.render(f"ERROR: {error_mag:.2f}", True, error_color_text)
        canvas.blit(text, (15, 15))

        # Status indicator
        status = "LOCKED" if error_mag < self.success_threshold else "TRACKING"
        status_color = (0, 255, 0) if error_mag < self.success_threshold else (255, 200, 0)
        text = font_medium.render(f"[{status}]", True, status_color)
        canvas.blit(text, (15, 55))

        # Top-right: Step counter
        text = font_small.render(f"STEP: {self.step_count}/{self.max_steps}", True, (150, 150, 150))
        canvas.blit(text, (650, 15))

        # Bottom-left: Velocity information
        agent_speed = np.linalg.norm(self.agent_vel)
        text = font_small.render(f"AGENT VEL: {agent_speed:.2f}", True, (100, 255, 100))
        canvas.blit(text, (15, 720))

        target_speed = np.linalg.norm(self.target_vel)
        text = font_small.render(f"TARGET VEL: {target_speed:.2f}", True, (255, 100, 100))
        canvas.blit(text, (15, 750))

        # Bottom-right: Instructions
        text = font_small.render("Goal: Nullify Error Vector", True, (100, 100, 100))
        canvas.blit(text, (520, 750))

        # Center info: Error vector magnitude indicator
        if distance > 1.0:
            error_angle = np.arctan2(-relative_pos[1], relative_pos[0]) * 180 / np.pi
            text = font_small.render(f"{error_angle:.0f}°", True, (0, 200, 200))
            text_rect = text.get_rect(center=(center_x, center_y - 60))
            canvas.blit(text, text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_arrow_head(self, surface, start_x, start_y, end_x, end_y, color):
        """Draw an arrowhead at the end of the error vector."""
        # Calculate arrow direction
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx**2 + dy**2)

        if length < 10:
            return

        # Normalize
        dx /= length
        dy /= length

        # Arrow head size
        arrow_length = 15
        arrow_width = 10

        # Calculate arrow head points
        # Point back from end
        base_x = end_x - dx * arrow_length
        base_y = end_y - dy * arrow_length

        # Perpendicular points
        perp_dx = -dy
        perp_dy = dx

        point1 = (int(base_x + perp_dx * arrow_width), int(base_y + perp_dy * arrow_width))
        point2 = (int(base_x - perp_dx * arrow_width), int(base_y - perp_dy * arrow_width))
        point3 = (int(end_x), int(end_y))

        pygame.draw.polygon(surface, color, [point1, point2, point3])

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
