"""
Custom Gymnasium Environment for 2.5D Vision-Based Pursuit-Evasion Control

This environment implements a 2.5D (pseudo-3D) pursuit-evasion simulation where:
- The agent controls 3D acceleration (lateral, vertical, thrust)
- Observations are egocentric 64x64 grayscale frames with depth encoding
- Depth (z-axis) is visually encoded as target size (closer = bigger)
- Multi-target support with color-coded differentiation
- Reward based on 3D error vector magnitude
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Optional, Tuple, Dict, Any, List


class Target:
    """Represents a single target in the environment."""

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        color: Tuple[int, int, int],
        name: str,
        size: float = 2.0,
    ):
        self.position = position.astype(np.float32)  # (x, y, z)
        self.velocity = velocity.astype(np.float32)  # (vx, vy, vz)
        self.color = color
        self.name = name
        self.base_size = size


class PursuitEvasion3DEnv(gym.Env):
    """
    A 2.5D pursuit-evasion environment with depth perception and multi-target support.

    Observation Space:
        Box(0, 255, (64, 64), uint8) - Single frame egocentric view
        - Target position encoded in (x, y) pixel coordinates
        - Depth (z) encoded as target size (scale)
        - Multiple targets differentiated by intensity/rendering order

    Action Space:
        Box(-1, 1, (3,), float32) - Continuous 3D acceleration [a_x, a_y, a_z]
        - a_x, a_y: Lateral/vertical acceleration (steering)
        - a_z: Forward/backward acceleration (thrust/throttle)

    Dynamics:
        v_{t+1} = friction * v_t + a_t * dt
        p_{t+1} = p_t + v_t * dt

    Reward:
        r_t = -reward_scale * ||p_target - p_agent||^2  (3D distance)
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
        depth_range: Tuple[float, float] = (10.0, 50.0),  # (min_z, max_z)
        target_brownian_std: float = 2.0,
        target_size: float = 2.0,
        agent_size: float = 1.5,
        max_steps: int = 500,
        success_threshold: float = 5.0,  # 3D distance
        reward_scale: float = 0.01,
        num_targets: int = 1,
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
        self.min_depth, self.max_depth = depth_range
        self.target_brownian_std = target_brownian_std
        self.target_size = target_size
        self.agent_size = agent_size
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.reward_scale = reward_scale
        self.num_targets = num_targets
        self.render_mode = render_mode

        # Define action and observation spaces (3D actions now!)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_size, frame_size), dtype=np.uint8
        )

        # State variables (now 3D!)
        self.agent_pos = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self.agent_vel = np.zeros(3, dtype=np.float32)  # [vx, vy, vz]

        # Multi-target support
        self.targets: List[Target] = []
        self.target_colors = [
            (255, 0, 0),      # Red (primary target)
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
        ]

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
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)

        # Initialize targets
        self.targets = []
        for i in range(self.num_targets):
            # Random position in 3D space
            angle = self.np_random.uniform(0, 2 * np.pi)
            elevation = self.np_random.uniform(-0.3, 0.3)  # Slight vertical spread
            distance_xy = self.np_random.uniform(10.0, self.view_radius * 0.8)
            distance_z = self.np_random.uniform(self.min_depth, self.max_depth * 0.7)

            target_pos = np.array([
                distance_xy * np.cos(angle),
                distance_xy * np.sin(angle) + elevation * distance_xy,
                distance_z
            ], dtype=np.float32)

            target_vel = np.zeros(3, dtype=np.float32)

            color = self.target_colors[i % len(self.target_colors)]
            name = f"Target-{chr(65 + i)}"  # A, B, C, ...

            self.targets.append(Target(target_pos, target_vel, color, name, self.target_size))

        self.step_count = 0

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Clip and scale action to 3D acceleration
        action = np.clip(action, -1.0, 1.0)
        acceleration = action * self.max_acceleration

        # Update agent dynamics (now 3D!)
        self.agent_vel = self.friction * self.agent_vel + acceleration * self.dt

        # Clip velocity to maximum
        vel_magnitude = np.linalg.norm(self.agent_vel)
        if vel_magnitude > self.max_velocity:
            self.agent_vel = self.agent_vel / vel_magnitude * self.max_velocity

        self.agent_pos += self.agent_vel * self.dt

        # Update all targets with Brownian motion (in 3D)
        for target in self.targets:
            brownian_acceleration = self.np_random.normal(
                0, self.target_brownian_std, size=3
            )
            target.velocity = self.friction * target.velocity + brownian_acceleration * self.dt

            # Clip target velocity
            target_vel_magnitude = np.linalg.norm(target.velocity)
            if target_vel_magnitude > self.max_velocity * 0.5:  # Targets move slower
                target.velocity = target.velocity / target_vel_magnitude * (self.max_velocity * 0.5)

            target.position += target.velocity * self.dt

            # Keep target within reasonable z-bounds
            if target.position[2] < self.min_depth:
                target.position[2] = self.min_depth
                target.velocity[2] = abs(target.velocity[2])  # Bounce back
            elif target.position[2] > self.max_depth:
                target.position[2] = self.max_depth
                target.velocity[2] = -abs(target.velocity[2])  # Bounce back

        # Calculate reward based on primary target (first one)
        primary_target = self.targets[0]
        distance_3d = np.linalg.norm(primary_target.position - self.agent_pos)
        reward = -self.reward_scale * (distance_3d ** 2)

        # Check termination conditions
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Check if target is too far (consider 3D distance)
        if distance_3d > self.view_radius * 2.0:
            truncated = True
            reward -= 100.0  # Large penalty for losing target

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["distance_3d"] = distance_3d
        info["distance_xy"] = np.linalg.norm(primary_target.position[:2] - self.agent_pos[:2])
        info["distance_z"] = abs(primary_target.position[2] - self.agent_pos[2])
        info["success"] = distance_3d < self.success_threshold

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Generate egocentric 2.5D observation.

        The agent is always at the center of the frame, and we render
        targets' relative positions with depth-based scaling.

        Returns:
            np.ndarray: (frame_size, frame_size) grayscale image
        """
        # Create blank frame
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

        # Render all targets
        for target in self.targets:
            # Calculate relative position in 3D
            relative_pos_3d = target.position - self.agent_pos

            # Project to 2D view plane (x, y)
            relative_pos_2d = relative_pos_3d[:2]
            depth = relative_pos_3d[2]

            # Skip if depth is invalid
            if depth <= 0:
                continue

            # Calculate size based on depth (inverse relationship)
            # Closer targets appear larger
            depth_factor = self.min_depth / max(depth, self.min_depth)
            apparent_size = target.base_size * depth_factor

            # Convert to pixel coordinates (center of frame is origin)
            pixel_scale = self.frame_size / (2 * self.view_radius)
            pixel_x = int(self.frame_size / 2 + relative_pos_2d[0] * pixel_scale)
            pixel_y = int(self.frame_size / 2 - relative_pos_2d[1] * pixel_scale)  # Flip Y

            # Draw target as a white circle if within view
            if 0 <= pixel_x < self.frame_size and 0 <= pixel_y < self.frame_size:
                target_radius_pixels = max(int(apparent_size * pixel_scale), 2)

                # Draw filled circle using numpy (simple rasterization)
                y_coords, x_coords = np.ogrid[:self.frame_size, :self.frame_size]
                mask = (x_coords - pixel_x)**2 + (y_coords - pixel_y)**2 <= target_radius_pixels**2

                # Use different intensities for multiple targets (brightest for closest/primary)
                intensity = 255  # All targets white in grayscale
                frame[mask] = np.maximum(frame[mask], intensity)

        return frame

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        info = {
            "agent_pos": self.agent_pos.copy(),
            "agent_vel": self.agent_vel.copy(),
            "step": self.step_count,
            "num_targets": len(self.targets),
        }

        # Add info for each target
        for i, target in enumerate(self.targets):
            info[f"target_{i}_pos"] = target.position.copy()
            info[f"target_{i}_vel"] = target.velocity.copy()
            info[f"target_{i}_name"] = target.name

        return info

    def render(self):
        """
        Render the environment with 2.5D HUD-style visualization.

        Shows depth information and multi-target tracking.
        """
        if self.render_mode is None:
            return None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((800, 800))
            pygame.display.set_caption("2.5D Dogfight HUD - 3D Error Vector Nullification")

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

        # 2. Draw cardinal direction lines
        grid_color = (20, 30, 40)
        pygame.draw.line(canvas, grid_color,
                        (center_x - 150, center_y),
                        (center_x + 150, center_y), 1)
        pygame.draw.line(canvas, grid_color,
                        (center_x, center_y - 150),
                        (center_x, center_y + 150), 1)

        # 3. Draw all targets with depth-based rendering
        for idx, target in enumerate(self.targets):
            relative_pos_3d = target.position - self.agent_pos
            relative_pos_2d = relative_pos_3d[:2]
            depth = relative_pos_3d[2]

            # Calculate screen position
            target_screen_x = center_x + int(relative_pos_2d[0] * self.render_scale)
            target_screen_y = center_y - int(relative_pos_2d[1] * self.render_scale)

            # Calculate size based on depth
            depth_factor = self.min_depth / max(depth, self.min_depth)
            apparent_size = target.base_size * depth_factor
            visual_radius = int(apparent_size * self.render_scale)

            # Draw error vector to primary target (first one)
            if idx == 0:
                distance_3d = np.linalg.norm(relative_pos_3d)
                if distance_3d > 0.1:
                    error_color = (0, 255, 255)  # Cyan
                    pygame.draw.line(
                        canvas,
                        error_color,
                        (center_x, center_y),
                        (target_screen_x, target_screen_y),
                        3
                    )
                    self._draw_arrow_head(canvas, center_x, center_y,
                                        target_screen_x, target_screen_y, error_color)

            # Draw target with depth-based size
            pygame.draw.circle(
                canvas,
                target.color,  # Color-coded
                (target_screen_x, target_screen_y),
                visual_radius + 2,
                2  # Outline
            )
            pygame.draw.circle(
                canvas,
                target.color,
                (target_screen_x, target_screen_y),
                visual_radius
            )

            # Draw range indicator
            font_small = pygame.font.Font(None, 20)
            range_text = font_small.render(f"{depth:.1f}m", True, target.color)
            canvas.blit(range_text, (target_screen_x + visual_radius + 5, target_screen_y - 10))

            # Draw target name
            name_text = font_small.render(target.name, True, target.color)
            canvas.blit(name_text, (target_screen_x + visual_radius + 5, target_screen_y + 5))

            # Draw velocity vector
            if np.linalg.norm(target.velocity[:2]) > 0.1:
                vel_end = (
                    target_screen_x + int(target.velocity[0] * self.render_scale * 2),
                    target_screen_y - int(target.velocity[1] * self.render_scale * 2)
                )
                lighter_color = tuple(min(255, c + 100) for c in target.color)
                pygame.draw.line(canvas, lighter_color,
                               (target_screen_x, target_screen_y), vel_end, 2)

        # 4. Draw agent CROSSHAIR at center
        crosshair_size = 20
        crosshair_color = (0, 255, 0)  # Green
        pygame.draw.line(canvas, crosshair_color,
                        (center_x - crosshair_size, center_y),
                        (center_x - 5, center_y), 3)
        pygame.draw.line(canvas, crosshair_color,
                        (center_x + 5, center_y),
                        (center_x + crosshair_size, center_y), 3)
        pygame.draw.line(canvas, crosshair_color,
                        (center_x, center_y - crosshair_size),
                        (center_x, center_y - 5), 3)
        pygame.draw.line(canvas, crosshair_color,
                        (center_x, center_y + 5),
                        (center_x, center_y + crosshair_size), 3)
        pygame.draw.circle(canvas, crosshair_color, (center_x, center_y), 3)

        # 5. Draw agent
        pygame.draw.circle(
            canvas,
            (0, 180, 0),
            (center_x, center_y),
            int(self.agent_size * self.render_scale),
            2
        )

        # === HUD TEXT DISPLAYS ===
        font_large = pygame.font.Font(None, 40)
        font_medium = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)

        # Primary target info
        primary_target = self.targets[0]
        relative_pos_3d = primary_target.position - self.agent_pos
        distance_3d = np.linalg.norm(relative_pos_3d)
        distance_xy = np.linalg.norm(relative_pos_3d[:2])
        distance_z = abs(relative_pos_3d[2])

        # Top-left: 3D Error
        error_color_text = (0, 255, 255) if distance_3d > self.success_threshold else (0, 255, 0)
        text = font_large.render(f"ERROR 3D: {distance_3d:.2f}", True, error_color_text)
        canvas.blit(text, (15, 15))

        # Error breakdown
        text = font_small.render(f"XY: {distance_xy:.2f}  Z: {distance_z:.2f}", True, (150, 200, 200))
        canvas.blit(text, (15, 55))

        # Status
        status = "LOCKED" if distance_3d < self.success_threshold else "TRACKING"
        status_color = (0, 255, 0) if distance_3d < self.success_threshold else (255, 200, 0)
        text = font_medium.render(f"[{status}]", True, status_color)
        canvas.blit(text, (15, 85))

        # Top-right: Step counter
        text = font_small.render(f"STEP: {self.step_count}/{self.max_steps}", True, (150, 150, 150))
        canvas.blit(text, (620, 15))

        # Multi-target count
        text = font_small.render(f"TARGETS: {len(self.targets)}", True, (150, 150, 150))
        canvas.blit(text, (620, 45))

        # Bottom-left: Velocity (3D)
        agent_speed = np.linalg.norm(self.agent_vel)
        text = font_small.render(f"AGENT VEL: {agent_speed:.2f}", True, (100, 255, 100))
        canvas.blit(text, (15, 690))

        text = font_small.render(f"  XY: {np.linalg.norm(self.agent_vel[:2]):.2f}  Z: {abs(self.agent_vel[2]):.2f}", True, (80, 200, 80))
        canvas.blit(text, (15, 715))

        target_speed = np.linalg.norm(primary_target.velocity)
        text = font_small.render(f"TARGET VEL: {target_speed:.2f}", True, (255, 100, 100))
        canvas.blit(text, (15, 745))

        # Bottom-right: Instructions
        text = font_small.render("Goal: Nullify 3D Error Vector", True, (100, 100, 100))
        canvas.blit(text, (480, 775))

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
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx**2 + dy**2)

        if length < 10:
            return

        dx /= length
        dy /= length

        arrow_length = 15
        arrow_width = 10

        base_x = end_x - dx * arrow_length
        base_y = end_y - dy * arrow_length

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
