"""
Reusable HUD Renderer for 3D Egocentric View

Extracted from demo_3d to provide consistent HUD rendering across all environments.
Provides the same square radar view with velocity vectors and focus tracking.
"""

import numpy as np
import pygame
from typing import List, Tuple, Dict, Any, Optional


class HUDRenderer3D:
    """
    Renders egocentric HUD view matching demo_3d style.

    Features:
    - Square radar boundaries (not circular)
    - Cardinal direction lines
    - Depth-based target sizing
    - Agent/target velocity vectors
    - Crosshair at center
    - Focus status and error display
    """

    def __init__(
        self,
        frame_size: int = 64,
        view_size: float = 30.0,
        depth_range: Tuple[float, float] = (10.0, 50.0),
        success_threshold: float = 9.0,
        target_size: float = 4.0,
        agent_size: float = 1.5,
        render_size: int = 400,
    ):
        """
        Initialize HUD renderer.

        Args:
            frame_size: Base frame resolution (for observation)
            view_size: Field of view size in world units
            depth_range: (min_depth, max_depth) for depth perception
            success_threshold: Distance for focus success (30% of view)
            target_size: Size of target in world units
            agent_size: Size of agent in world units
            render_size: Display size in pixels
        """
        self.frame_size = frame_size
        self.view_size = view_size
        self.depth_range = depth_range
        self.success_threshold = success_threshold
        self.target_size = target_size
        self.agent_size = agent_size
        self.render_size = render_size

        # Create surface for rendering
        self.surface = pygame.Surface((render_size, render_size))

        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (60, 60, 100)
        self.COLOR_AGENT = (0, 255, 0)
        self.COLOR_TARGET = (255, 100, 100)
        self.COLOR_VELOCITY = (100, 200, 255)
        self.COLOR_CROSSHAIR = (255, 255, 0)
        self.COLOR_TEXT = (200, 200, 200)
        self.COLOR_FOCUS_SUCCESS = (0, 255, 0)
        self.COLOR_FOCUS_FAIL = (255, 100, 100)

    def render(
        self,
        agent_velocity: np.ndarray,
        targets: List[Dict[str, Any]],
        info: Dict[str, Any],
        agent_id: Optional[int] = None,
    ) -> pygame.Surface:
        """
        Render HUD view.

        Args:
            agent_velocity: Agent velocity vector [vx, vy, vz] in world frame
            targets: List of target dicts with keys:
                - 'position': relative position [x, y, z] in agent frame
                - 'velocity': target velocity [vx, vy, vz] in world frame
                - 'team': team ID (optional, for coloring)
            info: Environment info dict with focus status, etc.
            agent_id: Agent ID for display (optional)

        Returns:
            pygame.Surface with rendered HUD
        """
        # Clear surface
        self.surface.fill(self.COLOR_BG)

        center_x = self.render_size // 2
        center_y = self.render_size // 2
        scale = self.render_size / self.view_size

        # 1. Draw SQUARE radar boundaries
        self._draw_radar_grid(center_x, center_y, scale)

        # 2. Draw targets
        for target_info in targets:
            self._draw_target(target_info, center_x, center_y, scale)

        # 3. Draw agent velocity vector (green arrow from center)
        self._draw_velocity_vector(agent_velocity, center_x, center_y, scale, self.COLOR_AGENT)

        # 4. Draw crosshair at center
        self._draw_crosshair(center_x, center_y)

        # 5. Draw HUD information
        self._draw_hud_info(agent_velocity, targets, info, agent_id)

        return self.surface

    def _draw_radar_grid(self, center_x: int, center_y: int, scale: float):
        """Draw square radar boundaries and grid lines."""
        # Outer boundary (view_size)
        half_size = int(self.view_size * scale / 2)
        pygame.draw.rect(
            self.surface,
            self.COLOR_GRID,
            (center_x - half_size, center_y - half_size, half_size * 2, half_size * 2),
            2
        )

        # Inner boundary (success threshold circle)
        threshold_radius = int(self.success_threshold * scale)
        pygame.draw.circle(
            self.surface,
            self.COLOR_GRID,
            (center_x, center_y),
            threshold_radius,
            1
        )

        # Cardinal direction lines (cross)
        pygame.draw.line(
            self.surface,
            self.COLOR_GRID,
            (center_x - half_size, center_y),
            (center_x + half_size, center_y),
            1
        )
        pygame.draw.line(
            self.surface,
            self.COLOR_GRID,
            (center_x, center_y - half_size),
            (center_x, center_y + half_size),
            1
        )

    def _draw_target(
        self,
        target_info: Dict[str, Any],
        center_x: int,
        center_y: int,
        scale: float
    ):
        """Draw target with depth-based sizing."""
        position = target_info['position']  # [x, y, z] in agent frame
        velocity = target_info.get('velocity', np.zeros(3))
        team = target_info.get('team', -1)

        # Project to 2D (x, y plane)
        x_proj = position[0]
        y_proj = position[1]
        depth = position[2]

        # Screen coordinates
        screen_x = int(center_x + x_proj * scale)
        screen_y = int(center_y - y_proj * scale)  # Flip Y for screen coords

        # Depth-based sizing (closer = larger)
        depth_clamped = np.clip(depth, self.depth_range[0], self.depth_range[1])
        depth_normalized = (depth_clamped - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
        size_factor = 1.5 - depth_normalized  # Closer = larger
        display_radius = int(self.target_size * scale * size_factor)
        display_radius = max(3, display_radius)  # Minimum size

        # Color based on team (if multi-agent)
        if team == 0:
            color = (255, 100, 100)  # Red team
        elif team == 1:
            color = (100, 100, 255)  # Blue team
        else:
            color = self.COLOR_TARGET  # Default red

        # Draw target circle
        pygame.draw.circle(
            self.surface,
            color,
            (screen_x, screen_y),
            display_radius,
            2
        )

        # Draw target velocity vector (relative to agent)
        # Target velocity in agent frame = target_vel - agent_vel (handled by caller)
        if np.linalg.norm(velocity[:2]) > 0.5:
            vel_end_x = int(screen_x + velocity[0] * scale * 2)
            vel_end_y = int(screen_y - velocity[1] * scale * 2)
            pygame.draw.line(
                self.surface,
                self.COLOR_VELOCITY,
                (screen_x, screen_y),
                (vel_end_x, vel_end_y),
                2
            )
            # Arrow tip
            pygame.draw.circle(
                self.surface,
                self.COLOR_VELOCITY,
                (vel_end_x, vel_end_y),
                3
            )

        # Draw depth indicator (distance text)
        font = pygame.font.Font(None, 20)
        depth_text = font.render(f"{depth:.1f}", True, (200, 200, 200))
        self.surface.blit(depth_text, (screen_x + display_radius + 5, screen_y - 10))

    def _draw_velocity_vector(
        self,
        velocity: np.ndarray,
        center_x: int,
        center_y: int,
        scale: float,
        color: Tuple[int, int, int]
    ):
        """Draw velocity vector as arrow from center."""
        # Agent velocity (green arrow from center)
        if np.linalg.norm(velocity[:2]) > 0.5:
            vel_end_x = int(center_x + velocity[0] * scale * 2)
            vel_end_y = int(center_y - velocity[1] * scale * 2)

            pygame.draw.line(
                self.surface,
                color,
                (center_x, center_y),
                (vel_end_x, vel_end_y),
                3
            )

            # Arrow tip
            pygame.draw.circle(
                self.surface,
                color,
                (vel_end_x, vel_end_y),
                4
            )

    def _draw_crosshair(self, center_x: int, center_y: int):
        """Draw crosshair at center (agent position)."""
        crosshair_size = 10
        pygame.draw.line(
            self.surface,
            self.COLOR_CROSSHAIR,
            (center_x - crosshair_size, center_y),
            (center_x + crosshair_size, center_y),
            2
        )
        pygame.draw.line(
            self.surface,
            self.COLOR_CROSSHAIR,
            (center_x, center_y - crosshair_size),
            (center_x, center_y + crosshair_size),
            2
        )

        # Center dot
        pygame.draw.circle(
            self.surface,
            self.COLOR_CROSSHAIR,
            (center_x, center_y),
            3
        )

    def _draw_hud_info(
        self,
        agent_velocity: np.ndarray,
        targets: List[Dict[str, Any]],
        info: Dict[str, Any],
        agent_id: Optional[int]
    ):
        """Draw HUD text information."""
        font = pygame.font.Font(None, 24)
        y_offset = 10

        # Agent ID (if provided)
        if agent_id is not None:
            id_text = font.render(f"Agent {agent_id}", True, self.COLOR_TEXT)
            self.surface.blit(id_text, (10, y_offset))
            y_offset += 25

        # Agent velocity magnitude
        vel_mag = np.linalg.norm(agent_velocity)
        vel_text = font.render(f"Vel: {vel_mag:.1f}", True, self.COLOR_AGENT)
        self.surface.blit(vel_text, (10, y_offset))
        y_offset += 25

        # Target information (closest target)
        if targets:
            closest_target = min(targets, key=lambda t: np.linalg.norm(t['position']))
            target_pos = closest_target['position']
            distance = np.linalg.norm(target_pos[:2])  # 2D distance

            dist_text = font.render(f"Dist: {distance:.1f}", True, self.COLOR_TARGET)
            self.surface.blit(dist_text, (10, y_offset))
            y_offset += 25

            # Focus status
            in_focus = distance < self.success_threshold
            focus_color = self.COLOR_FOCUS_SUCCESS if in_focus else self.COLOR_FOCUS_FAIL
            focus_status = "FOCUSED" if in_focus else "TRACKING"
            focus_text = font.render(focus_status, True, focus_color)
            self.surface.blit(focus_text, (10, y_offset))
            y_offset += 25

            # Error magnitude (distance from center)
            error_text = font.render(f"Error: {distance:.1f}", True, self.COLOR_TEXT)
            self.surface.blit(error_text, (10, y_offset))

        # Additional info from environment
        if 'mode' in info:
            mode_text = font.render(f"Mode: {info['mode']}", True, self.COLOR_TEXT)
            self.surface.blit(mode_text, (10, self.render_size - 30))


def create_hud_renderer(config: Dict[str, Any], render_size: int = 400) -> HUDRenderer3D:
    """
    Factory function to create HUD renderer from config.

    Args:
        config: Configuration dictionary
        render_size: Display size in pixels

    Returns:
        Configured HUDRenderer3D instance
    """
    env_config = config.get("environment", {})

    return HUDRenderer3D(
        frame_size=env_config.get("frame_size", 64),
        view_size=env_config.get("view_size", env_config.get("view_radius", 30.0)),
        depth_range=tuple(env_config.get("depth_range", [10.0, 50.0])),
        success_threshold=env_config.get("success_threshold", 9.0),
        target_size=env_config.get("target_size", 4.0),
        agent_size=env_config.get("agent_size", 1.5),
        render_size=render_size,
    )
