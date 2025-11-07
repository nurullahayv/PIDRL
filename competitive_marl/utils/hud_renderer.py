"""
3D HUD Renderer for Competitive MARL

Provides demo_3d style egocentric HUD visualization.
Optimized for local testing (not used during Kaggle training).
"""

import numpy as np
import pygame
from typing import Optional, Tuple


class HUDRenderer3D:
    """
    3D HUD renderer for pursuit-evasion visualization.

    Features:
    - Square radar view (not circular)
    - Velocity vectors for both agents
    - Depth-based target sizing
    - Focus status display
    - Real-time info panel
    """

    def __init__(
        self,
        view_size: float = 30.0,
        success_threshold: float = 9.0,
        target_size: float = 4.0,
        screen_size: Tuple[int, int] = (800, 800),
        fps: int = 30,
    ):
        self.view_size = view_size
        self.success_threshold = success_threshold
        self.target_size = target_size
        self.screen_size = screen_size
        self.fps = fps

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Competitive MARL - 3D HUD")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)

        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (60, 60, 100)
        self.COLOR_PURSUER = (0, 255, 0)      # Green
        self.COLOR_EVADER = (255, 100, 100)   # Red
        self.COLOR_VELOCITY = (100, 200, 255)  # Cyan
        self.COLOR_CROSSHAIR = (255, 255, 0)   # Yellow
        self.COLOR_SUCCESS = (0, 255, 0)
        self.COLOR_FAIL = (255, 100, 100)
        self.COLOR_TEXT = (200, 200, 200)

    def render(
        self,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        agent_velocity: np.ndarray,
        in_focus: bool,
        distance: float,
        steps_in_focus: int = 0,
        focus_threshold: int = 50,
        episode_step: int = 0,
        episode_reward: float = 0.0,
    ):
        """
        Render HUD frame.

        Args:
            target_position: Target position [x, y, z] in agent frame
            target_velocity: Target velocity [vx, vy, vz]
            agent_velocity: Agent velocity [vx, vy, vz]
            in_focus: Whether target is in focus
            distance: Distance to target
            steps_in_focus: Current focus duration
            focus_threshold: Steps needed for focus bonus
            episode_step: Current step in episode
            episode_reward: Cumulative episode reward
        """
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Calculate screen parameters
        center_x = self.screen_size[0] // 2
        center_y = self.screen_size[1] // 2
        scale = min(self.screen_size) / self.view_size

        # 1. Draw radar grid
        self._draw_radar_grid(center_x, center_y, scale)

        # 2. Draw target
        self._draw_target(target_position, target_velocity, center_x, center_y, scale)

        # 3. Draw agent velocity vector
        self._draw_agent_velocity(agent_velocity, center_x, center_y, scale)

        # 4. Draw crosshair (agent at center)
        self._draw_crosshair(center_x, center_y)

        # 5. Draw HUD info
        self._draw_hud_info(
            in_focus, distance, agent_velocity, target_velocity,
            steps_in_focus, focus_threshold, episode_step, episode_reward
        )

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to close

        return True  # Continue rendering

    def _draw_radar_grid(self, center_x: int, center_y: int, scale: float):
        """Draw square radar boundaries and grid."""
        half_size = int(self.view_size * scale / 2)

        # Outer boundary (square)
        pygame.draw.rect(
            self.screen,
            self.COLOR_GRID,
            (center_x - half_size, center_y - half_size, half_size * 2, half_size * 2),
            2
        )

        # Inner boundary (focus threshold circle)
        threshold_radius = int(self.success_threshold * scale)
        pygame.draw.circle(
            self.screen,
            self.COLOR_GRID,
            (center_x, center_y),
            threshold_radius,
            1
        )

        # Cardinal direction lines
        pygame.draw.line(
            self.screen,
            self.COLOR_GRID,
            (center_x - half_size, center_y),
            (center_x + half_size, center_y),
            1
        )
        pygame.draw.line(
            self.screen,
            self.COLOR_GRID,
            (center_x, center_y - half_size),
            (center_x, center_y + half_size),
            1
        )

    def _draw_target(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        center_x: int,
        center_y: int,
        scale: float
    ):
        """Draw target (evader) with velocity vector."""
        # Project to 2D
        x_proj = position[0]
        y_proj = position[1]
        depth = position[2]

        # Screen coordinates
        screen_x = int(center_x + x_proj * scale)
        screen_y = int(center_y - y_proj * scale)

        # Depth-based sizing
        depth_clamped = np.clip(depth, 10.0, 50.0)
        depth_normalized = (depth_clamped - 10.0) / 40.0
        size_factor = 1.5 - depth_normalized
        display_radius = int(self.target_size * scale * size_factor)
        display_radius = max(5, display_radius)

        # Draw target circle (RED)
        pygame.draw.circle(
            self.screen,
            self.COLOR_EVADER,
            (screen_x, screen_y),
            display_radius,
            3
        )

        # Draw velocity vector
        if np.linalg.norm(velocity[:2]) > 0.5:
            vel_end_x = int(screen_x + velocity[0] * scale * 3)
            vel_end_y = int(screen_y - velocity[1] * scale * 3)
            pygame.draw.line(
                self.screen,
                self.COLOR_VELOCITY,
                (screen_x, screen_y),
                (vel_end_x, vel_end_y),
                2
            )
            pygame.draw.circle(
                self.screen,
                self.COLOR_VELOCITY,
                (vel_end_x, vel_end_y),
                4
            )

        # Draw depth text
        depth_text = self.font_small.render(f"{depth:.1f}m", True, self.COLOR_TEXT)
        self.screen.blit(depth_text, (screen_x + display_radius + 5, screen_y - 10))

    def _draw_agent_velocity(
        self,
        velocity: np.ndarray,
        center_x: int,
        center_y: int,
        scale: float
    ):
        """Draw agent (pursuer) velocity vector."""
        if np.linalg.norm(velocity[:2]) > 0.5:
            vel_end_x = int(center_x + velocity[0] * scale * 3)
            vel_end_y = int(center_y - velocity[1] * scale * 3)

            # Green arrow from center
            pygame.draw.line(
                self.screen,
                self.COLOR_PURSUER,
                (center_x, center_y),
                (vel_end_x, vel_end_y),
                3
            )
            pygame.draw.circle(
                self.screen,
                self.COLOR_PURSUER,
                (vel_end_x, vel_end_y),
                5
            )

    def _draw_crosshair(self, center_x: int, center_y: int):
        """Draw crosshair at center (agent position)."""
        size = 15
        pygame.draw.line(
            self.screen,
            self.COLOR_CROSSHAIR,
            (center_x - size, center_y),
            (center_x + size, center_y),
            2
        )
        pygame.draw.line(
            self.screen,
            self.COLOR_CROSSHAIR,
            (center_x, center_y - size),
            (center_x, center_y + size),
            2
        )
        pygame.draw.circle(
            self.screen,
            self.COLOR_CROSSHAIR,
            (center_x, center_y),
            4
        )

    def _draw_hud_info(
        self,
        in_focus: bool,
        distance: float,
        agent_velocity: np.ndarray,
        target_velocity: np.ndarray,
        steps_in_focus: int,
        focus_threshold: int,
        episode_step: int,
        episode_reward: float,
    ):
        """Draw HUD information panel."""
        y_offset = 10

        # Title
        title = self.font.render("PURSUER HUD", True, self.COLOR_PURSUER)
        self.screen.blit(title, (10, y_offset))
        y_offset += 30

        # Agent velocity
        agent_vel_mag = np.linalg.norm(agent_velocity)
        vel_text = self.font.render(f"Vel: {agent_vel_mag:.1f}", True, self.COLOR_PURSUER)
        self.screen.blit(vel_text, (10, y_offset))
        y_offset += 25

        # Target distance
        dist_text = self.font.render(f"Dist: {distance:.1f}", True, self.COLOR_EVADER)
        self.screen.blit(dist_text, (10, y_offset))
        y_offset += 25

        # Target velocity
        target_vel_mag = np.linalg.norm(target_velocity)
        tvel_text = self.font.render(f"T-Vel: {target_vel_mag:.1f}", True, self.COLOR_EVADER)
        self.screen.blit(tvel_text, (10, y_offset))
        y_offset += 30

        # Focus status
        focus_color = self.COLOR_SUCCESS if in_focus else self.COLOR_FAIL
        focus_status = "FOCUSED" if in_focus else "TRACKING"
        focus_text = self.font.render(focus_status, True, focus_color)
        self.screen.blit(focus_text, (10, y_offset))
        y_offset += 25

        # Focus progress
        if in_focus:
            progress = steps_in_focus / focus_threshold
            progress_text = self.font_small.render(
                f"Focus: {steps_in_focus}/{focus_threshold} ({progress*100:.0f}%)",
                True, self.COLOR_TEXT
            )
            self.screen.blit(progress_text, (10, y_offset))
            y_offset += 25

        # Episode info (bottom)
        y_offset = self.screen_size[1] - 60

        step_text = self.font_small.render(f"Step: {episode_step}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (10, y_offset))
        y_offset += 25

        reward_text = self.font_small.render(f"Reward: {episode_reward:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(reward_text, (10, y_offset))

    def close(self):
        """Close renderer."""
        pygame.quit()
