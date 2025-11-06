"""
Integrated Multi-Agent Environment with Egocentric HUD Perspectives

This combines:
1. Global 3D arena (geographic coordinate system)
2. Individual egocentric HUD for each agent (pursuit-evasion view)
3. Trajectory tracking and visualization
4. Multi-view rendering

Each agent:
- Moves in global 3D space
- Sees egocentric HUD (like Phase 1)
- Has own trajectory history
- Can be displayed simultaneously
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import List, Tuple, Dict, Any, Optional
from collections import deque


class AgentWithPerspective:
    """
    Agent with both global position and egocentric perspective.
    """

    def __init__(
        self,
        agent_id: int,
        team: int,
        position: np.ndarray,
        velocity: np.ndarray,
        color: Tuple[int, int, int],
    ):
        self.id = agent_id
        self.team = team
        self.position = position.astype(np.float32)
        self.velocity = velocity.astype(np.float32)
        self.orientation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.color = color
        self.alive = True
        self.health = 100.0

        # Target tracking (for egocentric view)
        self.locked_target = None  # ID of tracked target

        # Trajectory history (for visualization)
        self.trajectory = deque(maxlen=200)  # Last 200 positions
        self.trajectory.append(position.copy())

        # Egocentric observation frame
        self.frame_size = 64
        self.view_size = 30.0  # FOV size for egocentric view
        self.egocentric_frame = None


class MultiAgentIntegratedEnv(gym.Env):
    """
    Integrated multi-agent environment combining:
    - Global geographic coordinates
    - Egocentric HUD perspectives for each agent
    - Trajectory visualization
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        num_agents: int = 4,
        num_teams: int = 2,
        arena_size: float = 1000.0,
        max_velocity: float = 50.0,
        max_acceleration: float = 5.0,
        fov_range: float = 300.0,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
        frame_size: int = 64,
        view_size: float = 30.0,
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
        self.frame_size = frame_size
        self.view_size = view_size

        # Observation space: Egocentric frame for each agent
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_size, frame_size),
            dtype=np.uint8
        )

        # Action space: 3D acceleration
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Agent colors
        self.team_colors = [
            (0, 255, 0),    # Team 0: Green
            (255, 0, 0),    # Team 1: Red
            (0, 0, 255),    # Team 2: Blue
            (255, 255, 0),  # Team 3: Yellow
        ]

        # Agents list
        self.agents: List[AgentWithPerspective] = []

        # Pygame surfaces for rendering
        self.screen = None
        self.clock = None
        self.global_map_surface = None
        self.hud_surfaces = {}  # One HUD per agent

        self.step_count = 0
        self.max_steps = 3000

    def _create_agents(self):
        """Initialize all agents."""
        self.agents = []

        agents_per_team = self.num_agents // self.num_teams if self.num_teams > 0 else self.num_agents

        for i in range(self.num_agents):
            team = i // agents_per_team if self.num_teams > 0 else i
            color = self.team_colors[team % len(self.team_colors)]

            # Spawn position based on team
            if team == 0:
                pos = self.np_random.uniform(
                    [-self.arena_size / 4, -self.arena_size / 4, 100],
                    [-self.arena_size / 8, -self.arena_size / 8, 200],
                    size=3
                )
            elif team == 1:
                pos = self.np_random.uniform(
                    [self.arena_size / 8, self.arena_size / 8, 100],
                    [self.arena_size / 4, self.arena_size / 4, 200],
                    size=3
                )
            else:
                pos = self.np_random.uniform(
                    -self.arena_size / 4,
                    self.arena_size / 4,
                    size=3
                )

            vel = self.np_random.uniform(-10, 10, size=3)

            agent = AgentWithPerspective(i, team, pos, vel, color)
            self.agents.append(agent)

    def _get_egocentric_observation(self, agent: AgentWithPerspective) -> np.ndarray:
        """
        Get egocentric HUD observation for specific agent.

        This creates the pursuit-evasion HUD view (Phase 1 style)
        but from the perspective of each agent in the global arena.
        """
        # Create empty frame
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

        # Find nearest enemy as target
        nearest_enemy = None
        min_dist = float('inf')

        for other in self.agents:
            if other.id == agent.id or not other.alive:
                continue
            if self.num_teams > 0 and other.team == agent.team:
                continue

            dist = np.linalg.norm(other.position - agent.position)
            if dist < min_dist:
                min_dist = dist
                nearest_enemy = other

        if nearest_enemy is None:
            return frame

        # Transform enemy position to agent's egocentric frame
        # Relative position
        relative_pos = nearest_enemy.position - agent.position

        # Rotate to agent's orientation (simplified: assume agent looks forward)
        # In full version, would use proper rotation matrix
        relative_pos_2d = relative_pos[:2]  # Take XY only

        # Check if in view
        if np.linalg.norm(relative_pos_2d) > self.view_size:
            return frame

        # Convert to pixel coordinates
        center = self.frame_size // 2
        scale = self.frame_size / (2 * self.view_size)

        pixel_x = int(center + relative_pos_2d[0] * scale)
        pixel_y = int(center - relative_pos_2d[1] * scale)

        # Check bounds
        if 0 <= pixel_x < self.frame_size and 0 <= pixel_y < self.frame_size:
            # Draw target (size based on distance/depth)
            distance_3d = np.linalg.norm(relative_pos)
            size = max(2, int(10 * (1.0 - distance_3d / self.fov_range)))

            # Draw filled circle
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    if dx * dx + dy * dy <= size * size:
                        px = pixel_x + dx
                        py = pixel_y + dy
                        if 0 <= px < self.frame_size and 0 <= py < self.frame_size:
                            frame[py, px] = 255

        # Store for rendering
        agent.egocentric_frame = frame

        return frame

    def _render_global_map(self, surface: pygame.Surface):
        """
        Render global geographic view (top-down).
        Shows all agents and their trajectories.
        """
        surface.fill((20, 20, 40))

        # Arena bounds
        width, height = surface.get_size()
        center_x, center_y = width // 2, height // 2
        scale = min(width, height) / self.arena_size

        # Draw arena boundary
        arena_rect_size = int(self.arena_size * scale)
        pygame.draw.rect(
            surface,
            (60, 60, 80),
            (center_x - arena_rect_size // 2, center_y - arena_rect_size // 2,
             arena_rect_size, arena_rect_size),
            2
        )

        # Draw trajectories
        for agent in self.agents:
            if not agent.alive or len(agent.trajectory) < 2:
                continue

            # Draw trajectory line
            points = []
            for pos in agent.trajectory:
                x = int(center_x + pos[0] * scale)
                y = int(center_y - pos[1] * scale)
                points.append((x, y))

            if len(points) >= 2:
                pygame.draw.lines(surface, agent.color, False, points, 1)

        # Draw agents
        for agent in self.agents:
            if not agent.alive:
                continue

            x = int(center_x + agent.position[0] * scale)
            y = int(center_y - agent.position[1] * scale)

            # Draw agent
            pygame.draw.circle(surface, agent.color, (x, y), 6)

            # Draw orientation arrow
            orient_2d = agent.orientation[:2]
            if np.linalg.norm(orient_2d) > 0.1:
                orient_2d = orient_2d / np.linalg.norm(orient_2d)
                end_x = int(x + orient_2d[0] * 15)
                end_y = int(y - orient_2d[1] * 15)
                pygame.draw.line(surface, agent.color, (x, y), (end_x, end_y), 2)

            # Draw agent ID
            font = pygame.font.Font(None, 20)
            id_text = font.render(str(agent.id), True, (255, 255, 255))
            surface.blit(id_text, (x - 5, y - 20))

        # Draw title
        font = pygame.font.Font(None, 24)
        title = font.render("Global Arena - Geographic Coordinates", True, (255, 255, 255))
        surface.blit(title, (10, 10))

    def _render_egocentric_hud(self, agent: AgentWithPerspective, surface: pygame.Surface):
        """
        Render egocentric HUD for specific agent.
        This is the pursuit-evasion view (Phase 1 style).
        """
        surface.fill((10, 10, 20))

        # Draw frame (if available)
        if agent.egocentric_frame is not None:
            # Scale up to surface size
            width, height = surface.get_size()
            scaled_frame = pygame.surfarray.make_surface(
                np.repeat(np.repeat(agent.egocentric_frame, width // self.frame_size, axis=0),
                          height // self.frame_size, axis=1)
            )
            surface.blit(scaled_frame, (0, 0))

        # Draw crosshair (agent at center)
        center_x, center_y = surface.get_width() // 2, surface.get_height() // 2
        pygame.draw.line(surface, (0, 255, 0), (center_x - 15, center_y), (center_x + 15, center_y), 2)
        pygame.draw.line(surface, (0, 255, 0), (center_x, center_y - 15), (center_x, center_y + 15), 2)

        # Draw FOV boundary (square)
        border_margin = 20
        pygame.draw.rect(
            surface,
            (100, 100, 120),
            (border_margin, border_margin,
             surface.get_width() - 2 * border_margin,
             surface.get_height() - 2 * border_margin),
            2
        )

        # Draw HUD info
        font = pygame.font.Font(None, 20)

        # Agent info
        info_lines = [
            f"Agent {agent.id} - Team {agent.team}",
            f"Health: {agent.health:.0f}%",
            f"Velocity: {np.linalg.norm(agent.velocity):.1f}",
            f"Altitude: {agent.position[2]:.0f}m",
        ]

        y_offset = 10
        for line in info_lines:
            text = font.render(line, True, agent.color)
            surface.blit(text, (10, y_offset))
            y_offset += 20

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self._create_agents()
        self.step_count = 0

        # Get observation for agent 0 (in single-agent interface)
        obs = self._get_egocentric_observation(self.agents[0])

        info = {
            "num_alive": sum(1 for a in self.agents if a.alive),
            "global_positions": [a.position.copy() for a in self.agents],
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Note: This is single-agent interface. For true multi-agent,
        use MultiAgentEnv or pass action dict.
        """
        self.step_count += 1

        # Update agent 0 (controlled agent)
        agent = self.agents[0]
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

            # Update trajectory
            agent.trajectory.append(agent.position.copy())

        # Update other agents (simple AI)
        for other in self.agents[1:]:
            if not other.alive:
                continue

            # Simple AI: Move towards nearest enemy
            nearest_enemy = None
            min_dist = float('inf')

            for target in self.agents:
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

                if vel_mag > 1.0:
                    other.orientation = other.velocity / vel_mag

                # Update trajectory
                other.trajectory.append(other.position.copy())

        # Keep all in bounds
        for a in self.agents:
            a.position = np.clip(a.position, -self.arena_size / 2, self.arena_size / 2)
            # Keep above ground
            if a.position[2] < 10.0:
                a.position[2] = 10.0
                a.velocity[2] = abs(a.velocity[2])

        # Get observation for agent 0
        obs = self._get_egocentric_observation(agent)

        # Simple reward
        if agent.alive:
            nearest_dist = float('inf')
            for other in self.agents:
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
        num_alive = sum(1 for a in self.agents if a.alive)
        terminated = num_alive <= 1 or not agent.alive
        truncated = self.step_count >= self.max_steps

        info = {
            "num_alive": num_alive,
            "agent_alive": agent.alive,
            "global_positions": [a.position.copy() for a in self.agents],
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render environment with multi-view:
        - Left: Global geographic map
        - Right: Grid of egocentric HUDs for each agent
        """
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                # Large screen: 1600x900 (global map + HUD grid)
                self.screen = pygame.display.set_mode((1600, 900))
                pygame.display.set_caption("Multi-Agent Integrated View")
            else:
                self.screen = pygame.Surface((1600, 900))

            # Create surfaces
            self.global_map_surface = pygame.Surface((800, 900))

            # Create HUD surfaces (2x2 grid for 4 agents)
            hud_width, hud_height = 400, 450
            for i in range(min(self.num_agents, 4)):
                self.hud_surfaces[i] = pygame.Surface((hud_width, hud_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Render global map (left half)
        self._render_global_map(self.global_map_surface)
        self.screen.blit(self.global_map_surface, (0, 0))

        # Render egocentric HUDs (right half, 2x2 grid)
        for i in range(min(self.num_agents, 4)):
            if i < len(self.agents):
                self._render_egocentric_hud(self.agents[i], self.hud_surfaces[i])

                # Position in grid
                grid_x = 800 + (i % 2) * 400
                grid_y = (i // 2) * 450
                self.screen.blit(self.hud_surfaces[i], (grid_x, grid_y))

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
