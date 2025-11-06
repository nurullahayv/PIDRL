"""
3D Visualization Module with MATLAB-style Plotting

This module provides 3D visualization for the multi-agent system:
- MATLAB-style 3D plot
- Agent trajectories
- Camera FOV cones
- Real-time updates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pygame


class MATLABStyle3DVisualizer:
    """
    MATLAB-style 3D visualization for multi-agent arena.

    Features:
    - 3D scatter plot for agents
    - Trajectory lines
    - FOV cones for each agent
    - Grid and axis labels
    - Real-time rotation capability
    """

    def __init__(self, arena_size=1000.0, fov_angle=60.0, fov_range=300.0, figsize=(10, 8)):
        """
        Initialize 3D visualizer.

        Args:
            arena_size: Size of arena
            fov_angle: Field of view angle in degrees
            fov_range: FOV range
            figsize: Figure size (width, height)
        """
        self.arena_size = arena_size
        self.fov_angle = np.deg2rad(fov_angle)
        self.fov_range = fov_range

        # Create figure
        self.fig = Figure(figsize=figsize, dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Canvas for rendering to surface
        self.canvas = FigureCanvasAgg(self.fig)

        # View angle (can be changed for rotation)
        self.elevation = 30
        self.azimuth = 45

        # Setup axes
        self._setup_axes()

    def _setup_axes(self):
        """Setup 3D axes with MATLAB style."""
        # Set limits
        lim = self.arena_size / 2
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(0, lim)

        # Labels
        self.ax.set_xlabel('X (m)', fontsize=10, labelpad=8)
        self.ax.set_ylabel('Y (m)', fontsize=10, labelpad=8)
        self.ax.set_zlabel('Z (m)', fontsize=10, labelpad=8)

        # Title
        self.ax.set_title('Global 3D Arena - Geographic Coordinates', fontsize=12, pad=20)

        # Grid
        self.ax.grid(True, alpha=0.3)

        # Background color (MATLAB style)
        self.ax.set_facecolor('#f0f0f0')
        self.fig.patch.set_facecolor('white')

    def _create_fov_cone(self, position, orientation, color, alpha=0.2):
        """
        Create FOV cone mesh for visualization.

        Args:
            position: Agent position [x, y, z]
            orientation: Agent orientation vector [dx, dy, dz]
            color: Cone color
            alpha: Transparency

        Returns:
            Poly3DCollection for the cone
        """
        # Normalize orientation
        orient_norm = orientation / (np.linalg.norm(orientation) + 1e-8)

        # Cone apex at agent position
        apex = position

        # Create cone base circle
        # Radius at range distance
        radius = self.fov_range * np.tan(self.fov_angle / 2)

        # Center of base circle
        base_center = apex + orient_norm * self.fov_range

        # Create perpendicular vectors for circle
        # Find a vector perpendicular to orientation
        if abs(orient_norm[0]) < 0.9:
            perp1 = np.cross(orient_norm, [1, 0, 0])
        else:
            perp1 = np.cross(orient_norm, [0, 1, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)

        perp2 = np.cross(orient_norm, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)

        # Create base circle points
        num_points = 16
        theta = np.linspace(0, 2 * np.pi, num_points)

        base_points = []
        for t in theta:
            point = base_center + radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
            base_points.append(point)

        base_points = np.array(base_points)

        # Create cone faces (triangles from apex to base)
        faces = []
        for i in range(num_points):
            next_i = (i + 1) % num_points
            # Triangle: apex, base_point_i, base_point_next
            face = [apex, base_points[i], base_points[next_i]]
            faces.append(face)

        # Create mesh
        cone_mesh = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='none')

        return cone_mesh

    def render(self, agents):
        """
        Render the 3D scene with agents, trajectories, and FOV cones.

        Args:
            agents: List of AgentWithPerspective objects

        Returns:
            pygame.Surface with rendered 3D view
        """
        # Clear axes
        self.ax.cla()
        self._setup_axes()

        # Set view angle
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)

        # Draw agents and their components
        for agent in agents:
            # Support both dict and object formats
            if isinstance(agent, dict):
                agent_id = agent['id']
                position = np.array(agent['position'])
                orientation = np.array(agent['orientation'])
                color = agent['color']
                trajectory = agent['trajectory']
            else:
                # Object format
                if not agent.alive:
                    continue
                agent_id = agent.id
                position = agent.position
                orientation = agent.orientation
                color = tuple(c / 255.0 for c in agent.color)
                trajectory = agent.trajectory

            # 1. Draw trajectory
            if len(trajectory) > 1:
                traj_array = np.array(list(trajectory))
                self.ax.plot(
                    traj_array[:, 0],
                    traj_array[:, 1],
                    traj_array[:, 2],
                    color=color,
                    alpha=0.5,
                    linewidth=1.5,
                    label=f'Agent {agent_id} trajectory'
                )

            # 2. Draw agent position (larger marker)
            self.ax.scatter(
                position[0],
                position[1],
                position[2],
                color=color,
                s=200,  # Size
                marker='o',
                edgecolors='black',
                linewidths=1.5,
                label=f'Agent {agent_id}'
            )

            # 3. Draw orientation arrow
            arrow_length = 50.0
            arrow_end = position + orientation * arrow_length
            self.ax.plot(
                [position[0], arrow_end[0]],
                [position[1], arrow_end[1]],
                [position[2], arrow_end[2]],
                color=color,
                linewidth=2,
                linestyle='--'
            )

            # 4. Draw FOV cone
            cone = self._create_fov_cone(position, orientation, color, alpha=0.15)
            self.ax.add_collection3d(cone)

            # 5. Add agent ID text
            self.ax.text(
                position[0],
                position[1],
                position[2] + 30,
                f'#{agent_id}',
                fontsize=9,
                color='black',
                ha='center'
            )

        # Add legend (only for first 4 agents to avoid clutter)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            # Get unique labels (trajectory + agent)
            unique_labels = []
            unique_handles = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen:
                    unique_labels.append(l)
                    unique_handles.append(h)
                    seen.add(l)

            if len(unique_handles) <= 8:  # Show legend if not too many
                self.ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=8)

        # Draw arena boundary box
        self._draw_arena_box()

        # Render to canvas
        self.canvas.draw()

        # Convert to pygame surface
        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()

        surface = pygame.image.fromstring(raw_data, size, "RGB")

        return surface

    def _draw_arena_box(self):
        """Draw arena boundary box."""
        lim = self.arena_size / 2

        # Bottom square
        bottom = [
            [-lim, -lim, 0],
            [lim, -lim, 0],
            [lim, lim, 0],
            [-lim, lim, 0]
        ]

        # Draw bottom
        bottom_array = np.array(bottom + [bottom[0]])  # Close the loop
        self.ax.plot(
            bottom_array[:, 0],
            bottom_array[:, 1],
            bottom_array[:, 2],
            'k-',
            alpha=0.3,
            linewidth=0.5
        )

        # Vertical lines
        for corner in bottom:
            self.ax.plot(
                [corner[0], corner[0]],
                [corner[1], corner[1]],
                [0, lim],
                'k-',
                alpha=0.3,
                linewidth=0.5
            )

    def rotate_view(self, delta_azimuth=0, delta_elevation=0):
        """
        Rotate the 3D view.

        Args:
            delta_azimuth: Change in azimuth angle
            delta_elevation: Change in elevation angle
        """
        self.azimuth += delta_azimuth
        self.elevation = np.clip(self.elevation + delta_elevation, -90, 90)
