"""
Target Evader PID Controller

This controller makes the target try to evade the agent and escape
the field of view (FOV) by moving towards the boundaries.

Strategy:
- Move away from center (escape to boundaries)
- Add some randomness for unpredictability
- Use PID control for smooth acceleration commands
"""

import numpy as np
from typing import Tuple, Optional


class TargetEvaderPID:
    """
    PID controller for target evasion behavior.

    The target tries to:
    1. Move towards the FOV boundaries (away from center)
    2. Add random perturbations for unpredictability
    3. Avoid staying in one place
    """

    def __init__(
        self,
        kp: float = 0.8,
        ki: float = 0.02,
        kd: float = 0.3,
        view_size: float = 30.0,
        escape_margin: float = 5.0,
        randomness: float = 0.3,
        dt: float = 0.1,
    ):
        """
        Initialize target evader PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            view_size: Size of the FOV (target tries to escape this area)
            escape_margin: Distance from boundary to target (safety margin)
            randomness: Amount of random perturbation (0-1)
            dt: Time step
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.view_size = view_size
        self.escape_margin = escape_margin
        self.randomness = randomness
        self.dt = dt

        # PID state
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)

        # Evasion strategy
        self.escape_direction = None
        self.direction_change_timer = 0
        self.direction_change_interval = 50  # Change direction every N steps

    def reset(self):
        """Reset PID state."""
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        self.escape_direction = None
        self.direction_change_timer = 0

    def _get_escape_direction(self, position: np.ndarray) -> np.ndarray:
        """
        Determine the best direction to escape.

        Strategy:
        - If near center, move towards nearest boundary
        - If already near boundary, move along boundary
        - Add randomness for unpredictability

        Args:
            position: Current position [x, y, z]

        Returns:
            Escape direction vector (normalized)
        """
        x, y, z = position

        # Calculate distance from center
        distance_from_center = np.linalg.norm(position)

        # Determine escape strategy based on position
        if distance_from_center < self.view_size * 0.5:
            # Near center: Move towards nearest boundary
            # Find the axis with smallest absolute value (easiest escape)
            abs_pos = np.abs(position)
            min_axis = np.argmin(abs_pos)

            escape_dir = np.zeros(3)
            escape_dir[min_axis] = np.sign(position[min_axis]) if position[min_axis] != 0 else 1.0

        else:
            # Near boundary: Move along boundary (tangential movement)
            # This makes it harder for agent to predict

            # Get perpendicular direction to radial vector
            radial = position / (np.linalg.norm(position) + 1e-8)

            # Create a perpendicular vector
            if abs(radial[0]) < 0.9:
                perpendicular = np.cross(radial, [1, 0, 0])
            else:
                perpendicular = np.cross(radial, [0, 1, 0])

            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)

            # Mix radial (escape) and perpendicular (evasion)
            escape_dir = 0.3 * radial + 0.7 * perpendicular

        # Normalize
        escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-8)

        # Add random perturbation
        if self.randomness > 0:
            random_dir = np.random.randn(3)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)

            escape_dir = (1 - self.randomness) * escape_dir + self.randomness * random_dir
            escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-8)

        return escape_dir

    def compute_acceleration(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute acceleration command for target evasion.

        Args:
            position: Current position [x, y, z] (relative to agent)
            velocity: Current velocity [vx, vy, vz]

        Returns:
            Acceleration command [ax, ay, az] (normalized to [-1, 1])
        """
        # Update escape direction periodically
        self.direction_change_timer += 1
        if self.escape_direction is None or self.direction_change_timer >= self.direction_change_interval:
            self.escape_direction = self._get_escape_direction(position)
            self.direction_change_timer = 0

        # Compute desired position (escape point)
        desired_distance = self.view_size - self.escape_margin
        desired_position = self.escape_direction * desired_distance

        # PID control to reach desired position
        error = desired_position - position

        # Update integral
        self.integral_error += error * self.dt

        # Clamp integral to prevent windup
        integral_max = 10.0
        self.integral_error = np.clip(self.integral_error, -integral_max, integral_max)

        # Compute derivative
        derivative_error = (error - self.last_error) / self.dt
        self.last_error = error.copy()

        # PID output
        acceleration = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative_error
        )

        # Normalize to [-1, 1] range
        max_acc = np.linalg.norm(acceleration)
        if max_acc > 1e-6:
            acceleration = acceleration / max_acc

        return acceleration

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, None]:
        """
        Predict action (compatible with agent interface).

        Args:
            observation: Current observation (not used, we use internal state)

        Returns:
            action: Acceleration command
            state: None (stateless for compatibility)
        """
        # For this controller, we don't use the observation directly
        # The environment will provide position and velocity separately
        # This method is just for interface compatibility
        return np.zeros(3), None


class TargetEvaderAgent:
    """
    Wrapper class to make TargetEvaderPID compatible with agent interface.
    """

    def __init__(self, config: dict):
        """
        Initialize target evader agent.

        Args:
            config: Configuration dictionary
        """
        evader_config = config.get("target_evader", {})

        self.controller = TargetEvaderPID(
            kp=evader_config.get("kp", 0.8),
            ki=evader_config.get("ki", 0.02),
            kd=evader_config.get("kd", 0.3),
            view_size=config["environment"].get("view_size", 30.0),
            escape_margin=evader_config.get("escape_margin", 5.0),
            randomness=evader_config.get("randomness", 0.3),
            dt=config["environment"].get("dt", 0.1),
        )

    def predict(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action based on current state.

        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            deterministic: Not used (for compatibility)

        Returns:
            action: Acceleration command
            state: None
        """
        action = self.controller.compute_acceleration(position, velocity)
        return action, None

    def reset(self):
        """Reset controller state."""
        self.controller.reset()
