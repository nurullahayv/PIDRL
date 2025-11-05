"""
Kalman Filter + PID Controller for robust vision-based tracking.

This controller combines Kalman filtering for state estimation with
PID control for smooth and robust tracking performance.
"""

import numpy as np
from typing import Dict, Any, Optional
from controllers.kalman_filter import KalmanFilter
from utils.visual_detection import VisualDetector


class KalmanPIDController:
    """
    Combined Kalman Filter + PID controller.

    Pipeline:
        1. Visual detection extracts noisy position measurements
        2. Kalman Filter estimates smooth position and velocity
        3. PID controller generates acceleration commands based on filtered estimates
    """

    def __init__(
        self,
        # PID parameters
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.2,
        integral_limit: float = 10.0,
        # Kalman Filter parameters
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        measurement_noise: float = 2.0,
        initial_covariance: float = 10.0,
        # Environment parameters
        dt: float = 0.1,
        frame_size: int = 64,
        view_radius: float = 30.0,
        # Additional options
        use_velocity_feedforward: bool = True,
    ):
        """
        Initialize Kalman-PID controller.

        Args:
            kp, ki, kd: PID gains
            integral_limit: Anti-windup limit
            process_noise_pos, process_noise_vel: Kalman process noise
            measurement_noise: Kalman measurement noise
            initial_covariance: Kalman initial covariance
            dt: Time step
            frame_size: Observation frame size
            view_radius: View radius in world units
            use_velocity_feedforward: Whether to use velocity estimate in control
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.dt = dt
        self.use_velocity_feedforward = use_velocity_feedforward

        # Visual detector
        self.detector = VisualDetector(frame_size=frame_size)

        # Kalman Filter
        self.kalman = KalmanFilter(
            dt=dt,
            process_noise_pos=process_noise_pos,
            process_noise_vel=process_noise_vel,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
        )

        # Conversion factor from pixels to world units
        self.pixel_to_world = (2 * view_radius) / frame_size

        # PID state
        self.integral = np.zeros(2, dtype=np.float32)
        self.previous_error = np.zeros(2, dtype=np.float32)
        self.first_step = True

        # Statistics
        self.detections_failed = 0
        self.total_steps = 0

    def reset(self):
        """Reset controller state."""
        self.kalman.reset()
        self.integral = np.zeros(2, dtype=np.float32)
        self.previous_error = np.zeros(2, dtype=np.float32)
        self.first_step = True
        self.detections_failed = 0
        self.total_steps = 0

    def compute_action(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute control action from observation.

        Args:
            observation: Stacked frames (num_frames, H, W) or (H, W, num_frames)

        Returns:
            Tuple of (action, info):
                - action: 2D acceleration command in [-1, 1]^2
                - info: Dictionary with debug information
        """
        self.total_steps += 1

        # Detect target position (in pixels relative to center)
        position_pixels, confidence = self.detector.detect_target_stacked(observation)

        info = {
            "detection_success": position_pixels is not None,
            "confidence": confidence,
        }

        # Prepare measurement for Kalman Filter
        if position_pixels is not None:
            measurement = position_pixels * self.pixel_to_world
        else:
            measurement = None
            self.detections_failed += 1

        # Update Kalman Filter
        state = self.kalman.step(measurement)
        estimated_position = state[0:2]  # [px, py]
        estimated_velocity = state[2:4]  # [vx, vy]

        info["estimated_position"] = estimated_position
        info["estimated_velocity"] = estimated_velocity
        info["raw_measurement"] = measurement if measurement is not None else np.zeros(2)

        # Compute error signal (target position relative to agent center)
        error = estimated_position

        # PID Control
        # Proportional term
        proportional = self.kp * error

        # Integral term (with anti-windup)
        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        integral = self.ki * self.integral

        # Derivative term
        # Note: We can use either error derivative or velocity estimate
        if self.first_step:
            derivative = np.zeros(2, dtype=np.float32)
            self.first_step = False
        else:
            if self.use_velocity_feedforward:
                # Use Kalman velocity estimate directly
                derivative = self.kd * estimated_velocity
            else:
                # Use numerical derivative of error
                derivative = self.kd * (error - self.previous_error) / self.dt

        # Update state
        self.previous_error = error

        # Compute total control signal
        action = proportional + integral + derivative

        # Clip action to valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        info["error"] = error
        info["action_components"] = {
            "proportional": proportional,
            "integral": integral,
            "derivative": derivative,
        }

        return action, info

    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics."""
        return {
            "detection_rate": 1.0 - (self.detections_failed / max(self.total_steps, 1)),
            "detections_failed": self.detections_failed,
            "total_steps": self.total_steps,
        }


class KalmanPIDAgent:
    """
    Wrapper class to make Kalman-PID controller compatible with agent interface.

    This allows easy comparison with RL agents by providing a consistent API.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Kalman-PID agent.

        Args:
            config: Configuration dictionary with 'pid', 'kalman', and 'environment' keys
        """
        pid_config = config.get("pid", {})
        kalman_config = config.get("kalman", {})
        env_config = config.get("environment", {})

        self.controller = KalmanPIDController(
            # PID parameters
            kp=pid_config.get("kp", 0.5),
            ki=pid_config.get("ki", 0.01),
            kd=pid_config.get("kd", 0.2),
            integral_limit=pid_config.get("integral_limit", 10.0),
            # Kalman parameters
            process_noise_pos=kalman_config.get("process_noise_pos", 0.1),
            process_noise_vel=kalman_config.get("process_noise_vel", 0.5),
            measurement_noise=kalman_config.get("measurement_noise", 2.0),
            initial_covariance=kalman_config.get("initial_covariance", 10.0),
            # Environment parameters
            dt=env_config.get("dt", 0.1),
            frame_size=env_config.get("frame_size", 64),
            view_radius=env_config.get("view_radius", 30.0),
            use_velocity_feedforward=True,
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[Any]]:
        """
        Predict action (compatible with Stable-Baselines3 interface).

        Args:
            observation: Environment observation
            deterministic: Not used (always deterministic)

        Returns:
            Tuple of (action, state) where state is None
        """
        action, info = self.controller.compute_action(observation)
        return action, None

    def reset(self):
        """Reset agent state."""
        self.controller.reset()

    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics."""
        return self.controller.get_stats()
