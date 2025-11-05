"""
Classical PID Controller for vision-based pursuit-evasion control.

This controller uses OpenCV to detect the target position and applies
a PID control law to generate acceleration commands.
"""

import numpy as np
from typing import Optional, Dict, Any
from utils.visual_detection import VisualDetector


class PIDController:
    """
    PID (Proportional-Integral-Derivative) controller for 2D tracking.

    The controller computes acceleration commands based on the position error
    between the target and the frame center.

    Control law:
        u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt

    where e(t) is the error signal (target position relative to center).
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.2,
        integral_limit: float = 10.0,
        dt: float = 0.1,
        frame_size: int = 64,
        view_radius: float = 30.0,
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Anti-windup limit for integral term
            dt: Time step
            frame_size: Size of observation frame
            view_radius: View radius in world units
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.dt = dt

        # Visual detector
        self.detector = VisualDetector(frame_size=frame_size)

        # Conversion factor from pixels to world units
        self.pixel_to_world = (2 * view_radius) / frame_size

        # State variables
        self.integral = np.zeros(2, dtype=np.float32)
        self.previous_error = np.zeros(2, dtype=np.float32)
        self.first_step = True

        # Statistics
        self.detections_failed = 0
        self.total_steps = 0

    def reset(self):
        """Reset controller state."""
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

        # Handle detection failure
        if position_pixels is None:
            self.detections_failed += 1
            # Return zero action or continue with last known error
            action = np.zeros(2, dtype=np.float32)
            info["error"] = self.previous_error
            info["action_components"] = {
                "proportional": np.zeros(2),
                "integral": self.integral,
                "derivative": np.zeros(2),
            }
            return action, info

        # Convert pixel position to world units
        # Error is negative because we want to move toward the target
        error = position_pixels * self.pixel_to_world

        # Proportional term
        proportional = self.kp * error

        # Integral term (with anti-windup)
        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        integral = self.ki * self.integral

        # Derivative term
        if self.first_step:
            derivative = np.zeros(2, dtype=np.float32)
            self.first_step = False
        else:
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


class PIDAgent:
    """
    Wrapper class to make PID controller compatible with agent interface.

    This allows easy comparison with RL agents by providing a consistent API.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PID agent.

        Args:
            config: Configuration dictionary with 'pid' and 'environment' keys
        """
        pid_config = config.get("pid", {})
        env_config = config.get("environment", {})

        self.controller = PIDController(
            kp=pid_config.get("kp", 0.5),
            ki=pid_config.get("ki", 0.01),
            kd=pid_config.get("kd", 0.2),
            integral_limit=pid_config.get("integral_limit", 10.0),
            dt=env_config.get("dt", 0.1),
            frame_size=env_config.get("frame_size", 64),
            view_radius=env_config.get("view_radius", 30.0),
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[Any]]:
        """
        Predict action (compatible with Stable-Baselines3 interface).

        Args:
            observation: Environment observation
            deterministic: Not used for PID (always deterministic)

        Returns:
            Tuple of (action, state) where state is None for PID
        """
        action, info = self.controller.compute_action(observation)
        return action, None

    def reset(self):
        """Reset agent state."""
        self.controller.reset()

    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics."""
        return self.controller.get_stats()
