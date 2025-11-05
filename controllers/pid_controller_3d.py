"""
3D PID Controller for vision-based pursuit-evasion control with depth.

This controller extends the classical PID to 3D space, handling lateral/vertical
steering (XY) and range management (Z) using depth estimation from visual size.
"""

import numpy as np
from typing import Optional, Dict, Any
from utils.visual_detection import VisualDetector


class PIDController3D:
    """
    3D PID (Proportional-Integral-Derivative) controller for 3D tracking.

    The controller computes 3D acceleration commands based on:
    - XY error: target position on the view plane
    - Z error: estimated depth/range from target size

    Control law:
        u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt

    where e(t) is the 3D error vector [ex, ey, ez].
    """

    def __init__(
        self,
        kp_xy: float = 0.5,
        ki_xy: float = 0.01,
        kd_xy: float = 0.2,
        kp_z: float = 0.3,
        ki_z: float = 0.005,
        kd_z: float = 0.15,
        integral_limit: float = 10.0,
        dt: float = 0.1,
        frame_size: int = 64,
        view_radius: float = 30.0,
        min_depth: float = 10.0,
        max_depth: float = 50.0,
        target_base_size: float = 2.0,
    ):
        """
        Initialize 3D PID controller.

        Args:
            kp_xy, ki_xy, kd_xy: PID gains for XY (lateral/vertical steering)
            kp_z, ki_z, kd_z: PID gains for Z (range/depth control)
            integral_limit: Anti-windup limit for integral term
            dt: Time step
            frame_size: Size of observation frame
            view_radius: View radius in world units
            min_depth, max_depth: Depth range for estimation
            target_base_size: Base size of target at min_depth
        """
        self.kp_xy = kp_xy
        self.ki_xy = ki_xy
        self.kd_xy = kd_xy
        self.kp_z = kp_z
        self.ki_z = ki_z
        self.kd_z = kd_z
        self.integral_limit = integral_limit
        self.dt = dt

        # Visual detector
        self.detector = VisualDetector(frame_size=frame_size)

        # Conversion factor from pixels to world units
        self.pixel_to_world = (2 * view_radius) / frame_size

        # Depth estimation parameters
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_base_size = target_base_size
        self.desired_depth = (min_depth + max_depth) / 2  # Target engagement range

        # State variables (3D)
        self.integral = np.zeros(3, dtype=np.float32)  # [ix, iy, iz]
        self.previous_error = np.zeros(3, dtype=np.float32)  # [ex, ey, ez]
        self.first_step = True

        # Statistics
        self.detections_failed = 0
        self.total_steps = 0

    def reset(self):
        """Reset controller state."""
        self.integral = np.zeros(3, dtype=np.float32)
        self.previous_error = np.zeros(3, dtype=np.float32)
        self.first_step = True
        self.detections_failed = 0
        self.total_steps = 0

    def estimate_depth_from_size(self, target_radius_pixels: float) -> float:
        """
        Estimate depth from target size using inverse relationship.

        apparent_size = base_size * (min_depth / depth)
        => depth = base_size * min_depth / apparent_size

        Args:
            target_radius_pixels: Radius of detected target in pixels

        Returns:
            Estimated depth in world units
        """
        # Convert pixel radius to world units
        apparent_size = target_radius_pixels * self.pixel_to_world

        # Estimate depth (with safety bounds)
        if apparent_size > 0.1:
            estimated_depth = (self.target_base_size * self.min_depth) / apparent_size
            # Clamp to valid range
            estimated_depth = np.clip(estimated_depth, self.min_depth, self.max_depth)
        else:
            # If target too small, assume max depth
            estimated_depth = self.max_depth

        return estimated_depth

    def compute_action(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute 3D control action from observation.

        Args:
            observation: Stacked frames (num_frames, H, W) or (H, W, num_frames)

        Returns:
            Tuple of (action, info):
                - action: 3D acceleration command [ax, ay, az] in [-1, 1]^3
                - info: Dictionary with debug information
        """
        self.total_steps += 1

        # Detect target position (in pixels relative to center) and size
        position_pixels, confidence = self.detector.detect_target_stacked(observation)

        info = {
            "detection_success": position_pixels is not None,
            "confidence": confidence,
        }

        # Handle detection failure
        if position_pixels is None:
            self.detections_failed += 1
            action = np.zeros(3, dtype=np.float32)
            info["error_3d"] = self.previous_error
            info["estimated_depth"] = self.max_depth
            return action, info

        # Convert pixel position to world units (XY error)
        error_xy = position_pixels * self.pixel_to_world

        # Estimate depth from target size (Z error)
        # confidence from detector is actually the area, related to radius
        target_radius_pixels = np.sqrt(confidence / np.pi) if confidence > 0 else 1.0
        estimated_depth = self.estimate_depth_from_size(target_radius_pixels)

        # Z error: difference from desired engagement range
        error_z = estimated_depth - self.desired_depth

        # Combine into 3D error vector
        error = np.array([error_xy[0], error_xy[1], error_z], dtype=np.float32)

        # Proportional term (separate gains for XY and Z)
        proportional = np.array([
            self.kp_xy * error[0],
            self.kp_xy * error[1],
            self.kp_z * error[2]
        ], dtype=np.float32)

        # Integral term (with anti-windup, separate for XY and Z)
        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        integral = np.array([
            self.ki_xy * self.integral[0],
            self.ki_xy * self.integral[1],
            self.ki_z * self.integral[2]
        ], dtype=np.float32)

        # Derivative term
        if self.first_step:
            derivative = np.zeros(3, dtype=np.float32)
            self.first_step = False
        else:
            derivative = np.array([
                self.kd_xy * (error[0] - self.previous_error[0]) / self.dt,
                self.kd_xy * (error[1] - self.previous_error[1]) / self.dt,
                self.kd_z * (error[2] - self.previous_error[2]) / self.dt
            ], dtype=np.float32)

        # Update state
        self.previous_error = error

        # Compute total control signal
        action = proportional + integral + derivative

        # Clip action to valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        info["error_3d"] = error
        info["estimated_depth"] = estimated_depth
        info["desired_depth"] = self.desired_depth
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


class PIDAgent3D:
    """
    Wrapper class to make 3D PID controller compatible with agent interface.

    This allows easy comparison with RL agents by providing a consistent API.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize 3D PID agent.

        Args:
            config: Configuration dictionary with 'pid' and 'environment' keys
        """
        pid_config = config.get("pid", {})
        env_config = config.get("environment", {})

        # Get depth range
        depth_range = env_config.get("depth_range", [10.0, 50.0])

        self.controller = PIDController3D(
            kp_xy=pid_config.get("kp", 0.5),
            ki_xy=pid_config.get("ki", 0.01),
            kd_xy=pid_config.get("kd", 0.2),
            kp_z=pid_config.get("kp_z", 0.3),  # Can be different for Z
            ki_z=pid_config.get("ki_z", 0.005),
            kd_z=pid_config.get("kd_z", 0.15),
            integral_limit=pid_config.get("integral_limit", 10.0),
            dt=env_config.get("dt", 0.1),
            frame_size=env_config.get("frame_size", 64),
            view_radius=env_config.get("view_radius", 30.0),
            min_depth=depth_range[0],
            max_depth=depth_range[1],
            target_base_size=env_config.get("target_size", 2.0),
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
