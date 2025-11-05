"""
3D Kalman Filter + PID Controller for robust vision-based 3D tracking.

This controller combines Kalman filtering for 3D state estimation with
PID control for smooth and robust 3D tracking with range management.
"""

import numpy as np
from typing import Dict, Any, Optional
from utils.visual_detection import VisualDetector


class KalmanFilter3D:
    """
    Discrete-time Kalman Filter for 3D target tracking.

    State vector: x = [px, py, pz, vx, vy, vz]^T
        - px, py, pz: Position in 3D
        - vx, vy, vz: Velocity in 3D

    Measurement vector: z = [px, py, pz]^T
        - Position (XY from visual detection, Z from size estimation)

    Dynamics model (constant velocity):
        x_{k+1} = F * x_k + w_k
        z_k = H * x_k + v_k
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        measurement_noise_xy: float = 2.0,
        measurement_noise_z: float = 5.0,  # Depth estimation is noisier
        initial_covariance: float = 10.0,
    ):
        """
        Initialize 3D Kalman Filter.

        Args:
            dt: Time step
            process_noise_pos: Process noise for position
            process_noise_vel: Process noise for velocity
            measurement_noise_xy: Measurement noise for XY
            measurement_noise_z: Measurement noise for Z (depth)
            initial_covariance: Initial state covariance
        """
        self.dt = dt

        # State transition matrix (constant velocity model in 3D)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float32)

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance matrix Q (6x6)
        self.Q = np.diag([
            process_noise_pos, process_noise_pos, process_noise_pos,
            process_noise_vel, process_noise_vel, process_noise_vel
        ]).astype(np.float32)

        # Measurement noise covariance matrix R (3x3)
        self.R = np.diag([
            measurement_noise_xy,
            measurement_noise_xy,
            measurement_noise_z  # Depth estimation is noisier
        ]).astype(np.float32)

        # State estimate and covariance
        self.x = np.zeros(6, dtype=np.float32)  # [px, py, pz, vx, vy, vz]
        self.P = np.eye(6, dtype=np.float32) * initial_covariance

        # Flag for initialization
        self.initialized = False

    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 10.0
        self.initialized = False

    def initialize(self, measurement: np.ndarray):
        """
        Initialize filter with first measurement.

        Args:
            measurement: Initial 3D position measurement [px, py, pz]
        """
        self.x[0:3] = measurement  # Set position
        self.x[3:6] = 0.0  # Initialize velocity to zero
        self.initialized = True

    def predict(self) -> np.ndarray:
        """
        Prediction step (time update).

        Returns:
            Predicted state vector [px, py, pz, vx, vy, vz]
        """
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step (measurement update).

        Args:
            measurement: 3D position measurement [px, py, pz]

        Returns:
            Updated state estimate [px, py, pz, vx, vy, vz]
        """
        # Innovation (measurement residual)
        y = measurement - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    def step(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform one complete filter step (predict + update).

        Args:
            measurement: 3D position measurement [px, py, pz], or None

        Returns:
            Estimated state [px, py, pz, vx, vy, vz]
        """
        if not self.initialized:
            if measurement is not None:
                self.initialize(measurement)
            return self.x.copy()

        # Always predict
        self.predict()

        # Update only if measurement is available
        if measurement is not None:
            self.update(measurement)

        return self.x.copy()

    def get_position(self) -> np.ndarray:
        """Get estimated 3D position [px, py, pz]."""
        return self.x[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get estimated 3D velocity [vx, vy, vz]."""
        return self.x[3:6].copy()


class KalmanPIDController3D:
    """
    Combined 3D Kalman Filter + PID controller.

    Pipeline:
        1. Visual detection extracts XY position and estimates Z from size
        2. Kalman Filter estimates smooth 3D position and velocity
        3. PID controller generates 3D acceleration commands
    """

    def __init__(
        self,
        # PID parameters
        kp_xy: float = 0.5,
        ki_xy: float = 0.01,
        kd_xy: float = 0.2,
        kp_z: float = 0.3,
        ki_z: float = 0.005,
        kd_z: float = 0.15,
        integral_limit: float = 10.0,
        # Kalman Filter parameters
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        measurement_noise_xy: float = 2.0,
        measurement_noise_z: float = 5.0,
        initial_covariance: float = 10.0,
        # Environment parameters
        dt: float = 0.1,
        frame_size: int = 64,
        view_radius: float = 30.0,
        min_depth: float = 10.0,
        max_depth: float = 50.0,
        target_base_size: float = 2.0,
        # Additional options
        use_velocity_feedforward: bool = True,
    ):
        """Initialize 3D Kalman-PID controller."""
        self.kp_xy = kp_xy
        self.ki_xy = ki_xy
        self.kd_xy = kd_xy
        self.kp_z = kp_z
        self.ki_z = ki_z
        self.kd_z = kd_z
        self.integral_limit = integral_limit
        self.dt = dt
        self.use_velocity_feedforward = use_velocity_feedforward

        # Visual detector
        self.detector = VisualDetector(frame_size=frame_size)

        # 3D Kalman Filter
        self.kalman = KalmanFilter3D(
            dt=dt,
            process_noise_pos=process_noise_pos,
            process_noise_vel=process_noise_vel,
            measurement_noise_xy=measurement_noise_xy,
            measurement_noise_z=measurement_noise_z,
            initial_covariance=initial_covariance,
        )

        # Conversion factor from pixels to world units
        self.pixel_to_world = (2 * view_radius) / frame_size

        # Depth estimation parameters
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_base_size = target_base_size
        self.desired_depth = (min_depth + max_depth) / 2

        # PID state (3D)
        self.integral = np.zeros(3, dtype=np.float32)
        self.previous_error = np.zeros(3, dtype=np.float32)
        self.first_step = True

        # Statistics
        self.detections_failed = 0
        self.total_steps = 0

    def reset(self):
        """Reset controller state."""
        self.kalman.reset()
        self.integral = np.zeros(3, dtype=np.float32)
        self.previous_error = np.zeros(3, dtype=np.float32)
        self.first_step = True
        self.detections_failed = 0
        self.total_steps = 0

    def estimate_depth_from_size(self, target_radius_pixels: float) -> float:
        """Estimate depth from target size."""
        apparent_size = target_radius_pixels * self.pixel_to_world
        if apparent_size > 0.1:
            estimated_depth = (self.target_base_size * self.min_depth) / apparent_size
            estimated_depth = np.clip(estimated_depth, self.min_depth, self.max_depth)
        else:
            estimated_depth = self.max_depth
        return estimated_depth

    def compute_action(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute 3D control action from observation.

        Args:
            observation: Stacked frames

        Returns:
            Tuple of (action, info)
        """
        self.total_steps += 1

        # Detect target
        position_pixels, confidence = self.detector.detect_target_stacked(observation)

        info = {
            "detection_success": position_pixels is not None,
            "confidence": confidence,
        }

        # Prepare measurement for Kalman Filter
        if position_pixels is not None:
            # XY from detection
            measurement_xy = position_pixels * self.pixel_to_world

            # Z from size estimation
            target_radius_pixels = np.sqrt(confidence / np.pi) if confidence > 0 else 1.0
            estimated_depth = self.estimate_depth_from_size(target_radius_pixels)

            # Combine into 3D measurement
            measurement = np.array([
                measurement_xy[0],
                measurement_xy[1],
                estimated_depth
            ], dtype=np.float32)
        else:
            measurement = None
            self.detections_failed += 1

        # Update Kalman Filter
        state = self.kalman.step(measurement)
        estimated_position = state[0:3]  # [px, py, pz]
        estimated_velocity = state[3:6]  # [vx, vy, vz]

        info["estimated_position"] = estimated_position
        info["estimated_velocity"] = estimated_velocity

        # Compute error (3D)
        error_xy = estimated_position[0:2]
        error_z = estimated_position[2] - self.desired_depth
        error = np.array([error_xy[0], error_xy[1], error_z], dtype=np.float32)

        # PID Control
        proportional = np.array([
            self.kp_xy * error[0],
            self.kp_xy * error[1],
            self.kp_z * error[2]
        ], dtype=np.float32)

        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        integral = np.array([
            self.ki_xy * self.integral[0],
            self.ki_xy * self.integral[1],
            self.ki_z * self.integral[2]
        ], dtype=np.float32)

        if self.first_step:
            derivative = np.zeros(3, dtype=np.float32)
            self.first_step = False
        else:
            if self.use_velocity_feedforward:
                derivative = np.array([
                    self.kd_xy * estimated_velocity[0],
                    self.kd_xy * estimated_velocity[1],
                    self.kd_z * estimated_velocity[2]
                ], dtype=np.float32)
            else:
                derivative = np.array([
                    self.kd_xy * (error[0] - self.previous_error[0]) / self.dt,
                    self.kd_xy * (error[1] - self.previous_error[1]) / self.dt,
                    self.kd_z * (error[2] - self.previous_error[2]) / self.dt
                ], dtype=np.float32)

        self.previous_error = error

        # Compute total control signal
        action = proportional + integral + derivative
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        info["error_3d"] = error
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


class KalmanPIDAgent3D:
    """Wrapper for 3D Kalman-PID controller with agent interface."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize 3D Kalman-PID agent."""
        pid_config = config.get("pid", {})
        kalman_config = config.get("kalman", {})
        env_config = config.get("environment", {})

        depth_range = env_config.get("depth_range", [10.0, 50.0])

        self.controller = KalmanPIDController3D(
            # PID parameters
            kp_xy=pid_config.get("kp", 0.5),
            ki_xy=pid_config.get("ki", 0.01),
            kd_xy=pid_config.get("kd", 0.2),
            kp_z=pid_config.get("kp_z", 0.3),
            ki_z=pid_config.get("ki_z", 0.005),
            kd_z=pid_config.get("kd_z", 0.15),
            integral_limit=pid_config.get("integral_limit", 10.0),
            # Kalman parameters
            process_noise_pos=kalman_config.get("process_noise_pos", 0.1),
            process_noise_vel=kalman_config.get("process_noise_vel", 0.5),
            measurement_noise_xy=kalman_config.get("measurement_noise", 2.0),
            measurement_noise_z=kalman_config.get("measurement_noise_z", 5.0),
            initial_covariance=kalman_config.get("initial_covariance", 10.0),
            # Environment parameters
            dt=env_config.get("dt", 0.1),
            frame_size=env_config.get("frame_size", 64),
            view_radius=env_config.get("view_radius", 30.0),
            min_depth=depth_range[0],
            max_depth=depth_range[1],
            target_base_size=env_config.get("target_size", 2.0),
            use_velocity_feedforward=True,
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[Any]]:
        """Predict action."""
        action, info = self.controller.compute_action(observation)
        return action, None

    def reset(self):
        """Reset agent state."""
        self.controller.reset()

    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics."""
        return self.controller.get_stats()
