"""
Kalman Filter for state estimation in pursuit-evasion control.

This module implements a discrete-time Kalman Filter to estimate the
target's position and velocity from noisy visual measurements.
"""

import numpy as np
from typing import Optional, Tuple


class KalmanFilter:
    """
    Discrete-time Kalman Filter for 2D target tracking.

    State vector: x = [px, py, vx, vy]^T
        - px, py: Position in x and y
        - vx, vy: Velocity in x and y

    Measurement vector: z = [px, py]^T
        - Only position is directly measured (from visual detection)

    Dynamics model (constant velocity):
        x_{k+1} = F * x_k + w_k
        z_k = H * x_k + v_k

    where:
        - F is the state transition matrix
        - H is the measurement matrix
        - w_k ~ N(0, Q) is process noise
        - v_k ~ N(0, R) is measurement noise
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        measurement_noise: float = 2.0,
        initial_covariance: float = 10.0,
    ):
        """
        Initialize Kalman Filter.

        Args:
            dt: Time step
            process_noise_pos: Process noise for position
            process_noise_vel: Process noise for velocity
            measurement_noise: Measurement noise covariance
            initial_covariance: Initial state covariance
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        # x_{k+1} = F * x_k
        # [px']   [1  0  dt  0] [px]
        # [py'] = [0  1  0  dt] [py]
        # [vx']   [0  0  1   0] [vx]
        # [vy']   [0  0  0   1] [vy]
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Measurement matrix (observe position only)
        # z = H * x
        # [px_obs]   [1  0  0  0] [px]
        # [py_obs] = [0  1  0  0] [py]
        #                          [vx]
        #                          [vy]
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance matrix Q
        # Assumes acceleration noise affects velocity
        self.Q = np.array(
            [
                [process_noise_pos, 0, 0, 0],
                [0, process_noise_pos, 0, 0],
                [0, 0, process_noise_vel, 0],
                [0, 0, 0, process_noise_vel],
            ],
            dtype=np.float32,
        )

        # Measurement noise covariance matrix R
        self.R = np.array(
            [[measurement_noise, 0], [0, measurement_noise]], dtype=np.float32
        )

        # State estimate and covariance
        self.x = np.zeros(4, dtype=np.float32)  # [px, py, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * initial_covariance

        # Flag for initialization
        self.initialized = False

    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = False

    def initialize(self, measurement: np.ndarray):
        """
        Initialize filter with first measurement.

        Args:
            measurement: Initial position measurement [px, py]
        """
        self.x[0:2] = measurement  # Set position
        self.x[2:4] = 0.0  # Initialize velocity to zero
        self.initialized = True

    def predict(self) -> np.ndarray:
        """
        Prediction step (time update).

        Propagate state and covariance forward in time:
            x̂_k = F * x_{k-1}
            P_k = F * P_{k-1} * F^T + Q

        Returns:
            Predicted state vector [px, py, vx, vy]
        """
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step (measurement update).

        Correct prediction using new measurement:
            K = P * H^T * (H * P * H^T + R)^{-1}
            x̂ = x̂ + K * (z - H * x̂)
            P = (I - K * H) * P

        Args:
            measurement: Position measurement [px, py]

        Returns:
            Updated state estimate [px, py, vx, vy]
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
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    def step(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform one complete filter step (predict + update).

        Args:
            measurement: Position measurement [px, py], or None if no measurement

        Returns:
            Estimated state [px, py, vx, vy]
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
        """Get estimated position [px, py]."""
        return self.x[0:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Get estimated velocity [vx, vy]."""
        return self.x[2:4].copy()

    def get_state(self) -> np.ndarray:
        """Get full state estimate [px, py, vx, vy]."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter for nonlinear systems.

    This is a placeholder for future extension if nonlinear dynamics
    need to be modeled. Currently identical to linear Kalman Filter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Additional EKF-specific parameters can be added here
