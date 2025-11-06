"""
RL Agent for Target (Evader)

This module provides a wrapper for using RL agents (like SAC) to control
the target in competitive scenarios.

The target tries to:
- Escape from the agent's FOV
- Maximize time outside the focus area
- Avoid being tracked
"""

import numpy as np
from typing import Tuple, Optional
from stable_baselines3 import SAC


class TargetRLAgent:
    """
    RL agent wrapper for target control.

    This class wraps a trained SAC model to control the target's behavior.
    The target receives its own observation (position, velocity, agent info)
    and outputs acceleration commands to evade the agent.
    """

    def __init__(self, model_path: Optional[str] = None, model: Optional[SAC] = None):
        """
        Initialize target RL agent.

        Args:
            model_path: Path to trained SAC model (for loading)
            model: Pre-trained SAC model instance
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = SAC.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")

    def predict(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action for target given current state.

        For RL agent, we create a simple state representation:
        - position: [x, y, z] relative to agent (agent at origin)
        - velocity: [vx, vy, vz] target's current velocity

        Note: In full implementation, target would get visual observation.
        For now, we use state-based input for simplicity.

        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            deterministic: Whether to use deterministic policy

        Returns:
            action: Acceleration command [ax, ay, az] (normalized to [-1, 1])
            state: None (stateless for compatibility)
        """
        # Create state vector: [position, velocity]
        # This is a simplified state representation
        # In full implementation, we'd use visual observations like agent
        state = np.concatenate([position, velocity])

        # Get action from model
        action, _ = self.model.predict(state, deterministic=deterministic)

        return action, None

    def reset(self):
        """Reset agent state (if needed)."""
        pass


class TargetRLAgentVision:
    """
    RL agent wrapper for target with visual observations.

    This version uses the same visual observation approach as the pursuer agent.
    Target sees the world from its own egocentric perspective.
    """

    def __init__(self, model_path: Optional[str] = None, model: Optional[SAC] = None):
        """
        Initialize vision-based target RL agent.

        Args:
            model_path: Path to trained SAC model
            model: Pre-trained SAC model instance
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = SAC.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")

        self.last_observation = None

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action from visual observation.

        Args:
            observation: Visual observation (frame stack)
            deterministic: Whether to use deterministic policy

        Returns:
            action: Acceleration command
            state: None
        """
        self.last_observation = observation
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, None

    # For compatibility with environment controller interface
    def predict_from_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predict from state (fallback when observation not available).

        Note: This uses the last visual observation or creates a dummy one.
        """
        if self.last_observation is None:
            # Create dummy observation (will be replaced by actual observation)
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from environments import make_env
            import yaml

            config = yaml.safe_load(open("configs/config.yaml"))
            frame_size = config.get("environment", {}).get("frame_size", 64)
            frame_stack = config.get("environment", {}).get("frame_stack", 4)

            dummy_obs = np.zeros((frame_stack, frame_size, frame_size), dtype=np.uint8)
            action, _ = self.model.predict(dummy_obs, deterministic=deterministic)
            return action, None

        # Use last observation
        action, _ = self.model.predict(self.last_observation, deterministic=deterministic)
        return action, None

    def reset(self):
        """Reset agent state."""
        self.last_observation = None
