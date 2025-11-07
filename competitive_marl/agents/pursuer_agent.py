"""
Pursuer Agent for Competitive MARL

Wrapper around Stable-Baselines3 SAC agent.
"""

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Optional, Dict, Any
import os


class PursuerAgent:
    """
    Pursuer (agent) wrapper for competitive training.

    Uses SAC (Soft Actor-Critic) for continuous control.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str = "auto",
        target_update_interval: int = 1,
        net_arch: list = [256, 256],
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize pursuer agent.

        Args:
            env: Gymnasium environment
            ... (SAC hyperparameters)
        """
        self.env = env

        # Create SAC model
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
    ):
        """
        Train the pursuer agent.

        Args:
            total_timesteps: Number of steps to train
            callback: Optional callback for logging/checkpointing
            log_interval: Log interval
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            reset_num_timesteps=False,  # Continue training
        )

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict action from observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            Action vector
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Pursuer model saved to: {path}")

    def load(self, path: str):
        """Load model from file."""
        if os.path.exists(path):
            self.model = SAC.load(path, env=self.env)
            print(f"Pursuer model loaded from: {path}")
        else:
            print(f"Warning: Model file not found: {path}")

    @classmethod
    def load_from_file(cls, path: str, env):
        """
        Load existing model from file.

        Args:
            path: Path to saved model
            env: Environment

        Returns:
            PursuerAgent with loaded model
        """
        agent = cls(env, verbose=0)
        agent.load(path)
        return agent

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "num_timesteps": self.model.num_timesteps,
            "learning_rate": self.model.learning_rate,
        }
