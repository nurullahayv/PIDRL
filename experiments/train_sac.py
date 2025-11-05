"""
Training script for SAC agent on pursuit-evasion task.

This script trains a Soft Actor-Critic (SAC) agent to learn vision-based
pursuit control directly from pixel observations.
"""

import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

from environments import make_env


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Stable-Baselines3 SAC.

    Processes stacked frame observations through a CNN to extract features.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Get observation shape
        obs_shape = observation_space.shape

        # Determine number of input channels
        if len(obs_shape) == 3:
            n_input_channels = obs_shape[0] if obs_shape[0] <= 4 else obs_shape[2]
        else:
            n_input_channels = 1

        # CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            if sample_input.dtype == torch.uint8:
                sample_input = sample_input.float() / 255.0
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0

        features = self.cnn(observations)
        return self.linear(features)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_sac_env(config: Dict[str, Any], seed: int = 0, render_mode=None):
    """Create environment for SAC training."""
    env = make_env(config, render_mode=render_mode)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_sac(
    config: Dict[str, Any],
    save_dir: str = "models/sac",
    tensorboard_log: str = "logs/sac",
    seed: int = 42,
):
    """
    Train SAC agent on pursuit-evasion task.

    Args:
        config: Configuration dictionary
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
        seed: Random seed
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get configuration
    sac_config = config.get("sac", {})
    experiment_config = config.get("experiment", {})

    # Create training environment
    print("Creating training environment...")
    train_env = make_sac_env(config, seed=seed)

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_sac_env(config, seed=seed + 100)

    # Policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )

    # Create SAC agent
    print("Initializing SAC agent...")
    model = SAC(
        "CnnPolicy",
        train_env,
        learning_rate=sac_config.get("learning_rate", 3e-4),
        buffer_size=sac_config.get("buffer_size", 100000),
        batch_size=sac_config.get("batch_size", 256),
        gamma=sac_config.get("gamma", 0.99),
        tau=sac_config.get("tau", 0.005),
        ent_coef=sac_config.get("alpha", 0.2),
        learning_starts=sac_config.get("learning_starts", 10000),
        train_freq=sac_config.get("train_freq", 1),
        gradient_steps=sac_config.get("gradient_steps", 1),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
    )

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=sac_config.get("eval_freq", 10000),
        n_eval_episodes=sac_config.get("n_eval_episodes", 10),
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="sac_model",
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # Train the agent
    print(f"Training SAC for {sac_config.get('total_timesteps', 500000)} timesteps...")
    model.learn(
        total_timesteps=sac_config.get("total_timesteps", 500000),
        callback=callback_list,
        log_interval=experiment_config.get("log_interval", 1000),
    )

    # Save final model
    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Clean up
    train_env.close()
    eval_env.close()

    return model


def main():
    """Main training function."""
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Train SAC agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--save-dir", type=str, default="models/sac", help="Model save directory"
    )
    parser.add_argument(
        "--tensorboard-log", type=str, default="logs/sac", help="Tensorboard log directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Train agent
    train_sac(
        config=config,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
