#!/usr/bin/env python3
"""
Competitive MARL Training: Agent (Pursuer) vs Target (Evader)

Phase 2 training script where:
- Agent: Uses SAC to learn pursuit
- Target: Uses SAC to learn evasion

Training modes:
1. Fixed agent (PID) vs Learning target
2. Learning agent vs Fixed target (PID evader)
3. Self-play: Both learning simultaneously

Usage:
    python experiments/train_competitive.py --mode target  # Train target vs PID agent
    python experiments/train_competitive.py --mode agent   # Train agent vs PID evader
    python experiments/train_competitive.py --mode both    # Self-play (both learn)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from environments import make_env
from controllers import PIDAgent3D, TargetEvaderAgent


def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_competitive(
    config: dict,
    mode: str = "target",
    total_timesteps: int = 100000,
    save_dir: str = "models/competitive",
    tensorboard_log: str = "logs/competitive",
    seed: int = 42,
):
    """
    Train competitive MARL agents.

    Args:
        config: Configuration dictionary
        mode: Training mode ('target', 'agent', or 'both')
        total_timesteps: Total training steps
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
        seed: Random seed
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 70)
    print(f"COMPETITIVE MARL TRAINING - Phase 2")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print()

    if mode == "target":
        print("🎯 Training TARGET (evader) against PID agent (pursuer)")
        print("   Agent: PID controller (fixed)")
        print("   Target: SAC RL (learning to evade)")
        print()

        # Create fixed PID agent as opponent
        agent_controller = PIDAgent3D(config)

        # Target will learn with RL - but we need custom environment
        # For now, use standard environment (target gets negative agent rewards)
        target_controller = None  # Will be the RL agent we're training

        env = make_env(config, render_mode=None, target_controller=target_controller)

        # Train SAC for target
        # Note: This is simplified - in full version, we'd create custom env
        # where target is the learner and gets observations
        print("⚠️  Note: Simplified training - using agent perspective")
        print("   Full implementation would train from target's perspective")
        print()

        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=config["sac"]["learning_rate"],
            buffer_size=config["sac"]["buffer_size"],
            batch_size=config["sac"]["batch_size"],
            gamma=config["sac"]["gamma"],
            tau=config["sac"]["tau"],
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{save_dir}/target",
            name_prefix="target_sac",
        )

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="target_vs_pid_agent",
        )

        # Save final model
        model.save(f"{save_dir}/target/final_model")
        print(f"\n✓ Target model saved to {save_dir}/target/final_model")

    elif mode == "agent":
        print("🎯 Training AGENT (pursuer) against PID evader (target)")
        print("   Agent: SAC RL (learning to pursue)")
        print("   Target: PID evader (fixed)")
        print()

        # Create fixed PID evader as opponent
        target_controller = TargetEvaderAgent(config)

        env = make_env(config, render_mode=None, target_controller=target_controller)

        # Train SAC for agent (standard pursuit training)
        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=config["sac"]["learning_rate"],
            buffer_size=config["sac"]["buffer_size"],
            batch_size=config["sac"]["batch_size"],
            gamma=config["sac"]["gamma"],
            tau=config["sac"]["tau"],
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{save_dir}/agent",
            name_prefix="agent_sac",
        )

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="agent_vs_pid_evader",
        )

        # Save final model
        model.save(f"{save_dir}/agent/final_model")
        print(f"\n✓ Agent model saved to {save_dir}/agent/final_model")

    elif mode == "both":
        print("🎯 Self-Play: Both AGENT and TARGET learning")
        print("   Agent: SAC RL (learning to pursue)")
        print("   Target: SAC RL (learning to evade)")
        print()
        print("⚠️  Note: Full self-play requires alternating training")
        print("   Implementing simplified version with fixed target initially")
        print()

        # For self-play, we need to alternate training
        # This is a simplified version - train agent first against PID evader
        target_controller = TargetEvaderAgent(config)
        env = make_env(config, render_mode=None, target_controller=target_controller)

        agent_model = SAC(
            "CnnPolicy",
            env,
            learning_rate=config["sac"]["learning_rate"],
            buffer_size=config["sac"]["buffer_size"],
            batch_size=config["sac"]["batch_size"],
            gamma=config["sac"]["gamma"],
            tau=config["sac"]["tau"],
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{save_dir}/self_play",
            name_prefix="agent_sac",
        )

        agent_model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="self_play",
        )

        agent_model.save(f"{save_dir}/self_play/agent_final")
        print(f"\n✓ Agent model saved to {save_dir}/self_play/agent_final")

        # TODO: Implement full self-play with alternating training
        print("\n📝 TODO: Implement full self-play iteration")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'target', 'agent', or 'both'")

    env.close()
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Competitive MARL Training (Phase 2)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="agent",
        choices=["target", "agent", "both"],
        help="Training mode: 'target' (train target vs PID agent), 'agent' (train agent vs PID evader), 'both' (self-play)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/competitive",
        help="Directory to save models",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/competitive",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    train_competitive(
        config=config,
        mode=args.mode,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
