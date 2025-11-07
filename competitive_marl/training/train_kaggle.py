"""
Kaggle Training Script for Competitive MARL

Optimized for Kaggle GPU environment:
- NO rendering (faster training)
- Alternating training (pursuer → evader → pursuer → ...)
- Automatic checkpointing
- Minimal verbose output

Usage on Kaggle:
    python train_kaggle.py --rounds 50 --steps-per-round 10000
"""

import sys
import os
import argparse
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.pursuit_evasion_3d import CompetitivePursuitEvasion3D, get_target_observation
from agents.pursuer_agent import PursuerAgent
from agents.evader_agent import EvaderAgent
from config import get_config


class CompetitiveTrainer:
    """
    Competitive MARL trainer with alternating training.

    Pursuer and evader take turns improving against each other.
    """

    def __init__(self, config: dict):
        self.config = config
        self.env_config = config["env"]
        self.training_config = config["training"]
        self.paths = config["paths"]

        # Create environment (NO RENDERING for Kaggle)
        print("Creating environment...")
        self.env = CompetitivePursuitEvasion3D(
            view_size=self.env_config["view_size"],
            frame_size=self.env_config["frame_size"],
            depth_range=self.env_config["depth_range"],
            success_threshold=self.env_config["success_threshold"],
            target_size=self.env_config["target_size"],
            max_velocity=self.env_config["max_velocity"],
            max_acceleration=self.env_config["max_acceleration"],
            max_angular_velocity=self.env_config["max_angular_velocity"],
            dt=self.env_config["dt"],
            max_steps=self.env_config["max_steps"],
            focus_reward=self.env_config["focus_reward"],
            focus_bonus=self.env_config["focus_bonus"],
            escape_penalty=self.env_config["escape_penalty"],
            outside_penalty_scale=self.env_config["outside_penalty_scale"],
            focus_time_threshold=self.env_config["focus_time_threshold"],
            render_mode=None,  # NO RENDERING
        )

        # Create agents
        print("Creating agents...")
        self.pursuer = PursuerAgent(
            self.env,
            **config["pursuer"],
            tensorboard_log=self.paths["tensorboard_log"],
            verbose=1,
        )

        self.evader = EvaderAgent(
            self.env,
            **config["evader"],
            tensorboard_log=self.paths["tensorboard_log"],
            verbose=1,
        )

        # Training statistics
        self.round_stats = []

    def train_pursuer_round(self, steps: int, round_num: int):
        """
        Train pursuer for one round against current evader.

        Args:
            steps: Number of training steps
            round_num: Current round number
        """
        print(f"\n{'='*70}")
        print(f"Round {round_num}: Training PURSUER ({steps} steps)")
        print(f"{'='*70}")

        # Reset environment
        obs, info = self.env.reset()
        episode_rewards = []
        episode_reward = 0
        episode_count = 0

        start_time = time.time()

        for step in range(steps):
            # Pursuer takes action
            pursuer_action = self.pursuer.predict(obs, deterministic=False)

            # Evader takes action (using current policy)
            evader_obs = get_target_observation(self.env)
            evader_action = self.evader.predict(evader_obs, deterministic=False)
            self.env.set_target_action(evader_action)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(pursuer_action)

            # Update episode stats
            episode_reward += reward

            # Reset if episode ended
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_count += 1
                episode_reward = 0
                obs, info = self.env.reset()
            else:
                obs = next_obs

        elapsed_time = time.time() - start_time

        # Statistics
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        std_reward = np.std(episode_rewards) if episode_rewards else 0

        print(f"\nPursuer Training Complete:")
        print(f"  Episodes: {episode_count}")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Time: {elapsed_time:.1f}s")

        return {
            "agent": "pursuer",
            "round": round_num,
            "episodes": episode_count,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "time": elapsed_time,
        }

    def train_evader_round(self, steps: int, round_num: int):
        """
        Train evader for one round against current pursuer.

        Args:
            steps: Number of training steps
            round_num: Current round number
        """
        print(f"\n{'='*70}")
        print(f"Round {round_num}: Training EVADER ({steps} steps)")
        print(f"{'='*70}")

        # Reset environment
        obs, info = self.env.reset()
        episode_rewards = []
        episode_reward = 0
        episode_count = 0

        start_time = time.time()

        for step in range(steps):
            # Pursuer takes action (using current policy)
            pursuer_action = self.pursuer.predict(obs, deterministic=False)

            # Evader takes action
            evader_obs = get_target_observation(self.env)
            evader_action = self.evader.predict(evader_obs, deterministic=False)
            self.env.set_target_action(evader_action)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(pursuer_action)

            # Track evader reward (opposite of pursuer)
            episode_reward += info["target_reward"]

            # Reset if episode ended
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_count += 1
                episode_reward = 0
                obs, info = self.env.reset()
            else:
                obs = next_obs

        elapsed_time = time.time() - start_time

        # Statistics
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        std_reward = np.std(episode_rewards) if episode_rewards else 0

        print(f"\nEvader Training Complete:")
        print(f"  Episodes: {episode_count}")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Time: {elapsed_time:.1f}s")

        return {
            "agent": "evader",
            "round": round_num,
            "episodes": episode_count,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "time": elapsed_time,
        }

    def train_alternating(self, num_rounds: int, steps_per_round: int):
        """
        Alternating training: pursuer → evader → pursuer → ...

        Args:
            num_rounds: Number of rounds
            steps_per_round: Steps per round for each agent
        """
        print("\n" + "=" * 70)
        print("COMPETITIVE MARL TRAINING - ALTERNATING MODE")
        print("=" * 70)
        print(f"Rounds: {num_rounds}")
        print(f"Steps per round: {steps_per_round}")
        print()

        for round_num in range(1, num_rounds + 1):
            # Train pursuer
            pursuer_stats = self.train_pursuer_round(steps_per_round, round_num)
            self.round_stats.append(pursuer_stats)

            # Save pursuer checkpoint
            self.pursuer.save(self.paths["pursuer_model"])

            # Train evader
            evader_stats = self.train_evader_round(steps_per_round, round_num)
            self.round_stats.append(evader_stats)

            # Save evader checkpoint
            self.evader.save(self.paths["evader_model"])

            # Print round summary
            print(f"\n{'-'*70}")
            print(f"Round {round_num}/{num_rounds} Complete")
            print(f"  Pursuer: {pursuer_stats['mean_reward']:.2f}")
            print(f"  Evader: {evader_stats['mean_reward']:.2f}")
            print(f"{'-'*70}\n")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total rounds: {num_rounds}")
        print(f"Pursuer model: {self.paths['pursuer_model']}")
        print(f"Evader model: {self.paths['evader_model']}")
        print("=" * 70 + "\n")

    def close(self):
        """Close environment."""
        self.env.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Kaggle Competitive MARL Training")
    parser.add_argument("--rounds", type=int, default=50, help="Number of training rounds")
    parser.add_argument("--steps-per-round", type=int, default=10000, help="Steps per round")
    parser.add_argument("--config", type=str, default=None, help="Custom config file")

    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Override with command-line args
    config["training"]["num_rounds"] = args.rounds
    config["training"]["pursuer_steps_per_round"] = args.steps_per_round
    config["training"]["evader_steps_per_round"] = args.steps_per_round

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Create trainer
    trainer = CompetitiveTrainer(config)

    try:
        # Run alternating training
        trainer.train_alternating(
            num_rounds=args.rounds,
            steps_per_round=args.steps_per_round,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
