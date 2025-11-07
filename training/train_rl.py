"""
Training Script for RL Algorithms (Kaggle Optimized - No Rendering)

Supports:
- DQN (Discrete action space)
- PPO (Policy Gradient)
- SAC (Actor-Critic, Off-Policy)
- TD3 (Actor-Critic, Off-Policy)

Usage:
    # Train with PPO (recommended for continuous control)
    python training/train_rl.py --algo ppo --scenario 1 --timesteps 500000

    # Train with SAC (good for exploration)
    python training/train_rl.py --algo sac --scenario 1 --timesteps 500000

    # Train with DQN (discrete actions only)
    python training/train_rl.py --algo dqn --scenario 1 --timesteps 500000

    # Train competitive MARL (Scenario 2)
    python training/train_rl.py --algo sac --scenario 2 --timesteps 500000 --competitive

    # Resume training
    python training/train_rl.py --algo ppo --scenario 1 --timesteps 1000000 --resume models/ppo_pursuer.zip
"""

import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from environments import make_env


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_algorithm(algo_name: str, env, config: dict, tensorboard_log: str = None):
    """
    Create RL algorithm instance.

    Args:
        algo_name: Algorithm name (dqn, ppo, sac, td3)
        env: Training environment
        config: Configuration dictionary
        tensorboard_log: Path for tensorboard logs

    Returns:
        Algorithm instance
    """
    algo_config = config.get(algo_name.lower(), {})

    common_kwargs = {
        "policy": "CnnPolicy",
        "env": env,
        "verbose": 1,
        "tensorboard_log": tensorboard_log,
    }

    if algo_name.lower() == "dqn":
        return DQN(
            **common_kwargs,
            learning_rate=algo_config.get("learning_rate", 1e-4),
            buffer_size=algo_config.get("buffer_size", 100000),
            batch_size=algo_config.get("batch_size", 128),
            gamma=algo_config.get("gamma", 0.99),
            target_update_interval=algo_config.get("target_update_interval", 10000),
            exploration_fraction=algo_config.get("exploration_fraction", 0.3),
            exploration_initial_eps=algo_config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=algo_config.get("exploration_final_eps", 0.05),
            learning_starts=algo_config.get("learning_starts", 10000),
            train_freq=algo_config.get("train_freq", 4),
        )

    elif algo_name.lower() == "ppo":
        return PPO(
            **common_kwargs,
            learning_rate=algo_config.get("learning_rate", 3e-4),
            n_steps=algo_config.get("n_steps", 2048),
            batch_size=algo_config.get("batch_size", 128),
            n_epochs=algo_config.get("n_epochs", 10),
            gamma=algo_config.get("gamma", 0.99),
            gae_lambda=algo_config.get("gae_lambda", 0.95),
            clip_range=algo_config.get("clip_range", 0.2),
            ent_coef=algo_config.get("ent_coef", 0.01),
        )

    elif algo_name.lower() == "sac":
        return SAC(
            **common_kwargs,
            learning_rate=algo_config.get("learning_rate", 3e-4),
            buffer_size=algo_config.get("buffer_size", 200000),
            batch_size=algo_config.get("batch_size", 256),
            gamma=algo_config.get("gamma", 0.99),
            tau=algo_config.get("tau", 0.005),
            ent_coef=algo_config.get("ent_coef", "auto"),
            learning_starts=algo_config.get("learning_starts", 10000),
            train_freq=algo_config.get("train_freq", 1),
        )

    elif algo_name.lower() == "td3":
        return TD3(
            **common_kwargs,
            learning_rate=algo_config.get("learning_rate", 3e-4),
            buffer_size=algo_config.get("buffer_size", 200000),
            batch_size=algo_config.get("batch_size", 256),
            gamma=algo_config.get("gamma", 0.99),
            tau=algo_config.get("tau", 0.005),
            policy_delay=algo_config.get("policy_delay", 2),
            target_policy_noise=algo_config.get("target_policy_noise", 0.2),
            target_noise_clip=algo_config.get("target_noise_clip", 0.5),
            learning_starts=algo_config.get("learning_starts", 10000),
            train_freq=algo_config.get("train_freq", 1),
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from: dqn, ppo, sac, td3")


def train_scenario_1(
    algo_name: str,
    config: dict,
    total_timesteps: int = 500000,
    save_dir: str = "models",
    tensorboard_log: str = "logs",
    resume_path: str = None,
):
    """
    Train Scenario 1: RL Agent vs Random Target

    Args:
        algo_name: Algorithm name (dqn, ppo, sac, td3)
        config: Configuration dictionary
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
        resume_path: Path to resume training from
    """
    print(f"\n{'='*60}")
    print(f"TRAINING SCENARIO 1: RL AGENT vs RANDOM TARGET")
    print(f"{'='*60}")
    print(f"Algorithm: {algo_name.upper()}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Save Directory: {save_dir}")
    print(f"TensorBoard Logs: {tensorboard_log}")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    print()

    # Create save directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Create environment (NO RENDERING for Kaggle)
    print("Creating training environment (no rendering)...")
    env = make_env(config, render_mode=None, use_3d=True)
    env = Monitor(env)

    # Create or load algorithm
    if resume_path:
        print(f"Loading model from {resume_path}...")
        if algo_name.lower() == "dqn":
            model = DQN.load(resume_path, env=env)
        elif algo_name.lower() == "ppo":
            model = PPO.load(resume_path, env=env)
        elif algo_name.lower() == "sac":
            model = SAC.load(resume_path, env=env)
        elif algo_name.lower() == "td3":
            model = TD3.load(resume_path, env=env)
    else:
        print("Creating new model...")
        model = create_algorithm(algo_name, env, config, tensorboard_log)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_dir}/checkpoints",
        name_prefix=f"{algo_name}_pursuer",
    )

    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("Monitor progress with: tensorboard --logdir " + tensorboard_log)
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_dir}/{algo_name}_pursuer_final.zip"
    model.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")

    env.close()


def train_scenario_2_competitive(
    algo_name: str,
    config: dict,
    total_timesteps: int = 500000,
    save_dir: str = "models",
    tensorboard_log: str = "logs",
    alternating_rounds: int = 10,
):
    """
    Train Scenario 2: Competitive MARL (RL vs RL) with Alternating Training

    Args:
        algo_name: Algorithm name (dqn, ppo, sac, td3)
        config: Configuration dictionary
        total_timesteps: Total training timesteps PER AGENT
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
        alternating_rounds: Number of alternating training rounds
    """
    print(f"\n{'='*60}")
    print(f"TRAINING SCENARIO 2: COMPETITIVE MARL (RL vs RL)")
    print(f"{'='*60}")
    print(f"Algorithm: {algo_name.upper()}")
    print(f"Training Mode: ALTERNATING")
    print(f"Rounds: {alternating_rounds}")
    print(f"Timesteps per round: {total_timesteps // alternating_rounds}")
    print()

    # Create save directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Initialize pursuer and evader
    pursuer_path = f"{save_dir}/{algo_name}_pursuer_latest.zip"
    evader_path = f"{save_dir}/{algo_name}_evader_latest.zip"

    timesteps_per_round = total_timesteps // alternating_rounds

    for round_num in range(alternating_rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1}/{alternating_rounds}")
        print(f"{'='*60}")

        # Train pursuer against current evader
        print(f"\n[Pursuer Training]")
        env_pursuer = make_env(config, render_mode=None, use_3d=True)
        env_pursuer = Monitor(env_pursuer)

        if round_num == 0:
            pursuer = create_algorithm(algo_name, env_pursuer, config, f"{tensorboard_log}/pursuer")
        else:
            pursuer = SAC.load(pursuer_path, env=env_pursuer) if algo_name.lower() == "sac" else PPO.load(pursuer_path, env=env_pursuer)

        pursuer.learn(total_timesteps=timesteps_per_round, progress_bar=True)
        pursuer.save(pursuer_path)
        print(f"Pursuer saved to: {pursuer_path}")
        env_pursuer.close()

        # Train evader against updated pursuer
        print(f"\n[Evader Training]")
        env_evader = make_env(config, render_mode=None, use_3d=True)
        env_evader = Monitor(env_evader)

        if round_num == 0:
            evader = create_algorithm(algo_name, env_evader, config, f"{tensorboard_log}/evader")
        else:
            evader = SAC.load(evader_path, env=env_evader) if algo_name.lower() == "sac" else PPO.load(evader_path, env=env_evader)

        evader.learn(total_timesteps=timesteps_per_round, progress_bar=True)
        evader.save(evader_path)
        print(f"Evader saved to: {evader_path}")
        env_evader.close()

    print(f"\n{'='*60}")
    print(f"COMPETITIVE MARL TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Pursuer model: {pursuer_path}")
    print(f"Evader model: {evader_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train RL agents for 3D pursuit-evasion (Kaggle optimized)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "ppo", "sac", "td3"],
        help="RL algorithm",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training scenario: 1 (RL vs Random) or 2 (RL vs RL)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to resume training from",
    )
    parser.add_argument(
        "--competitive",
        action="store_true",
        help="Use competitive MARL (alternating training)",
    )
    parser.add_argument(
        "--alternating-rounds",
        type=int,
        default=10,
        help="Number of alternating rounds for competitive training",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    if args.scenario == 1:
        train_scenario_1(
            algo_name=args.algo,
            config=config,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            tensorboard_log=args.tensorboard_log,
            resume_path=args.resume,
        )

    elif args.scenario == 2:
        if args.competitive:
            train_scenario_2_competitive(
                algo_name=args.algo,
                config=config,
                total_timesteps=args.timesteps,
                save_dir=args.save_dir,
                tensorboard_log=args.tensorboard_log,
                alternating_rounds=args.alternating_rounds,
            )
        else:
            print("Scenario 2 requires --competitive flag for alternating training")


if __name__ == "__main__":
    main()
