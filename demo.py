"""
Demo script to visualize different controllers in action.

This script allows you to quickly test and visualize any of the three
control methods: PID, Kalman-PID, or SAC.
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import SAC
from environments import make_env
from controllers import PIDAgent, KalmanPIDAgent


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_demo(agent_type: str, config_path: str, sac_model_path: str = None, n_episodes: int = 5):
    """
    Run a demo of the specified agent type.

    Args:
        agent_type: One of 'pid', 'kalman-pid', or 'sac'
        config_path: Path to configuration file
        sac_model_path: Path to trained SAC model (only for SAC agent)
        n_episodes: Number of episodes to run
    """
    # Load configuration
    config = load_config(config_path)

    # Create environment with rendering
    print(f"\nInitializing {agent_type.upper()} agent...")
    env = make_env(config, render_mode="human")

    # Create agent based on type
    if agent_type.lower() == "pid":
        agent = PIDAgent(config)
        agent_name = "PID Controller"
    elif agent_type.lower() == "kalman-pid":
        agent = KalmanPIDAgent(config)
        agent_name = "Kalman Filter + PID Controller"
    elif agent_type.lower() == "sac":
        if sac_model_path is None:
            print("Error: SAC model path required for SAC agent")
            return
        agent = SAC.load(sac_model_path)
        agent_name = "SAC Deep RL Agent"
    else:
        print(f"Error: Unknown agent type '{agent_type}'")
        print("Available types: 'pid', 'kalman-pid', 'sac'")
        return

    print(f"Running {agent_name} for {n_episodes} episodes...")
    print("Close the window to stop the demo.\n")

    # Run episodes
    for episode in range(n_episodes):
        obs, info = env.reset()

        if hasattr(agent, "reset"):
            agent.reset()

        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Render
            env.render()

            # Print info periodically
            if episode_length % 50 == 0:
                distance = info.get("distance", -1)
                print(f"  Step {episode_length}: Distance = {distance:.2f}")

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")

        # Print controller-specific stats
        if hasattr(agent, "get_stats"):
            stats = agent.get_stats()
            print(f"  Detection Rate: {stats.get('detection_rate', 0):.2%}")

    env.close()
    print("\nDemo complete!")


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Demo script for visualizing controllers")
    parser.add_argument(
        "agent_type",
        type=str,
        choices=["pid", "kalman-pid", "sac"],
        help="Type of agent to demo",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--sac-model",
        type=str,
        default="models/sac/final_model",
        help="Path to trained SAC model (for SAC agent)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PURSUIT-EVASION CONTROL DEMO")
    print("=" * 80)
    print(f"\nAgent Type: {args.agent_type.upper()}")
    print(f"Episodes: {args.n_episodes}")

    run_demo(
        agent_type=args.agent_type,
        config_path=args.config,
        sac_model_path=args.sac_model,
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
