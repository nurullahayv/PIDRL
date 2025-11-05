"""
Evaluation script for all controllers (PID, Kalman-PID, SAC).

This script evaluates trained controllers and computes performance metrics
for comparison in the research paper.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from environments import make_env
from controllers import PIDAgent, KalmanPIDAgent


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_agent(
    agent,
    env,
    n_episodes: int = 100,
    render: bool = False,
    agent_name: str = "Agent",
) -> Dict[str, Any]:
    """
    Evaluate an agent and collect performance metrics.

    Args:
        agent: Agent to evaluate (PID, Kalman-PID, or SAC)
        env: Environment to evaluate in
        n_episodes: Number of episodes to run
        render: Whether to render episodes
        agent_name: Name of agent for logging

    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    success_rates = []
    distances_over_time = []

    print(f"\nEvaluating {agent_name} for {n_episodes} episodes...")

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()

        if hasattr(agent, "reset"):
            agent.reset()

        episode_reward = 0
        episode_length = 0
        episode_distances = []
        successes = []

        done = False
        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track distance and success
            if "distance" in info:
                episode_distances.append(info["distance"])

            if "success" in info:
                successes.append(info["success"])

            if render:
                env.render()

        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_distances:
            mean_distance = np.mean(episode_distances)
            tracking_errors.append(mean_distance)
            distances_over_time.append(episode_distances)

        if successes:
            success_rate = np.mean(successes)
            success_rates.append(success_rate)

    # Compute aggregate statistics
    results = {
        "agent_name": agent_name,
        "n_episodes": n_episodes,
        # Reward statistics
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        # Episode length
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        # Tracking error
        "mean_tracking_error": np.mean(tracking_errors),
        "std_tracking_error": np.std(tracking_errors),
        "min_tracking_error": np.min(tracking_errors),
        "max_tracking_error": np.max(tracking_errors),
        # Success rate
        "mean_success_rate": np.mean(success_rates),
        "std_success_rate": np.std(success_rates),
        # Raw data for plotting
        "episode_rewards": episode_rewards,
        "tracking_errors": tracking_errors,
        "success_rates": success_rates,
        "distances_over_time": distances_over_time,
    }

    # Add controller-specific statistics if available
    if hasattr(agent, "get_stats"):
        stats = agent.get_stats()
        results.update({f"agent_{k}": v for k, v in stats.items()})

    return results


def evaluate_all_methods(
    config: Dict[str, Any],
    sac_model_path: Optional[str] = None,
    n_episodes: int = 100,
    render: bool = False,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all three methods: PID, Kalman-PID, and SAC.

    Args:
        config: Configuration dictionary
        sac_model_path: Path to trained SAC model
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        seed: Random seed

    Returns:
        Dictionary with results for each method
    """
    all_results = {}

    # Create environment
    print("Creating evaluation environment...")
    env = make_env(config, render_mode="human" if render else None)
    env.reset(seed=seed)

    # 1. Evaluate PID Controller
    print("\n" + "=" * 50)
    print("Evaluating PID Controller (Baseline 1)")
    print("=" * 50)
    pid_agent = PIDAgent(config)
    pid_results = evaluate_agent(
        pid_agent, env, n_episodes=n_episodes, render=render, agent_name="PID"
    )
    all_results["PID"] = pid_results

    # 2. Evaluate Kalman-PID Controller
    print("\n" + "=" * 50)
    print("Evaluating Kalman Filter + PID (Baseline 2)")
    print("=" * 50)
    kalman_pid_agent = KalmanPIDAgent(config)
    kalman_results = evaluate_agent(
        kalman_pid_agent,
        env,
        n_episodes=n_episodes,
        render=render,
        agent_name="Kalman-PID",
    )
    all_results["Kalman-PID"] = kalman_results

    # 3. Evaluate SAC Agent (if model path provided)
    if sac_model_path and os.path.exists(sac_model_path + ".zip"):
        print("\n" + "=" * 50)
        print("Evaluating SAC Agent (Deep RL)")
        print("=" * 50)
        sac_agent = SAC.load(sac_model_path)
        sac_results = evaluate_agent(
            sac_agent, env, n_episodes=n_episodes, render=render, agent_name="SAC"
        )
        all_results["SAC"] = sac_results
    else:
        print(f"\nWarning: SAC model not found at {sac_model_path}")
        print("Skipping SAC evaluation.")

    env.close()

    return all_results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table of results."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Create comparison dataframe
    methods = list(results.keys())
    metrics = [
        ("Mean Reward", "mean_reward"),
        ("Std Reward", "std_reward"),
        ("Mean Tracking Error", "mean_tracking_error"),
        ("Std Tracking Error", "std_tracking_error"),
        ("Mean Success Rate (%)", "mean_success_rate"),
        ("Mean Episode Length", "mean_length"),
    ]

    print(f"\n{'Metric':<30}", end="")
    for method in methods:
        print(f"{method:>15}", end="")
    print()
    print("-" * (30 + 15 * len(methods)))

    for metric_name, metric_key in metrics:
        print(f"{metric_name:<30}", end="")
        for method in methods:
            value = results[method].get(metric_key, 0)
            if "success_rate" in metric_key.lower():
                value *= 100  # Convert to percentage
            print(f"{value:>15.4f}", end="")
        print()

    print("=" * 80)


def save_results(results: Dict[str, Dict[str, Any]], output_dir: str = "results"):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary statistics to CSV
    summary_data = []
    for method, method_results in results.items():
        row = {"Method": method}
        for key, value in method_results.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = value
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # Save detailed results to numpy files
    for method, method_results in results.items():
        method_dir = os.path.join(output_dir, method.lower())
        os.makedirs(method_dir, exist_ok=True)

        # Save arrays
        np.save(
            os.path.join(method_dir, "episode_rewards.npy"),
            method_results["episode_rewards"],
        )
        np.save(
            os.path.join(method_dir, "tracking_errors.npy"),
            method_results["tracking_errors"],
        )
        np.save(
            os.path.join(method_dir, "success_rates.npy"),
            method_results["success_rates"],
        )

    print(f"Detailed results saved to {output_dir}/")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all controllers")
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
        help="Path to trained SAC model",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during evaluation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Evaluate all methods
    results = evaluate_all_methods(
        config=config,
        sac_model_path=args.sac_model,
        n_episodes=args.n_episodes,
        render=args.render,
        seed=args.seed,
    )

    # Print comparison table
    print_comparison_table(results)

    # Save results
    save_results(results, output_dir=args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
