"""
Test Script for Trained RL Models (Local - With Rendering)

Test trained models with visualization on local machine.

Usage:
    # Test Scenario 1 model
    python testing/test_rl.py --algo ppo --model models/ppo_pursuer_final.zip --episodes 10

    # Test Scenario 2 competitive models
    python testing/test_rl.py --scenario 2 \
                              --pursuer-algo sac --pursuer-model models/sac_pursuer_latest.zip \
                              --evader-algo ppo --evader-model models/ppo_evader_latest.zip \
                              --episodes 5

    # Test with statistics
    python testing/test_rl.py --algo sac --model models/sac_pursuer_final.zip --episodes 20 --stats
"""

import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import DQN, PPO, SAC, TD3
from environments import make_env


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_rl_agent(algo: str, model_path: str):
    """Load trained RL agent."""
    print(f"Loading {algo.upper()} model from {model_path}...")

    if algo.lower() == "dqn":
        return DQN.load(model_path)
    elif algo.lower() == "ppo":
        return PPO.load(model_path)
    elif algo.lower() == "sac":
        return SAC.load(model_path)
    elif algo.lower() == "td3":
        return TD3.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from: dqn, ppo, sac, td3")


def test_scenario_1(
    algo: str,
    model_path: str,
    config: dict,
    n_episodes: int = 10,
    render: bool = True,
    compute_stats: bool = False,
):
    """
    Test Scenario 1: RL Agent vs Random Target

    Args:
        algo: Algorithm name
        model_path: Path to trained model
        config: Configuration dictionary
        n_episodes: Number of test episodes
        render: Enable rendering
        compute_stats: Compute detailed statistics
    """
    print(f"\n{'='*60}")
    print(f"TESTING SCENARIO 1: RL AGENT vs RANDOM TARGET")
    print(f"{'='*60}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Rendering: {render}")
    print()

    # Load agent
    agent = load_rl_agent(algo, model_path)

    # Create environment
    render_mode = "human" if render else None
    env = make_env(config, render_mode=render_mode, use_3d=True)

    # Test statistics
    episode_rewards = []
    episode_lengths = []
    focus_times = []
    final_distances = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        focus_count = 0
        done = False

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")

        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track focus
            if info.get("in_focus", False):
                focus_count += 1

            # Render
            if render:
                env.render()

            # Print info periodically
            if episode_length % 50 == 0:
                distance = info.get("distance_3d", -1)
                in_focus = info.get("in_focus", False)
                print(f"  Step {episode_length}: Distance={distance:.2f}, In Focus={in_focus}")

        # Episode summary
        focus_percentage = (focus_count / episode_length) * 100 if episode_length > 0 else 0
        final_distance = info.get("distance_3d", -1)

        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Focus Time: {focus_count}/{episode_length} ({focus_percentage:.1f}%)")
        print(f"  Final Distance: {final_distance:.2f}")

        # Store stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        focus_times.append(focus_percentage)
        final_distances.append(final_distance)

    env.close()

    # Overall statistics
    if compute_stats:
        print(f"\n{'='*60}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Episodes: {n_episodes}")
        print()
        print(f"Rewards:")
        print(f"  Mean: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Min/Max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
        print()
        print(f"Episode Lengths:")
        print(f"  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"  Min/Max: {np.min(episode_lengths)} / {np.max(episode_lengths)}")
        print()
        print(f"Focus Time (%):")
        print(f"  Mean: {np.mean(focus_times):.1f}% ± {np.std(focus_times):.1f}%")
        print(f"  Min/Max: {np.min(focus_times):.1f}% / {np.max(focus_times):.1f}%")
        print()
        print(f"Final Distances:")
        print(f"  Mean: {np.mean(final_distances):.2f} ± {np.std(final_distances):.2f}")
        print(f"  Min/Max: {np.min(final_distances):.2f} / {np.max(final_distances):.2f}")

    print("\nTest complete!")


def test_scenario_2(
    pursuer_algo: str,
    evader_algo: str,
    pursuer_model: str,
    evader_model: str,
    config: dict,
    n_episodes: int = 10,
    render: bool = True,
    compute_stats: bool = False,
):
    """
    Test Scenario 2: Competitive MARL (RL vs RL)

    Args:
        pursuer_algo: Pursuer algorithm name
        evader_algo: Evader algorithm name
        pursuer_model: Path to pursuer model
        evader_model: Path to evader model
        config: Configuration dictionary
        n_episodes: Number of test episodes
        render: Enable rendering
        compute_stats: Compute detailed statistics
    """
    print(f"\n{'='*60}")
    print(f"TESTING SCENARIO 2: COMPETITIVE MARL (RL vs RL)")
    print(f"{'='*60}")
    print(f"Pursuer: {pursuer_algo.upper()} ({pursuer_model})")
    print(f"Evader: {evader_algo.upper()} ({evader_model})")
    print(f"Episodes: {n_episodes}")
    print()

    # Load agents
    pursuer_agent = load_rl_agent(pursuer_algo, pursuer_model)

    # For now, test only pursuer (evader integration TBD)
    print("NOTE: Full competitive testing requires evader environment integration.")
    print("Testing pursuer only for now...")

    test_scenario_1(
        algo=pursuer_algo,
        model_path=pursuer_model,
        config=config,
        n_episodes=n_episodes,
        render=render,
        compute_stats=compute_stats,
    )


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test trained RL models with rendering (local)"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2],
        default=1,
        help="Test scenario: 1 (RL vs Random) or 2 (RL vs RL)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute detailed statistics",
    )

    # Scenario 1 arguments
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo", "sac", "td3"],
        help="RL algorithm for Scenario 1",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model for Scenario 1",
    )

    # Scenario 2 arguments
    parser.add_argument(
        "--pursuer-algo",
        type=str,
        choices=["dqn", "ppo", "sac", "td3"],
        help="Pursuer algorithm for Scenario 2",
    )
    parser.add_argument(
        "--evader-algo",
        type=str,
        choices=["dqn", "ppo", "sac", "td3"],
        help="Evader algorithm for Scenario 2",
    )
    parser.add_argument(
        "--pursuer-model",
        type=str,
        help="Path to pursuer model for Scenario 2",
    )
    parser.add_argument(
        "--evader-model",
        type=str,
        help="Path to evader model for Scenario 2",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Test
    if args.scenario == 1:
        if args.algo is None or args.model is None:
            print("ERROR: Scenario 1 requires --algo and --model")
            return

        test_scenario_1(
            algo=args.algo,
            model_path=args.model,
            config=config,
            n_episodes=args.episodes,
            render=not args.no_render,
            compute_stats=args.stats,
        )

    elif args.scenario == 2:
        if (args.pursuer_algo is None or args.pursuer_model is None or
            args.evader_algo is None or args.evader_model is None):
            print("ERROR: Scenario 2 requires --pursuer-algo, --pursuer-model, --evader-algo, --evader-model")
            return

        test_scenario_2(
            pursuer_algo=args.pursuer_algo,
            evader_algo=args.evader_algo,
            pursuer_model=args.pursuer_model,
            evader_model=args.evader_model,
            config=config,
            n_episodes=args.episodes,
            render=not args.no_render,
            compute_stats=args.stats,
        )


if __name__ == "__main__":
    main()
