#!/usr/bin/env python3
"""
Test trained SAC model on 3D pursuit-evasion environment.

Usage:
    python test_trained_model.py --model models/sac_quick/best_model/best_model.zip
    python test_trained_model.py --model models/sac_full/best_model/best_model.zip --episodes 10
"""

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import SAC
from environments import make_env

# Fix numpy compatibility between versions
# Kaggle uses numpy 2.0+, local might use 1.x
try:
    import numpy._core.numeric as _numeric
except (ImportError, AttributeError):
    import numpy.core.numeric as _numeric
    # Alias for compatibility with numpy 2.0+
    sys.modules['numpy._core.numeric'] = _numeric
    sys.modules['numpy._core'] = sys.modules['numpy.core']


def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test_model(model_path: str, n_episodes: int = 5, render: bool = True):
    """
    Test trained model.

    Args:
        model_path: Path to trained model (.zip file)
        n_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    # Load config
    config = load_config()

    # Load trained model
    print(f"Loading model from: {model_path}")

    try:
        # Try loading with custom_objects to handle version mismatches
        from agents.networks import CustomCNN
        custom_objects = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {},
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not deserialize.*")
            model = SAC.load(model_path, custom_objects=custom_objects)

        print("✓ Model loaded successfully!")

    except Exception as e:
        print(f"\n⚠️  Warning: Failed to load with custom objects: {e}")
        print("Attempting basic load...")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not deserialize.*")
            model = SAC.load(model_path)

        print("✓ Model loaded (some parameters may be default)")
        print("   The model should still work for testing.")

    # Create environment
    render_mode = "human" if render else None
    env = make_env(config, render_mode=render_mode, use_3d=True)

    print(f"\nTesting model for {n_episodes} episodes...")
    print("=" * 70)

    episode_rewards = []
    episode_lengths = []
    focus_times = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        obs = np.array(obs)  # Convert LazyFrames to numpy array
        episode_reward = 0
        episode_length = 0
        total_focus_time = 0
        done = False

        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("-" * 70)

        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.array(obs)  # Convert LazyFrames to numpy array
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track focus time
            if info.get("in_focus", False):
                total_focus_time += 1

            # Render
            if render:
                env.render()

            # Print info periodically
            if episode_length % 50 == 0:
                distance = info.get("distance_3d", -1)
                focus_progress = info.get("focus_progress", 0)
                print(f"  Step {episode_length}: Distance={distance:.2f}, Focus Progress={focus_progress:.1%}")

        # Episode summary
        focus_percentage = (total_focus_time / episode_length) * 100
        print(f"\n  Episode Summary:")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Episode Length: {episode_length}")
        print(f"    Time in Focus: {focus_percentage:.1f}%")
        print(f"    Final Distance: {info.get('distance_3d', -1):.2f}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        focus_times.append(focus_percentage)

    # Overall statistics
    print()
    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Focus Time: {np.mean(focus_times):.1f}% ± {np.std(focus_times):.1f}%")
    print()

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test trained SAC model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (faster testing)"
    )

    args = parser.parse_args()

    test_model(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
