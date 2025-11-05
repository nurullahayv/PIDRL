"""
2.5D Dogfight HUD Demo: Visualize 3D Error Vector Nullification

This script provides an interactive visualization of the upgraded 2.5D environment
where depth perception and multi-target tracking are key features.

What you'll see:
- Green crosshair (center): Your agent - always fixed at center
- Color-coded targets: Multiple targets with depth-based scaling
- Cyan 3D error vector arrow: Points to primary target in 3D space
- Depth indicators: Range display showing z-distance
- HUD display: 3D error (XY + Z breakdown), velocities, lock status

The agent now controls 3D acceleration (ax, ay, az) to nullify the
full 3D error vector, including range management via thrust control.

Usage:
    python demo_3d.py --n-episodes 3
    python demo_3d.py --num-targets 3  # Multi-target scenario
"""

import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from environments import make_env


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_demo_3d(
    config_path: str = "configs/config.yaml",
    n_episodes: int = 5,
    num_targets: int = 1,
):
    """
    Run a demo of the 2.5D environment with manual (random) control.

    Args:
        config_path: Path to configuration file
        n_episodes: Number of episodes to run
        num_targets: Number of targets to spawn
    """
    # Load configuration
    config = load_config(config_path)

    # Override number of targets
    if "environment" not in config:
        config["environment"] = {}
    config["environment"]["num_targets"] = num_targets

    # Create 2.5D environment with rendering
    print(f"\n{'='*60}")
    print(f"2.5D DOGFIGHT HUD DEMO")
    print(f"{'='*60}")
    print(f"Initializing 2.5D environment with {num_targets} target(s)...")
    print(f"Action space: 3D acceleration (ax, ay, az)")
    print(f"Observation: Egocentric view with depth encoding")
    print(f"\nClose the window to stop the demo.\n")

    env = make_env(config, render_mode="human", use_3d=True)

    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")
        print(f"Targets: {info['num_targets']}")

        while not done:
            # Simple demo controller: random actions
            # In practice, you'd use PID or RL agent here
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Render
            env.render()

            # Print info periodically
            if episode_length % 50 == 0:
                distance_3d = info.get("distance_3d", -1)
                distance_xy = info.get("distance_xy", -1)
                distance_z = info.get("distance_z", -1)
                print(f"  Step {episode_length}: 3D Error = {distance_3d:.2f} (XY: {distance_xy:.2f}, Z: {distance_z:.2f})")

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Final 3D Distance: {info.get('distance_3d', -1):.2f}")

    env.close()
    print("\nDemo complete!")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="2.5D Dogfight HUD demo with depth perception"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=1,
        help="Number of targets (1-5)",
    )

    args = parser.parse_args()

    run_demo_3d(
        config_path=args.config,
        n_episodes=args.n_episodes,
        num_targets=min(args.num_targets, 5),  # Cap at 5
    )


if __name__ == "__main__":
    main()
