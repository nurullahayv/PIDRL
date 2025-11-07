"""
Local Testing Script with Rendering

Tests trained models with 3D HUD visualization.

Usage:
    python test_with_render.py --pursuer models/pursuer_latest.zip --evader models/evader_latest.zip
"""

import sys
import os
import argparse
import numpy as np
import pygame

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.pursuit_evasion_3d import CompetitivePursuitEvasion3D, get_target_observation
from agents.pursuer_agent import PursuerAgent
from agents.evader_agent import EvaderAgent
from config import get_config


def test_competitive_agents(
    pursuer_model_path: str,
    evader_model_path: str,
    n_episodes: int = 5,
    deterministic: bool = True,
):
    """
    Test trained models with rendering.

    Args:
        pursuer_model_path: Path to pursuer model
        evader_model_path: Path to evader model
        n_episodes: Number of episodes to run
        deterministic: Use deterministic policy
    """
    # Load config
    config = get_config()
    env_config = config["env"]

    print("=" * 70)
    print("COMPETITIVE MARL - LOCAL TESTING WITH RENDERING")
    print("=" * 70)
    print(f"Pursuer Model: {pursuer_model_path}")
    print(f"Evader Model: {evader_model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("=" * 70)
    print()

    # Create environment WITH rendering
    print("Creating environment with rendering...")
    env = CompetitivePursuitEvasion3D(
        view_size=env_config["view_size"],
        frame_size=env_config["frame_size"],
        depth_range=env_config["depth_range"],
        success_threshold=env_config["success_threshold"],
        target_size=env_config["target_size"],
        max_velocity=env_config["max_velocity"],
        max_acceleration=env_config["max_acceleration"],
        max_angular_velocity=env_config["max_angular_velocity"],
        dt=env_config["dt"],
        max_steps=env_config["max_steps"],
        focus_reward=env_config["focus_reward"],
        focus_bonus=env_config["focus_bonus"],
        escape_penalty=env_config["escape_penalty"],
        outside_penalty_scale=env_config["outside_penalty_scale"],
        focus_time_threshold=env_config["focus_time_threshold"],
        render_mode="human",  # ENABLE RENDERING
    )

    # Load agents
    print("Loading agents...")

    # Check if models exist
    if not os.path.exists(pursuer_model_path):
        print(f"ERROR: Pursuer model not found: {pursuer_model_path}")
        return
    if not os.path.exists(evader_model_path):
        print(f"ERROR: Evader model not found: {evader_model_path}")
        return

    pursuer = PursuerAgent.load_from_file(pursuer_model_path, env)
    evader = EvaderAgent.load_from_file(evader_model_path, env)

    print("Agents loaded successfully!")
    print()

    # Run episodes
    episode_stats = []

    for episode in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*70}")

        obs, info = env.reset()
        episode_reward = 0
        episode_evader_reward = 0
        step = 0
        total_focus_time = 0
        running = True

        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("\nWindow closed by user.")
                    break

            if not running:
                break

            # Pursuer action
            pursuer_action = pursuer.predict(obs, deterministic=deterministic)

            # Evader action
            evader_obs = get_target_observation(env)
            evader_action = evader.predict(evader_obs, deterministic=deterministic)
            env.set_target_action(evader_action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(pursuer_action)

            # Track stats
            episode_reward += reward
            episode_evader_reward += info["target_reward"]
            if info["in_focus"]:
                total_focus_time += 1
            step += 1

            # Render
            should_continue = env.render()
            if not should_continue:
                running = False

            # Check termination
            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                print(f"  Pursuer Reward: {episode_reward:.2f}")
                print(f"  Evader Reward: {episode_evader_reward:.2f}")
                print(f"  Focus Time: {total_focus_time} steps ({total_focus_time/step*100:.1f}%)")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")

                episode_stats.append({
                    "episode": episode + 1,
                    "steps": step,
                    "pursuer_reward": episode_reward,
                    "evader_reward": episode_evader_reward,
                    "focus_time": total_focus_time,
                    "focus_percentage": total_focus_time / step * 100 if step > 0 else 0,
                })

                # Wait a bit before next episode
                pygame.time.wait(2000)
                break

        if not running:
            break

    # Print summary
    if episode_stats:
        print("\n" + "=" * 70)
        print("TESTING SUMMARY")
        print("=" * 70)

        pursuer_rewards = [s["pursuer_reward"] for s in episode_stats]
        evader_rewards = [s["evader_reward"] for s in episode_stats]
        focus_percentages = [s["focus_percentage"] for s in episode_stats]

        print(f"\nEpisodes Completed: {len(episode_stats)}")
        print(f"\nPursuer Performance:")
        print(f"  Mean Reward: {np.mean(pursuer_rewards):.2f} ± {np.std(pursuer_rewards):.2f}")
        print(f"  Min/Max: {np.min(pursuer_rewards):.2f} / {np.max(pursuer_rewards):.2f}")

        print(f"\nEvader Performance:")
        print(f"  Mean Reward: {np.mean(evader_rewards):.2f} ± {np.std(evader_rewards):.2f}")
        print(f"  Min/Max: {np.min(evader_rewards):.2f} / {np.max(evader_rewards):.2f}")

        print(f"\nFocus Statistics:")
        print(f"  Mean Focus Time: {np.mean(focus_percentages):.1f}% ± {np.std(focus_percentages):.1f}%")
        print(f"  Min/Max: {np.min(focus_percentages):.1f}% / {np.max(focus_percentages):.1f}%")

        print("\n" + "=" * 70)

    # Close environment
    env.close()


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test Competitive MARL with Rendering")
    parser.add_argument("--pursuer", type=str, default="models/pursuer_latest.zip",
                        help="Path to pursuer model")
    parser.add_argument("--evader", type=str, default="models/evader_latest.zip",
                        help="Path to evader model")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")

    args = parser.parse_args()

    test_competitive_agents(
        pursuer_model_path=args.pursuer,
        evader_model_path=args.evader,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
