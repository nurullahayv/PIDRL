#!/usr/bin/env python3
"""
Integrated Multi-Agent Demo with Geographic Coordinates and Egocentric HUDs

This demo shows:
1. Global geographic coordinate system (left panel)
   - All agents moving in 3D space
   - Trajectory trails
   - Top-down view

2. Individual egocentric HUDs (right panels)
   - Each agent sees own pursuit-evasion view
   - Phase 1 style HUD for each agent
   - 2x2 grid for up to 4 agents

Usage:
    python demo_integrated.py                    # 4 agents, 2 teams
    python demo_integrated.py --agents 6         # 6 agents
    python demo_integrated.py --teams 3          # 3 teams
    python demo_integrated.py --n-episodes 5     # 5 episodes
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environments.multi_agent_integrated import MultiAgentIntegratedEnv
import numpy as np


def run_demo(
    num_agents: int = 4,
    num_teams: int = 2,
    n_episodes: int = 3,
    max_steps_per_episode: int = 1000,
):
    """
    Run integrated multi-agent demo.

    Args:
        num_agents: Number of agents
        num_teams: Number of teams
        n_episodes: Number of episodes to run
        max_steps_per_episode: Max steps per episode
    """
    print("=" * 80)
    print("INTEGRATED MULTI-AGENT SYSTEM")
    print("=" * 80)
    print()
    print("LEFT PANEL: Global Geographic View")
    print("  - Shows all agents in 3D arena")
    print("  - Trajectory trails")
    print("  - Top-down perspective")
    print()
    print("RIGHT PANELS: Individual Egocentric HUDs (2x2 grid)")
    print("  - Each agent has own pursuit-evasion view")
    print("  - Agent always at center of own HUD")
    print("  - Enemies shown relative to agent")
    print()
    print(f"Configuration:")
    print(f"  - Agents: {num_agents}")
    print(f"  - Teams: {num_teams}")
    print(f"  - Episodes: {n_episodes}")
    print()
    print("Close window to stop demo")
    print("=" * 80)
    print()

    # Create environment
    env = MultiAgentIntegratedEnv(
        num_agents=num_agents,
        num_teams=num_teams,
        arena_size=1000.0,
        max_velocity=50.0,
        max_acceleration=5.0,
        fov_range=300.0,
        render_mode="human",
    )

    try:
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print("-" * 80)

            obs, info = env.reset()
            done = False
            step = 0
            episode_reward = 0

            while not done and step < max_steps_per_episode:
                # Simple policy: Random actions for agent 0
                # (Other agents use built-in AI)
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                step += 1

                # Render
                env.render()

                # Print progress every 100 steps
                if step % 100 == 0:
                    num_alive = info.get("num_alive", 0)
                    print(f"  Step {step}: Agents alive={num_alive}, Reward={reward:.2f}")

            print(f"\nEpisode {episode + 1} complete:")
            print(f"  Total steps: {step}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Final agents alive: {info.get('num_alive', 0)}")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")

    finally:
        env.close()
        print("\nDemo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Multi-Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_integrated.py                    # Default: 4 agents, 2 teams
  python demo_integrated.py --agents 6         # 6 agents
  python demo_integrated.py --teams 3          # 3 teams
  python demo_integrated.py --n-episodes 5     # 5 episodes

Views:
  LEFT:  Global geographic map (all agents, trajectories)
  RIGHT: Egocentric HUDs (2x2 grid, up to 4 agents shown)
        """
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=4,
        help="Number of agents (default: 4)"
    )
    parser.add_argument(
        "--teams",
        type=int,
        default=2,
        help="Number of teams (default: 2)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes (default: 3)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)"
    )

    args = parser.parse_args()

    run_demo(
        num_agents=args.agents,
        num_teams=args.teams,
        n_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps,
    )


if __name__ == "__main__":
    main()
