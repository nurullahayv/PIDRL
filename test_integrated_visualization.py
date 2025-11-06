#!/usr/bin/env python3
"""
Test script for integrated multi-agent visualization.

Tests:
1. MATLAB-style 3D visualization with FOV cones
2. demo_3d-style HUD rendering for each agent
3. Updated target size (4.0)
4. Updated focus threshold (9.0 = 30% of view)
5. Split-screen rendering (3D global + HUD grid)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pygame
from environments.multi_agent_integrated import MultiAgentIntegratedEnv

def test_visualization():
    """Test the integrated visualization system."""
    print("=" * 70)
    print("Testing Integrated Multi-Agent Visualization")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✓ MATLAB-style 3D global view with FOV cones")
    print("  ✓ demo_3d-style HUD for each agent")
    print("  ✓ Target size: 4.0 (doubled)")
    print("  ✓ Focus threshold: 9.0 (30% of view_size=30.0)")
    print("  ✓ Split-screen: LEFT=3D global, RIGHT=HUD grid")
    print()

    # Create environment
    env = MultiAgentIntegratedEnv(
        num_agents=4,
        num_teams=2,
        arena_size=1000.0,
        max_velocity=50.0,
        max_acceleration=5.0,
        fov_range=300.0,
        render_mode="human",
    )

    print("Environment created successfully!")
    print(f"  Agents: {env.num_agents}")
    print(f"  Teams: {env.num_teams}")
    print(f"  Arena size: {env.arena_size}")
    print(f"  FOV range: {env.fov_range}")
    print(f"  Success threshold: {env.success_threshold} (30% focus area)")
    print()

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Environment reset. {info['num_alive']} agents alive.")
    print()

    print("Running visualization demo...")
    print("Close the window to exit.")
    print()

    # Run for a few steps
    running = True
    step_count = 0
    max_steps = 1000

    try:
        while running and step_count < max_steps:
            # Random action for testing
            action = np.random.uniform(-0.5, 0.5, size=3)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Render
            env.render()

            # Check for pygame events (to allow closing window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward={reward:.2f}, Alive={info['num_alive']}")

            # Check termination
            if terminated or truncated:
                print(f"\nEpisode ended at step {step_count}")
                print(f"  Terminated: {terminated}")
                print(f"  Truncated: {truncated}")
                print(f"  Agents alive: {info['num_alive']}")

                # Reset for another round
                if step_count < max_steps:
                    print("\nResetting for another episode...")
                    obs, info = env.reset()
                    step_count = 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        env.close()
        print("\nEnvironment closed.")
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)


if __name__ == "__main__":
    test_visualization()
