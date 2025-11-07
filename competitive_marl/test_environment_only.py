"""
Simple environment test without requiring stable-baselines3.

Just verifies environment creation and basic simulation.
"""

import sys
import os
import numpy as np

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from environment.pursuit_evasion_3d import CompetitivePursuitEvasion3D, get_target_observation
from config import get_config


def test_environment():
    """Test environment without agents."""
    print("=" * 70)
    print("ENVIRONMENT ONLY TEST")
    print("=" * 70)
    print()

    # Load config
    print("1. Loading configuration...")
    config = get_config()
    print("   ✓ Config loaded")
    print()

    # Create environment
    print("2. Creating environment...")
    env = CompetitivePursuitEvasion3D(
        view_size=config['env']['view_size'],
        frame_size=config['env']['frame_size'],
        depth_range=config['env']['depth_range'],
        success_threshold=config['env']['success_threshold'],
        target_size=config['env']['target_size'],
        max_velocity=config['env']['max_velocity'],
        max_acceleration=config['env']['max_acceleration'],
        max_angular_velocity=config['env']['max_angular_velocity'],
        dt=config['env']['dt'],
        max_steps=config['env']['max_steps'],
        focus_reward=config['env']['focus_reward'],
        focus_bonus=config['env']['focus_bonus'],
        escape_penalty=config['env']['escape_penalty'],
        outside_penalty_scale=config['env']['outside_penalty_scale'],
        focus_time_threshold=config['env']['focus_time_threshold'],
        render_mode=None,
    )
    print("   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print()

    # Reset
    print("3. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Reset successful")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Initial target position: {info['target_position']}")
    print()

    # Run simulation
    print("4. Running simulation (20 steps with random actions)...")
    episode_reward = 0
    focus_count = 0

    for step in range(20):
        # Random actions for both pursuer and evader
        pursuer_action = env.action_space.sample()
        evader_action = env.action_space.sample()

        # Apply evader action
        env.set_target_action(evader_action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(pursuer_action)

        episode_reward += reward
        if info['in_focus']:
            focus_count += 1

        if step % 5 == 0:
            print(f"   Step {step:2d}: Reward={reward:6.2f}, Distance={info['distance_3d']:5.1f}, Focus={info['in_focus']}")

        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            break

    print()
    print(f"   ✓ Simulation complete")
    print(f"   - Total steps: {step + 1}")
    print(f"   - Episode reward: {episode_reward:.2f}")
    print(f"   - Focus time: {focus_count} steps ({focus_count/(step+1)*100:.1f}%)")
    print()

    # Test target observation
    print("5. Testing target observation (evader's view)...")
    target_obs = get_target_observation(env)
    print(f"   ✓ Target observation shape: {target_obs.shape}")
    print(f"   - Non-zero pixels: {np.count_nonzero(target_obs)}")
    print()

    # Close
    env.close()

    # Summary
    print("=" * 70)
    print("TEST PASSED ✓")
    print("=" * 70)
    print()
    print("Environment is working correctly!")
    print()
    print("To train with RL agents:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run verification: python verify_setup.py")
    print("  3. Start training: python training/train_kaggle.py")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"\n✗ TEST FAILED:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
