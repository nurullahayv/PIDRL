"""
Quick verification script to test the competitive MARL setup.

Runs a few steps without training to verify:
1. Environment creation
2. Agent initialization
3. Basic simulation loop
4. No rendering (Kaggle-compatible)
"""

import sys
import os
import numpy as np

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from environment.pursuit_evasion_3d import CompetitivePursuitEvasion3D, get_target_observation
from agents.pursuer_agent import PursuerAgent
from agents.evader_agent import EvaderAgent
from config import get_config


def verify_setup():
    """Verify that everything is set up correctly."""
    print("=" * 70)
    print("COMPETITIVE MARL SETUP VERIFICATION")
    print("=" * 70)
    print()

    # 1. Load config
    print("1. Loading configuration...")
    config = get_config()
    print("   ✓ Config loaded successfully")
    print(f"   - View Size: {config['env']['view_size']}")
    print(f"   - Success Threshold: {config['env']['success_threshold']}")
    print(f"   - Target Size: {config['env']['target_size']}")
    print()

    # 2. Create environment
    print("2. Creating environment (NO RENDERING)...")
    env = CompetitivePursuitEvasion3D(
        **config['env'],
        render_mode=None,  # NO RENDERING
    )
    print("   ✓ Environment created successfully")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print()

    # 3. Initialize agents
    print("3. Initializing agents...")
    pursuer = PursuerAgent(env, **config['pursuer'], verbose=0)
    evader = EvaderAgent(env, **config['evader'], verbose=0)
    print("   ✓ Agents initialized successfully")
    print()

    # 4. Run a few steps
    print("4. Running simulation (10 steps)...")
    obs, info = env.reset(seed=42)
    print(f"   - Initial observation shape: {obs.shape}")

    for step in range(10):
        # Random actions for verification
        pursuer_action = env.action_space.sample()
        evader_action = env.action_space.sample()

        # Apply evader action
        env.set_target_action(evader_action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(pursuer_action)

        if step == 0:
            print(f"   - Step {step + 1}:")
            print(f"     * Reward: {reward:.2f}")
            print(f"     * Distance: {info['distance_3d']:.2f}")
            print(f"     * In Focus: {info['in_focus']}")

        if terminated or truncated:
            break

    print(f"   ✓ Simulation ran successfully ({step + 1} steps)")
    print()

    # 5. Test target observation
    print("5. Testing target observation...")
    target_obs = get_target_observation(env)
    print(f"   ✓ Target observation shape: {target_obs.shape}")
    print()

    # 6. Check directories
    print("6. Checking directory structure...")
    required_dirs = ['models', 'environment', 'agents', 'training', 'testing', 'utils']
    all_exist = True
    for dir_name in required_dirs:
        exists = os.path.exists(dir_name)
        status = "✓" if exists else "✗"
        print(f"   {status} {dir_name}/")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n   Creating missing directories...")
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
        print("   ✓ Directories created")
    print()

    # Summary
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print()
    print("✓ All components are working correctly!")
    print()
    print("Next steps:")
    print("  1. Kaggle training:")
    print("     python training/train_kaggle.py --rounds 50 --steps-per-round 10000")
    print()
    print("  2. Local testing (after training):")
    print("     python testing/test_with_render.py --episodes 5")
    print()
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    try:
        verify_setup()
    except Exception as e:
        print(f"\n✗ ERROR during verification:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
