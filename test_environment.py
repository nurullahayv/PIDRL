"""
Quick test script to verify environment and components.

This script tests:
1. Environment creation and basic functionality
2. Frame stacking
3. Visual detection
4. PID controller
5. Kalman Filter
"""

import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_environment():
    """Test basic environment functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: Environment Creation and Basic Functionality")
    print("=" * 60)

    from environments import make_env

    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create environment
    env = make_env(config)
    print("✓ Environment created successfully")

    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation dtype: {obs.dtype}")
    print(f"  - Info keys: {list(info.keys())}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  - Reward: {reward:.3f}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")

    # Test multiple steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    print("✓ Multiple steps completed successfully")

    env.close()
    print("\n✅ Environment test PASSED")


def test_visual_detection():
    """Test visual detection module."""
    print("\n" + "=" * 60)
    print("TEST 2: Visual Detection")
    print("=" * 60)

    from utils.visual_detection import VisualDetector
    from environments import make_env

    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create environment
    env = make_env(config)
    obs, info = env.reset()

    # Create detector
    detector = VisualDetector(frame_size=64)
    print("✓ Visual detector created")

    # Test detection on stacked frames
    position, confidence = detector.detect_target_stacked(obs)

    if position is not None:
        print(f"✓ Target detected successfully")
        print(f"  - Position: ({position[0]:.2f}, {position[1]:.2f})")
        print(f"  - Confidence: {confidence:.2f}")
    else:
        print("✓ Detection returned None (acceptable if target not in view)")

    # Test multiple frames
    detection_count = 0
    for _ in range(20):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        position, confidence = detector.detect_target_stacked(obs)
        if position is not None:
            detection_count += 1

    detection_rate = detection_count / 20
    print(f"✓ Detection rate over 20 frames: {detection_rate:.1%}")

    env.close()
    print("\n✅ Visual detection test PASSED")


def test_pid_controller():
    """Test PID controller."""
    print("\n" + "=" * 60)
    print("TEST 3: PID Controller")
    print("=" * 60)

    from controllers import PIDAgent
    from environments import make_env

    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create environment and agent
    env = make_env(config)
    agent = PIDAgent(config)
    print("✓ PID agent created")

    # Reset
    obs, info = env.reset()
    agent.reset()

    # Run episode
    episode_reward = 0
    for step in range(50):
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"✓ PID controller ran for {step + 1} steps")
    print(f"  - Episode reward: {episode_reward:.2f}")

    # Check stats
    stats = agent.get_stats()
    print(f"✓ PID statistics:")
    print(f"  - Detection rate: {stats['detection_rate']:.1%}")
    print(f"  - Total steps: {stats['total_steps']}")

    env.close()
    print("\n✅ PID controller test PASSED")


def test_kalman_filter():
    """Test Kalman Filter."""
    print("\n" + "=" * 60)
    print("TEST 4: Kalman Filter")
    print("=" * 60)

    from controllers.kalman_filter import KalmanFilter

    # Create Kalman Filter
    kf = KalmanFilter(dt=0.1)
    print("✓ Kalman Filter created")

    # Test initialization
    measurement = np.array([10.0, 5.0])
    kf.initialize(measurement)
    print(f"✓ Kalman Filter initialized with measurement: {measurement}")

    # Test prediction
    predicted_state = kf.predict()
    print(f"✓ Prediction step: state = {predicted_state}")

    # Test update
    new_measurement = np.array([10.5, 5.2])
    updated_state = kf.update(new_measurement)
    print(f"✓ Update step: state = {updated_state}")

    # Test multiple steps with noisy measurements
    print("\n  Testing with noisy measurements:")
    true_position = np.array([0.0, 0.0])
    true_velocity = np.array([1.0, 0.5])

    for i in range(5):
        # Simulate true motion
        true_position += true_velocity * 0.1

        # Add measurement noise
        noisy_measurement = true_position + np.random.normal(0, 0.5, 2)

        # Kalman filter step
        state = kf.step(noisy_measurement)

        estimated_position = state[0:2]
        error = np.linalg.norm(estimated_position - true_position)
        print(f"    Step {i+1}: estimation error = {error:.3f}")

    print("✓ Kalman Filter tracking working")

    print("\n✅ Kalman Filter test PASSED")


def test_kalman_pid_controller():
    """Test Kalman-PID controller."""
    print("\n" + "=" * 60)
    print("TEST 5: Kalman-PID Controller")
    print("=" * 60)

    from controllers import KalmanPIDAgent
    from environments import make_env

    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create environment and agent
    env = make_env(config)
    agent = KalmanPIDAgent(config)
    print("✓ Kalman-PID agent created")

    # Reset
    obs, info = env.reset()
    agent.reset()

    # Run episode
    episode_reward = 0
    for step in range(50):
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"✓ Kalman-PID controller ran for {step + 1} steps")
    print(f"  - Episode reward: {episode_reward:.2f}")

    # Check stats
    stats = agent.get_stats()
    print(f"✓ Kalman-PID statistics:")
    print(f"  - Detection rate: {stats['detection_rate']:.1%}")
    print(f"  - Total steps: {stats['total_steps']}")

    env.close()
    print("\n✅ Kalman-PID controller test PASSED")


def test_networks():
    """Test neural network architectures."""
    print("\n" + "=" * 60)
    print("TEST 6: Neural Network Architectures")
    print("=" * 60)

    import torch
    from gymnasium import spaces
    from agents.networks import CNNFeatureExtractor

    # Create dummy observation space
    obs_space = spaces.Box(low=0, high=255, shape=(4, 64, 64), dtype=np.uint8)
    print(f"✓ Observation space: {obs_space.shape}")

    # Create CNN feature extractor
    cnn = CNNFeatureExtractor(obs_space, features_dim=256)
    print("✓ CNN feature extractor created")

    # Test forward pass
    dummy_obs = torch.randint(0, 255, (8, 4, 64, 64), dtype=torch.uint8)
    features = cnn(dummy_obs)

    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {dummy_obs.shape}")
    print(f"  - Output shape: {features.shape}")
    print(f"  - Feature dim: {cnn.features_dim}")

    assert features.shape == (8, 256), "Unexpected feature shape"

    print("\n✅ Neural network test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL COMPONENT TESTS")
    print("=" * 60)

    try:
        test_environment()
        test_visual_detection()
        test_pid_controller()
        test_kalman_filter()
        test_kalman_pid_controller()
        test_networks()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nYour environment is ready for experiments!")
        print("\nNext steps:")
        print("  1. Run demo: python demo.py pid")
        print("  2. Train SAC: python experiments/train_sac.py")
        print("  3. Evaluate: python experiments/compare_methods.py")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
