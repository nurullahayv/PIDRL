#!/usr/bin/env python3
"""
Kaggle Setup Script for PIDRL

Automatically sets up the environment for training on Kaggle with GPU support.

Usage (in Kaggle notebook):
    !git clone https://github.com/nurullahayv/PIDRL.git
    %cd PIDRL
    !python setup_kaggle.py

Then train:
    !python quick_train.py --full
"""

import os
import sys
import subprocess


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ GPU Available!")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No GPU found. Training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed")
        return False


def install_dependencies():
    """Install required packages."""
    print("\n" + "="*70)
    print("📦 INSTALLING DEPENDENCIES")
    print("="*70)

    # Upgrade pip first
    print("\n1. Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Install from requirements.txt
    print("\n2. Installing from requirements.txt...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    # Install additional packages for Kaggle
    print("\n3. Installing additional packages...")
    additional_packages = [
        "torch",
        "stable-baselines3[extra]",
        "tensorboard",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + additional_packages, check=True)

    print("\n✓ All dependencies installed!")


def setup_directories():
    """Create necessary directories."""
    print("\n" + "="*70)
    print("📁 CREATING DIRECTORIES")
    print("="*70)

    directories = [
        "models",
        "models/sac_kaggle",
        "logs",
        "logs/sac_kaggle",
        "results",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")

    print("\n✓ Directories created!")


def test_environment():
    """Quick test that environment works."""
    print("\n" + "="*70)
    print("🧪 TESTING ENVIRONMENT")
    print("="*70)

    try:
        import yaml
        import numpy as np
        from environments import make_env

        # Load config
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Create environment
        print("\n  Creating 3D environment...")
        env = make_env(config, use_3d=True)

        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Action space: {env.action_space.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Environment step")
        print(f"    Agent reward: {info['agent_reward']:.3f}")
        print(f"    Target reward: {info['target_reward']:.3f}")

        env.close()
        print("\n✓ Environment test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        return False


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("🚀 SETUP COMPLETE - READY TO TRAIN!")
    print("="*70)
    print("\n📋 Next steps:")
    print("\n1. Quick Test (5k steps, ~1 min):")
    print("   !python quick_train.py --test")
    print("\n2. Full Training (500k steps, ~2 hours with GPU):")
    print("   !python quick_train.py --full")
    print("\n3. Monitor with TensorBoard (in another cell):")
    print("   %load_ext tensorboard")
    print("   %tensorboard --logdir logs/sac_kaggle")
    print("\n4. Test trained model:")
    print("   !python test_trained_model.py --model models/sac_kaggle/best_model/best_model.zip --no-render")
    print("\n" + "="*70)
    print()


def main():
    """Main setup function."""
    print("\n" + "="*70)
    print("🎯 KAGGLE SETUP FOR PIDRL")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check GPU availability")
    print("  2. Install all dependencies")
    print("  3. Create necessary directories")
    print("  4. Test the environment")
    print()

    # Check GPU
    has_gpu = check_gpu()

    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nTry installing manually:")
        print("  !pip install -r requirements.txt")
        print("  !pip install torch stable-baselines3 tensorboard")
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Test environment
    test_ok = test_environment()

    if not test_ok:
        print("\n⚠️  Environment test failed, but you can still try training")

    # Print usage
    print_usage()

    # GPU reminder
    if not has_gpu:
        print("💡 TIP: Enable GPU in Kaggle:")
        print("   Settings → Accelerator → GPU")
        print()


if __name__ == "__main__":
    main()
