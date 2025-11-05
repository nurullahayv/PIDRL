"""
Setup script for PIDRL research project.

This script helps initialize the project environment.
"""

import os
import sys
import subprocess


def create_directories():
    """Create necessary directories for experiments."""
    directories = [
        "models/sac/best_model",
        "models/sac/checkpoints",
        "logs/sac",
        "results/figures",
        "results/pid",
        "results/kalman-pid",
        "results/sac",
    ]

    print("Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "gymnasium",
        "numpy",
        "pygame",
        "cv2",
        "torch",
        "stable_baselines3",
        "matplotlib",
        "seaborn",
        "yaml",
        "pandas",
        "tqdm",
    ]

    print("\nChecking dependencies...")
    missing = []

    for package in required_packages:
        try:
            if package == "cv2":
                __import__("cv2")
            elif package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def run_tests():
    """Run test suite to verify installation."""
    print("\nRunning test suite...")
    try:
        result = subprocess.run(
            [sys.executable, "test_environment.py"],
            check=True,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("PIDRL Research Project Setup")
    print("=" * 60)

    # Create directories
    create_directories()

    # Check dependencies
    deps_ok = check_dependencies()

    if not deps_ok:
        print("\n⚠️  Please install missing dependencies first:")
        print("    pip install -r requirements.txt")
        return

    # Run tests
    print("\n" + "=" * 60)
    tests_ok = run_tests()

    if tests_ok:
        print("\n" + "=" * 60)
        print("🎉 Setup Complete! 🎉")
        print("=" * 60)
        print("\nYour environment is ready for research!")
        print("\nQuick Start Commands:")
        print("  1. Demo PID:        python demo.py pid")
        print("  2. Demo Kalman-PID: python demo.py kalman-pid")
        print("  3. Train SAC:       python experiments/train_sac.py")
        print("  4. Evaluate All:    python experiments/compare_methods.py")
        print("\nFor more information, see README.md")
    else:
        print("\n⚠️  Setup completed with warnings. Check test output above.")


if __name__ == "__main__":
    main()
