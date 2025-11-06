#!/usr/bin/env python3
"""
🚀 QUICK TRAIN: Click and Run!

One-click training script for SAC agent on 3D pursuit-evasion task.

This script:
- Uses the new reward system (focus-based rewards)
- Trains on 3D environment with agile targets
- Saves best model automatically
- Logs to tensorboard

Usage:
    python quick_train.py              # Quick training (50k steps)
    python quick_train.py --full       # Full training (500k steps)
    python quick_train.py --test       # Test run (5k steps)

Tensorboard:
    tensorboard --logdir logs/sac
"""

import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.train_sac import train_sac, load_config


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Quick Train SAC Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_train.py              # Quick train (50k steps, ~10 min)
  python quick_train.py --full       # Full train (500k steps, ~2 hours)
  python quick_train.py --test       # Test run (5k steps, ~1 min)

Monitor training:
  tensorboard --logdir logs/sac
        """
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full training (500k steps instead of 50k)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test run (5k steps for quick testing)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Load config
    print("=" * 70)
    print("🚀 QUICK TRAIN: SAC on 3D Pursuit-Evasion")
    print("=" * 70)
    print()

    config = load_config(args.config)

    # Adjust total_timesteps based on mode
    if args.test:
        config["sac"]["total_timesteps"] = 5000
        config["sac"]["learning_starts"] = 1000
        config["sac"]["eval_freq"] = 1000
        mode_name = "TEST"
        print("🧪 TEST MODE: 5k steps (~1 minute)")
    elif args.full:
        config["sac"]["total_timesteps"] = 500000
        mode_name = "FULL"
        print("🏋️ FULL TRAINING: 500k steps (~2 hours)")
    else:
        config["sac"]["total_timesteps"] = 50000
        config["sac"]["learning_starts"] = 5000
        config["sac"]["eval_freq"] = 5000
        mode_name = "QUICK"
        print("⚡ QUICK TRAINING: 50k steps (~10 minutes)")

    print()
    print("📋 Configuration:")
    print(f"   - Environment: 3D Pursuit-Evasion with agile targets")
    print(f"   - Reward System: Focus-based (+0.1/step in focus, +10 for 5sec)")
    print(f"   - Agent Max Speed: {config['environment']['max_velocity']}")
    print(f"   - Target Max Speed: {config['environment']['max_velocity'] * config['environment']['target_max_speed_ratio']:.1f}")
    print(f"   - Training Steps: {config['sac']['total_timesteps']:,}")
    print(f"   - Random Seed: {args.seed}")
    print()
    print("📁 Output:")
    print(f"   - Models: models/sac_{mode_name.lower()}/")
    print(f"   - Logs: logs/sac_{mode_name.lower()}/")
    print()
    print("🎯 Goal: Train agent to keep target in focus area for 5 seconds")
    print()
    print("=" * 70)
    print()

    # Train
    save_dir = f"models/sac_{mode_name.lower()}"
    tensorboard_log = f"logs/sac_{mode_name.lower()}"

    try:
        model = train_sac(
            config=config,
            save_dir=save_dir,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
        )

        print()
        print("=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print()
        print("📦 Model saved to:", save_dir)
        print("📊 View training progress:")
        print(f"   tensorboard --logdir {tensorboard_log}")
        print()
        print("🎮 Test trained model:")
        print(f"   python test_trained_model.py --model {save_dir}/best_model/best_model.zip")
        print()

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⚠️  Training interrupted by user")
        print("=" * 70)
        print()
        print("💾 Checkpoints saved in:", save_dir)
        sys.exit(0)
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR DURING TRAINING")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
