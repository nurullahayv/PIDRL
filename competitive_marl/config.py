"""
Configuration for Competitive MARL Training

Optimized for Kaggle GPU training with local testing support.
"""

# Environment Configuration
ENV_CONFIG = {
    # 3D Space
    "view_size": 30.0,          # FOV size in world units
    "frame_size": 64,           # Observation resolution
    "depth_range": [10.0, 50.0],  # Min/max depth for rendering

    # Success Criteria (30% focus area)
    "success_threshold": 9.0,    # 30% of view_size
    "target_size": 7.0,          # Target size in world units (INCREASED for easier lock-on)
    "agent_size": 1.5,

    # Physics
    "max_velocity": 60.0,        # Max velocity (units/s) - INCREASED for faster action
    "max_acceleration": 6.0,     # Max acceleration (units/s²) - INCREASED
    "max_angular_velocity": 3.5, # Max turning rate (rad/s) - INCREASED
    "dt": 0.1,                   # Time step

    # Episode
    "max_steps": 1000,           # Max steps per episode

    # Rewards
    "focus_reward": 0.1,         # Per-step reward in focus
    "focus_bonus": 10.0,         # Bonus for 5 seconds in focus
    "escape_penalty": -2.0,      # Penalty for escaping near completion
    "outside_penalty_scale": 0.01,
    "focus_time_threshold": 50,  # Steps (5 seconds at dt=0.1)
}

# Pursuer Agent Configuration
PURSUER_CONFIG = {
    "policy_type": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "net_arch": [256, 256],
}

# Evader Agent Configuration
EVADER_CONFIG = {
    "policy_type": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "net_arch": [256, 256],
}

# Training Configuration
TRAINING_CONFIG = {
    # Training modes
    "mode": "alternating",  # "alternating", "simultaneous"

    # Alternating training
    "pursuer_steps_per_round": 10000,   # Steps to train pursuer
    "evader_steps_per_round": 10000,    # Steps to train evader
    "num_rounds": 50,                   # Number of alternating rounds

    # Simultaneous training
    "total_timesteps": 1000000,         # Total steps (if simultaneous)

    # Evaluation
    "eval_freq": 5000,
    "n_eval_episodes": 10,

    # Saving
    "save_freq": 10000,
    "checkpoint_dir": "models/checkpoints",

    # Kaggle specific
    "use_gpu": True,
    "verbose": 1,
}

# Rendering Configuration (for local testing only)
RENDER_CONFIG = {
    "mode": "human",         # "human" or "rgb_array"
    "fps": 30,
    "screen_size": (800, 800),  # HUD screen size
    "render_during_training": False,  # Always False for Kaggle
}

# Paths
PATHS = {
    "pursuer_model": "models/pursuer_latest.zip",
    "evader_model": "models/evader_latest.zip",
    "pursuer_best": "models/pursuer_best.zip",
    "evader_best": "models/evader_best.zip",
    "tensorboard_log": "./logs/tensorboard/",
}


def get_config():
    """Get complete configuration dictionary."""
    return {
        "env": ENV_CONFIG,
        "pursuer": PURSUER_CONFIG,
        "evader": EVADER_CONFIG,
        "training": TRAINING_CONFIG,
        "render": RENDER_CONFIG,
        "paths": PATHS,
    }


def print_config():
    """Print configuration summary."""
    config = get_config()

    print("=" * 70)
    print("COMPETITIVE MARL CONFIGURATION")
    print("=" * 70)
    print()

    print("Environment:")
    print(f"  View Size: {config['env']['view_size']}")
    print(f"  Frame Size: {config['env']['frame_size']}")
    print(f"  Success Threshold: {config['env']['success_threshold']} (30%)")
    print(f"  Target Size: {config['env']['target_size']}")
    print(f"  Max Steps: {config['env']['max_steps']}")
    print()

    print("Training:")
    print(f"  Mode: {config['training']['mode']}")
    if config['training']['mode'] == 'alternating':
        print(f"  Pursuer Steps/Round: {config['training']['pursuer_steps_per_round']}")
        print(f"  Evader Steps/Round: {config['training']['evader_steps_per_round']}")
        print(f"  Total Rounds: {config['training']['num_rounds']}")
    else:
        print(f"  Total Timesteps: {config['training']['total_timesteps']}")
    print(f"  Use GPU: {config['training']['use_gpu']}")
    print()

    print("Models:")
    print(f"  Pursuer: {config['paths']['pursuer_model']}")
    print(f"  Evader: {config['paths']['evader_model']}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print_config()
