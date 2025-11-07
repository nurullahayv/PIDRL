"""
3D HUD Demo: Two Training Scenarios

Scenario 1: RL Agent vs Random Target
- Agent learns to track a randomly moving target
- Target moves with Brownian motion + evasive maneuvers

Scenario 2: Competitive MARL (RL vs RL)
- Both pursuer and evader are RL agents
- Pursuer learns to catch, evader learns to escape
- Zero-sum competitive game

Usage:
    # Scenario 1: Random target
    python demo_3d.py --scenario 1 --algo ppo --model models/pursuer_ppo.zip

    # Scenario 2: Competitive MARL
    python demo_3d.py --scenario 2 --pursuer-algo sac --evader-algo ppo \
                      --pursuer-model models/pursuer_sac.zip \
                      --evader-model models/evader_ppo.zip

    # Random actions (baseline)
    python demo_3d.py --scenario 1 --random
"""

import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from environments import make_env


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_rl_agent(algo: str, model_path: str):
    """
    Load trained RL agent.

    Args:
        algo: Algorithm name (dqn, ppo, sac, td3)
        model_path: Path to saved model

    Returns:
        Loaded model
    """
    if algo.lower() == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo.lower() == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo.lower() == "sac":
        from stable_baselines3 import SAC
        return SAC.load(model_path)
    elif algo.lower() == "td3":
        from stable_baselines3 import TD3
        return TD3.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from: dqn, ppo, sac, td3")


def run_scenario_1(
    config: dict,
    n_episodes: int = 5,
    algo: str = None,
    model_path: str = None,
    use_random: bool = False,
):
    """
    Run Scenario 1: RL Agent vs Random Target

    Args:
        config: Configuration dictionary
        n_episodes: Number of episodes to run
        algo: RL algorithm (dqn, ppo, sac, td3)
        model_path: Path to trained model
        use_random: Use random actions (baseline)
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO 1: RL AGENT vs RANDOM TARGET")
    print(f"{'='*60}")

    if use_random:
        print(f"Agent: RANDOM ACTIONS (Baseline)")
        agent = None
    else:
        print(f"Agent: {algo.upper()} (Trained)")
        agent = load_rl_agent(algo, model_path)

    print(f"Target: RANDOM MOVEMENT (Brownian + Evasion)")
    print(f"Initializing 3D environment...")
    print(f"\nClose the window to stop the demo.\n")

    # Create environment with rendering
    env = make_env(config, render_mode="human", use_3d=True)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")

        while not done:
            # Get action
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Render
            env.render()

            # Print info periodically
            if episode_length % 50 == 0:
                distance = info.get("distance_3d", -1)
                in_focus = info.get("in_focus", False)
                print(f"  Step {episode_length}: Distance={distance:.2f}, In Focus={in_focus}")

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Final Distance: {info.get('distance_3d', -1):.2f}")

    env.close()
    print("\nDemo complete!")


def run_scenario_2(
    config: dict,
    n_episodes: int = 5,
    pursuer_algo: str = None,
    evader_algo: str = None,
    pursuer_model: str = None,
    evader_model: str = None,
    use_random_pursuer: bool = False,
    use_random_evader: bool = False,
):
    """
    Run Scenario 2: Competitive MARL (RL vs RL)

    Args:
        config: Configuration dictionary
        n_episodes: Number of episodes to run
        pursuer_algo: Pursuer RL algorithm
        evader_algo: Evader RL algorithm
        pursuer_model: Path to pursuer model
        evader_model: Path to evader model
        use_random_pursuer: Use random actions for pursuer
        use_random_evader: Use random actions for evader
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO 2: COMPETITIVE MARL (RL vs RL)")
    print(f"{'='*60}")

    if use_random_pursuer:
        print(f"Pursuer: RANDOM ACTIONS")
        pursuer_agent = None
    else:
        print(f"Pursuer: {pursuer_algo.upper()} (Trained)")
        pursuer_agent = load_rl_agent(pursuer_algo, pursuer_model)

    if use_random_evader:
        print(f"Evader: RANDOM ACTIONS")
        evader_agent = None
    else:
        print(f"Evader: {evader_algo.upper()} (Trained)")
        evader_agent = load_rl_agent(evader_algo, evader_model)

    print(f"Initializing 3D competitive environment...")
    print(f"\nClose the window to stop the demo.\n")

    # Create environment with rendering (competitive mode)
    # TODO: Need to implement evader control in environment
    # For now, use scenario 1 with RL agent
    print("NOTE: Full competitive MARL demo requires evader environment integration.")
    print("Falling back to Scenario 1 for now...")

    run_scenario_1(
        config=config,
        n_episodes=n_episodes,
        algo=pursuer_algo,
        model_path=pursuer_model,
        use_random=use_random_pursuer,
    )


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="3D HUD demo with two training scenarios"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2],
        default=1,
        help="Scenario: 1 (RL vs Random) or 2 (RL vs RL)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions (baseline)",
    )

    # Scenario 1 arguments
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo", "sac", "td3"],
        help="RL algorithm for Scenario 1",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model for Scenario 1",
    )

    # Scenario 2 arguments
    parser.add_argument(
        "--pursuer-algo",
        type=str,
        choices=["dqn", "ppo", "sac", "td3"],
        help="Pursuer RL algorithm for Scenario 2",
    )
    parser.add_argument(
        "--evader-algo",
        type=str,
        choices=["dqn", "ppo", "sac", "td3"],
        help="Evader RL algorithm for Scenario 2",
    )
    parser.add_argument(
        "--pursuer-model",
        type=str,
        help="Path to pursuer model for Scenario 2",
    )
    parser.add_argument(
        "--evader-model",
        type=str,
        help="Path to evader model for Scenario 2",
    )
    parser.add_argument(
        "--random-pursuer",
        action="store_true",
        help="Use random pursuer for Scenario 2",
    )
    parser.add_argument(
        "--random-evader",
        action="store_true",
        help="Use random evader for Scenario 2",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run selected scenario
    if args.scenario == 1:
        if not args.random and (args.algo is None or args.model is None):
            print("ERROR: Scenario 1 requires --algo and --model (or use --random)")
            return

        run_scenario_1(
            config=config,
            n_episodes=args.n_episodes,
            algo=args.algo,
            model_path=args.model,
            use_random=args.random,
        )

    elif args.scenario == 2:
        # Check arguments
        if not args.random_pursuer and (args.pursuer_algo is None or args.pursuer_model is None):
            print("ERROR: Scenario 2 requires --pursuer-algo and --pursuer-model (or --random-pursuer)")
            return

        if not args.random_evader and (args.evader_algo is None or args.evader_model is None):
            print("ERROR: Scenario 2 requires --evader-algo and --evader-model (or --random-evader)")
            return

        run_scenario_2(
            config=config,
            n_episodes=args.n_episodes,
            pursuer_algo=args.pursuer_algo,
            evader_algo=args.evader_algo,
            pursuer_model=args.pursuer_model,
            evader_model=args.evader_model,
            use_random_pursuer=args.random_pursuer,
            use_random_evader=args.random_evader,
        )


if __name__ == "__main__":
    main()
