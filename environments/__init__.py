"""Environments package for pursuit-evasion simulation."""

# Handle different gymnasium versions
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    try:
        from gymnasium.wrappers.frame_stack import FrameStack
    except ImportError:
        # Fallback: use gym's FrameStack or implement our own
        try:
            from gym.wrappers import FrameStack
        except ImportError:
            # If all else fails, we'll implement a simple version
            from gymnasium import Wrapper
            from collections import deque
            import numpy as np

            class FrameStack(Wrapper):
                """Simple FrameStack implementation for compatibility."""
                def __init__(self, env, num_stack):
                    super().__init__(env)
                    self.num_stack = num_stack
                    self.frames = deque(maxlen=num_stack)

                    # Update observation space
                    low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
                    high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
                    self.observation_space = type(env.observation_space)(
                        low=low, high=high, dtype=env.observation_space.dtype
                    )

                def reset(self, **kwargs):
                    obs, info = self.env.reset(**kwargs)
                    for _ in range(self.num_stack):
                        self.frames.append(obs)
                    return self._get_obs(), info

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.frames.append(obs)
                    return self._get_obs(), reward, terminated, truncated, info

                def _get_obs(self):
                    return np.array(list(self.frames))

from environments.pursuit_evasion_env import PursuitEvasionEnv


def make_env(config: dict, render_mode=None):
    """
    Create and configure the pursuit-evasion environment.

    Args:
        config: Dictionary containing environment configuration
        render_mode: Rendering mode ('human', 'rgb_array', or None)

    Returns:
        Wrapped Gymnasium environment with frame stacking
    """
    env_config = config.get("environment", {})

    # Create base environment
    env = PursuitEvasionEnv(
        frame_size=env_config.get("frame_size", 64),
        dt=env_config.get("dt", 0.1),
        max_velocity=env_config.get("max_velocity", 10.0),
        max_acceleration=env_config.get("max_acceleration", 1.0),
        friction=env_config.get("friction", 0.95),
        world_size=env_config.get("world_size", 100.0),
        view_radius=env_config.get("view_radius", 30.0),
        target_brownian_std=env_config.get("target_brownian_std", 2.0),
        target_size=env_config.get("target_size", 2.0),
        agent_size=env_config.get("agent_size", 1.5),
        max_steps=env_config.get("max_steps", 500),
        success_threshold=env_config.get("success_threshold", 5.0),
        reward_scale=env_config.get("reward_scale", 0.01),
        render_mode=render_mode,
    )

    # Apply frame stacking wrapper
    frame_stack = env_config.get("frame_stack", 4)
    env = FrameStack(env, num_stack=frame_stack)

    return env


__all__ = ["PursuitEvasionEnv", "make_env"]
