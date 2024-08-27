from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from pyrlagent.torch.util.env import get_env, get_vector_env


@dataclass
class EnvConfig:
    """Configuration of the RL environment."""

    env_type: str
    env_kwargs: dict[str, Any]


def create_env_train(env_config: EnvConfig, num_envs: int, device: str) -> gym.Env:
    """
    Create the Gymnasium environment for training.

    Args:
        env_config (EnvConfig):
            The configuration of the environment

        num_envs (int):
            The number of environments to run in parallel

    Returns:
        gym.Env:
            The Gymnasium environment
    """
    # Get the vectorized environment
    env = get_vector_env(
        env_id=env_config.env_type,
        num_envs=num_envs,
        device=device,
        render_mode=None,
        **env_config.env_kwargs,
    )
    return env


def create_env_eval(env_config: EnvConfig, device: str) -> gym.Env:
    """
    Create the Gymnasium environment for evaluation.

    Args:
        env_config (EnvConfig):
            The configuration of the environment

    Returns:
        gym.Env:
            The Gymnasium environment
    """
    env = get_env(
        env_id=env_config.env_type,
        device=device,
        render_mode="human",
        **env_config.env_kwargs,
    )

    return env
