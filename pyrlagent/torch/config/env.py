from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from pyrlagent.torch.util import get_env, get_vector_env


@dataclass
class EnvConfig:
    """Configuration of the RL environment."""

    id: str
    kwargs: dict[str, Any]


def create_env_train(
    env_config: EnvConfig, num_envs: int, device: str
) -> gym.vector.VectorEnv:
    """
    Create the Gymnasium environment for training.

    Args:
        env_config (EnvConfig):
            The configuration of the environment

        num_envs (int):
            The number of environments to run in parallel

        device (str):
            The device to run the PyTorch computation

    Returns:
        gym.Env:
            The Gymnasium environment
    """
    # Get the vectorized environment
    env = get_vector_env(
        env_id=env_config.id,
        num_envs=num_envs,
        device=device,
        render_mode=None,
        **env_config.kwargs,
    )
    return env


def create_env_eval(env_config: EnvConfig, device: str) -> gym.Env:
    """
    Create the Gymnasium environment for evaluation.

    Args:
        env_config (EnvConfig):
            The configuration of the environment

        device (str):
            The device to run the PyTorch computation

    Returns:
        gym.Env:
            The Gymnasium environment
    """
    env = get_env(
        env_id=env_config.id,
        device=device,
        render_mode="human",
        **env_config.kwargs,
    )

    return env
