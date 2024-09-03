from dataclasses import dataclass
from typing import Any

from pyrlagent.torch.buffer import Buffer, ReplayBuffer, RolloutBuffer


@dataclass
class BufferConfig:
    """Configuration of the RL Buffer."""

    id: str
    kwargs: dict[str, Any]


def create_buffer(
    buffer_config: BufferConfig,
    obs_dim: int,
    act_dim: int,
    env_dim: int,
    max_size: int,
    device: str,
) -> Buffer:
    """
    Creates the buffer for storing the experiences for training.

    Args:
        buffer_config (BufferConfig):
            The configuration of the buffer

        obs_dim (int):
            The dimension of the observation space

        act_dim (int):
            The dimension of the action space

        env_dim (int):
            The number of parallel environments

        max_size (int):
            The maximum size of the buffer

        device (str):
            The device to run the PyTorch computation

    Returns:
        Buffer:
            The buffer for storing the experiences for training
    """
    if buffer_config.id == "rollout":
        buffer = RolloutBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            env_dim=env_dim,
            max_size=max_size,
            device=device,
            **buffer_config.kwargs,
        )
    elif buffer_config.id == "replay":
        buffer = ReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            env_dim=env_dim,
            max_size=max_size,
            device=device,
            **buffer_config.kwargs,
        )
    else:
        raise ValueError(f"Invalid buffer type: {buffer_config.id}.")
    return buffer
