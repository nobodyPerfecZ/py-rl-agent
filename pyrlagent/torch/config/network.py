from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch.nn as nn

from pyrlagent.torch.network import (
    CNNContinuousPGActorCriticNetwork,
    CNNDiscretePGActorCriticNetwork,
    MLPContinuousDDPGActorCriticNetwork,
    MLPContinuousPGActorCriticNetwork,
    MLPDiscretePGActorCriticNetwork,
)


@dataclass
class NetworkConfig:
    """Configuration of the neural network."""
    method: str | None = None
    id: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


def create_network_id(
    method: str,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> str:
    """
    Creates the string ID of the neural network, given the RL method and the environment spaces.

    Args:
        method (str):
            The string identifier of the RL method

        obs_space (gym.spaces.Space):
            The observation space of the environment

        action_space (gym.spaces.Space):
            The action space of the environment

    Returns:
        str:
            The string ID of the neural network
    """
    # Check the method
    if method not in ["pg", "ddpg"]:
        raise ValueError(f"Invalid method: {method}.")
    method_id = method

    # Check the architecture
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
        obs_space_id = "mlp"
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
        obs_space_id = "cnn"
    else:
        raise ValueError(f"Invalid observation space: {obs_space}.")

    # Check the action space
    if isinstance(action_space, gym.spaces.Discrete):
        action_space_id = "discrete"
    elif isinstance(action_space, gym.spaces.Box):
        action_space_id = "continuous"
    else:
        raise ValueError(f"Invalid action space: {action_space}.")

    return f"{method_id}-{obs_space_id}-{action_space_id}"


def create_network(
    network_config: NetworkConfig,
    obs_dim: int | tuple[int, int, int],
    act_dim: int | tuple[int, int, int],
) -> nn.Module:
    """
    Create the neural network for the actor-critic agent.

    Args:
        network_config (NetworkConfig):
            The configuration of the neural network

        obs_dim (int | tuple[int, int, int]):
            The dimensions of the observation space

        act_dim (int | tuple[int, int, int]):
            The dimensions of the action space

    Returns:
        nn.Module:
            The neural network
    """
    # 1. Networks for Deep Deterministic Policy Gradient (DDPG) methods
    if network_config.id == "ddpg-mlp-continuous":
        network = MLPContinuousDDPGActorCriticNetwork
    # 2. Networks for Policy Gradient (PG) methods
    elif network_config.id == "pg-cnn-discrete":
        network = CNNDiscretePGActorCriticNetwork
    elif network_config.id == "pg-mlp-discrete":
        network = MLPDiscretePGActorCriticNetwork
    elif network_config.id == "pg-cnn-continuous":
        network = CNNContinuousPGActorCriticNetwork
    elif network_config.id == "pg-mlp-continuous":
        network = MLPContinuousPGActorCriticNetwork
    else:
        raise ValueError(f"Invalid network type: {network_config.id}.")

    return network(
        obs_dim=obs_dim,
        act_dim=act_dim,
        **network_config.kwargs,
    )
