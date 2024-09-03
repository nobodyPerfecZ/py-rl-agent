from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from pyrlagent.torch.network import (
    MLPContinuousDDPGActorCriticNetwork,
    CNNDiscretePGActorCriticNetwork,
    CNNContinuousPGActorCriticNetwork,
    MLPDiscretePGActorCriticNetwork,
    MLPContinuousPGActorCriticNetwork,
)


@dataclass
class NetworkConfig:
    """Configuration of the neural network."""

    id: str
    kwargs: dict[str, Any]


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
