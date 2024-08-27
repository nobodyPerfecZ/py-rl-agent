from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from pyrlagent.torch.network import (
    CNNCategoricalActorCriticNetwork,
    CNNGaussianActorCriticNetwork,
    MLPCategoricalActorCriticNetwork,
    MLPGaussianActorCriticNetwork,
)


@dataclass
class NetworkConfig:
    """Configuration of the neural network."""

    network_type: str
    network_kwargs: dict[str, Any]


def create_network(
    network_config: NetworkConfig,
    obs_dim: int,
    act_dim: int,
) -> nn.Module:
    """
    Create the neural network for the actor-critic agent.

    Args:
        network_config (NetworkConfig):
            The configuration of the neural network

        obs_dim (int):
            The dimension of the observation space

        act_dim (int):
            The dimension of the action space

    Returns:
        nn.Module:
            The neural network
    """
    if network_config.network_type == "cnn-discrete":
        network = CNNCategoricalActorCriticNetwork
    elif network_config.network_type == "mlp-discrete":
        network = MLPCategoricalActorCriticNetwork
    elif network_config.network_type == "cnn-continuous":
        network = CNNGaussianActorCriticNetwork
    elif network_config.network_type == "mlp-continuous":
        network = MLPGaussianActorCriticNetwork
    else:
        raise ValueError(f"Invalid network type: {network_config.network_type}.")

    return network(
        obs_dim=obs_dim,
        act_dim=act_dim,
        **network_config.network_kwargs,
    )
