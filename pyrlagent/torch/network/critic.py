from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from pyrlagent.torch.util import cnn, mlp


class CriticNetwork(nn.Module, ABC):
    """Base class for a critic network."""

    @abstractmethod
    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the value of the observation.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            torch.Tensor:
                The value of the observation.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the critic network.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            torch.Tensor:
                The value of the observation
        """
        x = x.to(dtype=torch.float32)
        return self.critic_value(x)


class MLPCriticNetwork(CriticNetwork):
    """MLP for the critic network."""

    def __init__(self, obs_dim: int, hidden_features: list[int], activation: nn.Module):
        super().__init__()
        self.critic_net = mlp(
            in_features=obs_dim,
            hidden_features=hidden_features,
            out_features=1,
            activation=activation,
        )

    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        return self.critic_net(x).squeeze(dim=-1)


class CNNCriticNetwork(CriticNetwork):
    """CNN for the critic network."""

    def __init__(
        self,
        obs_dim: tuple[int, int, int],
        act_dim: int,
        hidden_channels: list[int],
        hidden_features: list[int],
        pooling: nn.Module,
        activation: nn.Module,
        conv_kernel_sizes: list[int],
        pooling_kernel_sizes: list[int],
    ):
        super().__init__()
        self.critic_net = cnn(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            hidden_features=hidden_features,
            out_features=act_dim,
            pooling=pooling,
            activation=activation,
            conv_kernel_sizes=conv_kernel_sizes,
            pooling_kernel_sizes=pooling_kernel_sizes,
        )

    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_net(x).squeeze(dim=-1)
