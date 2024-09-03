from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from pyrlagent.torch.util import cnn, mlp


class ActorNetwork(nn.Module, ABC):
    """Base class for an actor network."""

    @abstractmethod
    def distribution(self, x: torch.Tensor) -> Distribution:
        """
        Returns the probability distribution over the actions.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            Distribution:
                The probability distribution over the actions
        """
        pass

    @abstractmethod
    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of the specific action.

        Args:
            pi (Distribution):
                The probability distribution over each action

            a (torch.Tensor):
                The selected action

        Returns:
            torch.Tensor:
                The log probability of the specific action
        """
        pass

    def forward(self, x: torch.Tensor) -> Distribution:
        """
        The forward pass of the actor network.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            Distribution:
                The probability distribution over the actions
        """
        x = x.to(dtype=torch.float32)
        pi = self.distribution(x)
        return pi


class MLPCategoricalActorNetwork(ActorNetwork):
    """MLP for the actor with discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_features: list[int],
        activation: nn.Module,
    ):
        super().__init__()
        self.logits_net = mlp(
            in_features=obs_dim,
            hidden_features=hidden_features,
            out_features=act_dim,
            activation=activation,
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        x = x.to(dtype=torch.float32)
        logits = self.logits_net(x)
        return Categorical(logits=logits)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        a = a.to(dtype=torch.float32)
        return pi.log_prob(a)


class CNNCategoricalActorNetwork(ActorNetwork):
    """CNN for the actor with discrete action spaces."""

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
        self.logits_net = cnn(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            hidden_features=hidden_features,
            out_features=act_dim,
            pooling=pooling,
            activation=activation,
            conv_kernel_sizes=conv_kernel_sizes,
            pooling_kernel_sizes=pooling_kernel_sizes,
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        logits = self.logits_net(x)
        return Categorical(logits=logits)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(a)


class MLPGaussianActorNetwork(ActorNetwork):
    """MLP for the actor with continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_features: list[int],
        activation: nn.Module,
    ):
        super().__init__()
        self.mu_net = mlp(
            in_features=obs_dim,
            hidden_features=hidden_features,
            out_features=act_dim,
            activation=activation,
        )
        self.log_std = torch.nn.Parameter(
            torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32))
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        x = x.to(dtype=torch.float32)
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        a = a.to(dtype=torch.float32)
        return pi.log_prob(a).sum(dim=-1)


class CNNGaussianActorNetwork(ActorNetwork):
    """CNN for the actor with continuous action spaces."""

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
        self.mu_net = cnn(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            hidden_features=hidden_features,
            out_features=act_dim,
            pooling=pooling,
            activation=activation,
            conv_kernel_sizes=conv_kernel_sizes,
            pooling_kernel_sizes=pooling_kernel_sizes,
        )
        self.log_std = torch.nn.Parameter(
            torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32))
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(a).sum(dim=-1)
