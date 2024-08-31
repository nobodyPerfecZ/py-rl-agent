from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Normal

from pyrlagent.torch.util import cnn, mlp


class AbstractActorCriticNetwork(nn.Module, ABC):
    """Abstract class for the actor critic network."""

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
                The log probability of the specific action.
        """
        pass

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

    def forward(self, x: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        """
        The forward pass of the actor critic network.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            tuple[Distribution, torch.Tensor]:
                pi (Distribution):
                    The probability distribution over the actions

                critic_value (torch.Tensor):
                    The value of the observation
        """
        x = x.to(dtype=torch.float32)
        pi = self.distribution(x)
        critic_value = self.critic_value(x)
        return pi, critic_value


class MLPCategoricalActorCriticNetwork(AbstractActorCriticNetwork):
    """MLP for the actor critic with discrete action spaces."""

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
        self.critic_net = mlp(
            in_features=obs_dim,
            hidden_features=hidden_features,
            out_features=1,
            activation=activation,
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        x = x.to(dtype=torch.float32)
        logits = self.logits_net(x)
        return Categorical(logits=logits)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        a = a.to(dtype=torch.float32)
        return pi.log_prob(a)

    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        return self.critic_net(x).squeeze(dim=-1)


class CNNCategoricalActorCriticNetwork(AbstractActorCriticNetwork):
    """CNN for the actor critic with discrete action spaces."""

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
        self.critic_net = cnn(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            hidden_features=hidden_features,
            out_features=1,
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

    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_net(x).squeeze(dim=-1)


class MLPGaussianActorCriticNetwork(AbstractActorCriticNetwork):
    """MLP for the actor critic with continuous action spaces."""

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
            torch.as_tensor(-0.5 * torch.ones(act_dim, dtype=torch.float32))
        )
        self.critic_net = mlp(
            in_features=obs_dim,
            hidden_features=hidden_features,
            out_features=1,
            activation=activation,
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        x = x.to(dtype=torch.float32)
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        a = a.to(dtype=torch.float32)
        return pi.log_prob(a).sum(dim=-1)

    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        return self.critic_net(x).squeeze(dim=-1)


class CNNGaussianActorCriticNetwork(AbstractActorCriticNetwork):
    """CNN for the actor critic with continuous action spaces."""

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
            torch.as_tensor(-0.5 * torch.ones(act_dim, dtype=torch.float32))
        )
        self.critic_net = cnn(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            hidden_features=hidden_features,
            out_features=1,
            pooling=pooling,
            activation=activation,
            conv_kernel_sizes=conv_kernel_sizes,
            pooling_kernel_sizes=pooling_kernel_sizes,
        )

    def distribution(self, x: torch.Tensor) -> Distribution:
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(a).sum(dim=-1)

    def critic_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_net(x).squeeze(dim=-1)
