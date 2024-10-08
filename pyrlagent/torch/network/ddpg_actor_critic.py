from abc import ABC, abstractmethod

import torch
from torch import nn

from pyrlagent.torch.util import cnn, cnn_in_features, cnn_mlp, mlp


class DDPGActorCriticNetwork(nn.Module, ABC):
    """Base class for a DDPG actor critic network."""

    @abstractmethod
    def action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the action of the observation.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            torch.Tensor:
                The action of the observation.
        """
        pass

    @abstractmethod
    def q_value(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the state-action value of the observation.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            torch.Tensor:
                The state-action value of the observation.
        """
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass of the DDPG actor critic network.

        Args:
            x (torch.Tensor):
                The observation of the environment

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                a (torch.Tensor):
                    The actions of the observation

                q_value (torch.Tensor):
                    The state-action value of the observation
        """
        action = self.action(x.to(dtype=torch.float32))
        return action, self.q_value(x.to(dtype=torch.float32), action)


class MLPContinuousDDPGActorCriticNetwork(DDPGActorCriticNetwork):
    """MLP for the actor critic with continuous action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_features: list[int],
        activation: nn.Module,
        noise_scale: float,
        low_action: float,
        high_action: float,
    ):
        super().__init__()
        self.mu_net = mlp(
            in_features=obs_dim,
            hidden_features=hidden_features,
            out_features=act_dim,
            activation=activation,
        )

        self.q_net = mlp(
            in_features=obs_dim + act_dim,
            hidden_features=hidden_features,
            out_features=1,
            activation=activation,
        )

        self.noise_scale = noise_scale
        self.low_action = low_action
        self.high_action = high_action

    def action(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mu_net(x.to(dtype=torch.float32))
        a += self.noise_scale * torch.randn(a.shape, device=a.device)
        return torch.clip(a, self.low_action, self.high_action)

    def q_value(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q_net(
            torch.cat([x.to(dtype=torch.float32), a.to(dtype=torch.float32)], dim=-1)
        ).squeeze(dim=-1)


class CNNContinuousDDPGActorCriticNetwork(DDPGActorCriticNetwork):
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
        noise_scale: float,
        low_action: float,
        high_action: float,
    ):
        super().__init__()
        self.cnn_mu_net, self.mlp_mu_net = cnn_mlp(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            hidden_features=hidden_features,
            out_features=act_dim,
            pooling=pooling,
            activation=activation,
            conv_kernel_sizes=conv_kernel_sizes,
            pooling_kernel_sizes=pooling_kernel_sizes,
        )

        self.cnn_q_net = cnn(
            input_shape=obs_dim,
            hidden_channels=hidden_channels,
            pooling=pooling,
            activation=activation,
            conv_kernel_sizes=conv_kernel_sizes,
            pooling_kernel_sizes=pooling_kernel_sizes,
        )
        self.mlp_q_net = mlp(
            in_features=cnn_in_features(obs_dim, list(self.cnn_q_net.modules()))
            + act_dim,
            hidden_features=hidden_features,
            out_features=1,
            activation=activation,
        )

        self.noise_scale = noise_scale
        self.low_action = low_action
        self.high_action = high_action

    def action(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp_mu_net(self.cnn_mu_net(x.to(dtype=torch.float32)))
        a += self.noise_scale * torch.randn(a.shape, device=a.device)
        return torch.clip(a, self.low_action, self.high_action)

    def q_value(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.mlp_q_net(
            torch.cat(
                [self.cnn_q_net(x.to(dtype=torch.float32)), a.to(dtype=torch.float32)],
                dim=-1,
            )
        ).squeeze(dim=-1)
