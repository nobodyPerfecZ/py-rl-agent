from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Normal

from PyRLAgent.common.network.mlp import create_mlp


class ActorNetwork(nn.Module, ABC):

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

    def forward(
            self,
            x: torch.Tensor,
            a: Optional[torch.Tensor] = None
    ) -> Union[Distribution, tuple[Distribution, torch.Tensor]]:
        """
        The forward pass of the actor network.

        If 'a' is not provided, then it will only return the probability distribution over the actions.

        Args:
            x (torch.Tensor):
                The observation of the environment

            a (torch.Tensor, optional):
                The selected action

        Returns:
            Union[Distribution, tuple[Distribution, torch.Tensor]]:
                pi (Distribution):
                    The probability distribution over the actions

                log_prob (torch.Tensor):
                    The log probability pi(a | s)
        """
        pi = self.distribution(x)
        if a is not None:
            log_prob = self.log_prob(pi, a)
            return pi, log_prob
        return pi


class CategoricalActorNetwork(ActorNetwork):

    def __init__(self, logits: nn.Module):
        super().__init__()
        self.logits = logits

    def distribution(self, x: torch.Tensor) -> Distribution:
        logits = self.logits(x)
        return Categorical(logits=logits)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(a)


class GaussianActorNetwork(ActorNetwork):

    def __init__(self, mu: nn.Module, log_std: nn.Parameter):
        super().__init__()
        self.mu = mu
        self.log_std = log_std

    def distribution(self, x: torch.Tensor) -> Distribution:
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob(self, pi: Distribution, a: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(a).sum(dim=-1)


class CriticNetwork(nn.Module):

    def __init__(self, critic: nn.Module):
        super().__init__()
        self.critic = critic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(dim=-1)


class ActorCriticNetwork(nn.Module):

    def __init__(self, actor: ActorNetwork, critic: CriticNetwork):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(
            self,
            x: torch.Tensor,
            a: Optional[torch.Tensor] = None
    ) -> Union[tuple[Distribution, torch.Tensor], tuple[Distribution, torch.Tensor, torch.Tensor]]:
        """
        The forward pass of the actor-critic network.

        If 'a' is not provided then 'log_prob' will not be returned.

        Args:
            x (torch.Tensor):
                The observation of the environment

            a (torch.Tensor, optional):
                The selected action

        Returns:
            Union[tuple[Distribution, torch.Tensor], tuple[Distribution, torch.Tensor, torch.Tensor]]:
                pi (Distribution):
                    The probability distribution over the actions

                log_prob (torch.Tensor):
                    The log probability pi(a | s)

                value (torch.Tensor):
                    The state-value V(s)
        """
        pi, log_prob = self.actor.forward(x, a)
        value = self.critic.forward(x)
        return pi, log_prob, value


def create_actor_critic_mlp(
        discrete: bool,
        input_dim: int,
        output_dim: int,
        actor_architecture: Optional[list[int]] = None,
        actor_activation_fn: Optional[list[int]] = None,
        actor_output_activation_fn: Optional[list[int]] = None,
        critic_architecture: Optional[list[int]] = None,
        critic_activation_fn: Optional[list[int]] = None,
        critic_output_activation_fn: Optional[list[int]] = None,
        bias: bool = True
) -> ActorCriticNetwork:
    # Create the actor network
    if discrete:
        # Case: The environment has discrete actions
        logits = create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            architecture=actor_architecture,
            activation_fn=actor_activation_fn,
            output_activation_fn=actor_output_activation_fn,
            bias=bias,
        )
        actor_network = CategoricalActorNetwork(logits=logits)
    else:
        # Case: The environment has continuous actions
        mu = create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            architecture=actor_architecture,
            activation_fn=actor_activation_fn,
            output_activation_fn=actor_output_activation_fn,
            bias=bias,
        )
        log_std = nn.Parameter(-0.5 * torch.ones(output_dim, dtype=torch.float32))
        actor_network = GaussianActorNetwork(mu=mu, log_std=log_std)

    # Create the critic network
    critic = create_mlp(
        input_dim=input_dim,
        output_dim=1,
        architecture=critic_architecture,
        activation_fn=critic_activation_fn,
        output_activation_fn=critic_output_activation_fn,
        bias=bias,
    )
    critic_network = CriticNetwork(critic=critic)

    # Return the actor-critic network
    return ActorCriticNetwork(actor=actor_network, critic=critic_network)
