from typing import Any, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces import Discrete

from PyRLAgent.common.network.actor_critic import create_actor_critic_mlp
from PyRLAgent.common.network.dueling import create_dueling_mlp
from PyRLAgent.common.network.mlp import create_mlp
from PyRLAgent.common.policy.abstract_policy import ActorCriticPolicy, DeterministicPolicy
from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.common.strategy.greedy import Greedy
from PyRLAgent.enum.strategy import StrategyEnum


class QNetwork(DeterministicPolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            architecture: Optional[list[int]],
            activation_fn: Optional[nn.Module],
            output_activation_fn: Optional[nn.Module],
            bias: bool,
            strategy_type: Union[str, Type[Strategy]],
            strategy_kwargs: dict[str, Any],
    ):
        model = create_mlp(
            input_dim=np.prod(observation_space.shape).item(),
            output_dim=action_space.n,
            architecture=architecture,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            bias=bias,
        )

        strategy = StrategyEnum(strategy_type).to(**strategy_kwargs)

        super().__init__(
            model=model,
            non_deterministic_strategy=strategy,
            deterministic_strategy=Greedy(),
        )

    def _predict(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        q_values = self.model.forward(state)

        if deterministic:
            # Case: Choose the action which has the highest q_value
            return self.deterministic_strategy.choose_action(state, q_values)
        else:
            # Case: Choose the action according to the given strategy
            return self.non_deterministic_strategy.choose_action(state, q_values)


class QDuelingNetwork(DeterministicPolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            feature_architecture: Optional[list[int]],
            feature_activation_fn: Optional[nn.Module],
            feature_output_activation_fn: Optional[nn.Module],
            value_architecture: Optional[list[int]],
            value_activation_fn: Optional[nn.Module],
            value_output_activation_fn: Optional[nn.Module],
            advantage_architecture: Optional[list[int]],
            advantage_activation_fn: Optional[nn.Module],
            advantage_output_activation_fn: Optional[nn.Module],
            bias: bool,
            strategy_type: Union[str, Type[Strategy]],
            strategy_kwargs: dict[str, Any],
    ):
        model = create_dueling_mlp(
            input_dim=np.prod(observation_space.shape).item(),
            output_dim=action_space.n,
            feature_architecture=feature_architecture,
            feature_activation_fn=feature_activation_fn,
            feature_output_activation_fn=feature_output_activation_fn,
            value_architecture=value_architecture,
            value_activation_fn=value_activation_fn,
            value_output_activation_fn=value_output_activation_fn,
            advantage_architecture=advantage_architecture,
            advantage_activation_fn=advantage_activation_fn,
            advantage_output_activation_fn=advantage_output_activation_fn,
            bias=bias,
        )

        strategy = StrategyEnum(strategy_type).to(**strategy_kwargs)

        super().__init__(
            model=model,
            non_deterministic_strategy=strategy,
            deterministic_strategy=Greedy(),
        )

    def _predict(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        q_values = self.model.forward(state)

        if deterministic:
            # Case: Choose the action which has the highest q_value
            return self.deterministic_strategy.choose_action(state, q_values)
        else:
            # Case: Choose the action according to the given strategy
            return self.non_deterministic_strategy.choose_action(state, q_values)


class QCNNetwork(DeterministicPolicy):
    # TODO: Implement here

    def _predict(self, observation: torch.Tensor, deterministic: str) -> torch.Tensor:
        # TODO: Implement here
        pass


class QProbNetwork(DeterministicPolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.space,
            Q_min: float,
            Q_max: float,
            num_atoms: int,
            architecture: Optional[list[int]],
            activation_fn: Optional[nn.Module],
            output_activation_fn: Optional[list[int]],
            bias: bool,
            strategy_type: Union[str, Type[Strategy]],
            strategy_kwargs: dict[str, Any],
    ):
        self.Q_min = Q_min
        self.Q_max = Q_max
        self.num_actions = action_space.n
        self.num_atoms = num_atoms
        self.delta_Z = (self.Q_max - self.Q_min) / (self.num_atoms - 1)
        self.Z = torch.linspace(self.Q_min, self.Q_max, self.num_atoms)  # support values

        model = create_mlp(
            input_dim=np.prod(observation_space.shape).item(),
            output_dim=action_space.n * num_atoms,
            architecture=architecture,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            bias=bias,
        )

        strategy = StrategyEnum(strategy_type).to(**strategy_kwargs)

        super().__init__(
            model=model,
            non_deterministic_strategy=strategy,
            deterministic_strategy=Greedy(),
        )

    def _predict(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        # Get the logits of the probability distribution
        logits = self.model.forward(state)

        # Reshape the logits to (NUM_ENVS, NUM_TIMESTEPS, NUM_ACTIONS, NUM_ATOMS)
        if state.ndim == 2:
            logits = logits.reshape(state.shape[0], self.num_actions, self.num_atoms)
        else:
            logits = logits.reshape(self.num_actions, self.num_atoms)

        # Compute the probability distribution
        probabilities = F.softmax(logits, dim=-1)

        # Compute the q-values of each action
        q_values = torch.sum(probabilities * self.Z, dim=-1)

        if deterministic:
            # Case: Choose the action which has the highest q_value
            return self.deterministic_strategy.choose_action(state, q_values)
        else:
            # Case: Choose the action according to the given strategy
            return self.non_deterministic_strategy.choose_action(state, q_values)

    def forward(self, observation_or_state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # Calculate the logits by the forward pass
        logits = super().forward(observation_or_state)

        # Reshape the logits to (NUM_BATCH, NUM_ACTIONS, NUM_ATOMS)
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.num_actions, self.num_atoms)

        return logits


class ActorCriticNetwork(ActorCriticPolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            actor_architecture: Optional[list[int]],
            actor_activation_fn: Optional[list[int]],
            actor_output_activation_fn: Optional[list[int]],
            critic_architecture: Optional[list[int]],
            critic_activation_fn: Optional[list[int]],
            critic_output_activation_fn: Optional[list[int]],
            bias: bool,
    ):
        input_dim = observation_space.shape[0]

        if isinstance(action_space, Discrete):
            # Case: Discrete action space
            output_dim = action_space.n
            discrete = True
        else:
            # Case: Continuous action space
            output_dim = action_space.shape[0]
            discrete = False

        model = create_actor_critic_mlp(
            discrete=discrete,
            input_dim=input_dim,
            output_dim=output_dim,
            actor_architecture=actor_architecture,
            actor_activation_fn=actor_activation_fn,
            actor_output_activation_fn=actor_output_activation_fn,
            critic_architecture=critic_architecture,
            critic_activation_fn=critic_activation_fn,
            critic_output_activation_fn=critic_output_activation_fn,
            bias=bias,
        )

        super().__init__(
            model=model
        )
