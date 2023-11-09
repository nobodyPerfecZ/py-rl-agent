from typing import Union, Any, Type, List
from gymnasium import spaces

import torch.nn as nn
import numpy as np
import torch

from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.common.strategy.random import Random
from PyRLAgent.common.strategy.greedy import Greedy
from PyRLAgent.common.strategy.epsilon_greedy import LinearDecayEpsilonGreedy, ExponentialDecayEpsilonGreedy
from PyRLAgent.common.strategy.ucb import UCB
from PyRLAgent.common.policy.abstract_policy import Policy
from PyRLAgent.util.torch_layers import create_mlp


class QNetwork(Policy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.space,
            architecture: List[int],
            activation_fn: Union[nn.Module, List[nn.Module]],
            bias: bool,
            strategy_type: Union[str, Type[Strategy]],
            strategy_kwargs: dict[str, Any],
    ):
        if isinstance(activation_fn, nn.Module):
            # Case: All hidden layers should have the same activation function
            activation_fn = [activation_fn for _ in range(len(architecture))]

        strategy_type = QNetwork.get_strategy_type(strategy_type)

        model = create_mlp(
            input_dim=np.prod(observation_space.shape).item(),
            output_dim=action_space.n,
            architecture=architecture,
            activation_fn=activation_fn,
            bias=bias,
        )
        strategy = strategy_type(**strategy_kwargs)

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

    @staticmethod
    def get_strategy_type(strategy_type: Union[str, Type[Strategy]]) -> Type[Strategy]:
        """
        Get and return a strategy class based on the specified strategy type.

        This method allows for checking and returning a strategy class based on the given strategy type.
        It can be either a string key or the class itself.

        The following keys are allowed for a policy:
            - "random" := Random
            - "greedy" := Greedy
            - "linear-epsilon" := LinearDecayEpsilonGreedy
            - "exp-epsilon" := ExponentialDecayEpsilonGreedy
            - "ucb" := UCB

        Args:
            strategy_type (Union[str, Type[Strategy]]):
                The strategy type, which can be a string or the class itself.

        Returns:
            Type[Strategy]:
                The concrete strategy class corresponding to the specified strategy type.
        """
        strategy_type_map = {
            "random": Random,
            "greedy": Greedy,
            "linear-epsilon": LinearDecayEpsilonGreedy,
            "exp-epsilon": ExponentialDecayEpsilonGreedy,
            "ucb": UCB,
        }

        if isinstance(strategy_type, str):
            strategy_type = strategy_type_map.get(strategy_type)
            if strategy_type is None:
                raise ValueError(
                    "Illegal strategy_type! The argument should be either 'random', 'greedy', 'linear-epsilon', 'exp-epsilon' or 'ucb'!"
                )
        else:
            if strategy_type not in strategy_type_map.items():
                raise ValueError(
                    "Illegal strategy_type! The argument should be either 'Random', 'Greedy', 'LinearDecayEpsilonGreedy', 'ExponentialDecayEpsilionGreedy' or 'UCB'!"
                )
        return strategy_type


class QCNNetwork(Policy):
    # TODO: Implement here

    def _predict(self, observation: torch.Tensor, deterministic: str) -> torch.Tensor:
        # TODO: Implement here
        pass
