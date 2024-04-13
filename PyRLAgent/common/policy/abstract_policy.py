from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from PyRLAgent.common.network.actor_critic import ActorCriticNetwork
from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.util.observation import obs_to_tensor


class DeterministicPolicy(nn.Module, ABC):
    """
    An abstract class representing a policy.

    A Policy is a fundamental component in reinforcement learning that defines how an agent selects actions
    in response to states from an environment. This abstract class provides a common interface for various policy
    implementations.

    Attributes:
        model (nn.Module):
            The PyTorch network (~ policy)

        non_deterministic_strategy (Strategy):
            The non-deterministic strategy for making decision or selecting actions if deterministic=False

        deterministic_strategy (Strategy):
            The deterministic strategy for making decision or selecting actions if deterministic=True
    """

    def __init__(
            self,
            model: nn.Module,
            non_deterministic_strategy: Strategy,
            deterministic_strategy: Strategy,
            **kwargs
    ):
        super().__init__()
        self.model = model
        self.non_deterministic_strategy = non_deterministic_strategy
        self.deterministic_strategy = deterministic_strategy

    def freeze(self):
        """
        Freeze the models parameters (requires_grad=False).
        """
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, observation_or_state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        The forward pass of the policy.

        Args:
            observation_or_state (Union[np.ndarray, torch.Tensor]):
                The input of the forward pass as the observation (np.ndarray) or preprocessed state (torch.Tensor)

        Returns:
            torch.Tensor:
                Output of the policy
        """
        if isinstance(observation_or_state, torch.Tensor):
            # Case: State is given
            return self.model.forward(observation_or_state)
        else:
            # Case: Observation is given - transform to state
            return self.model.forward(obs_to_tensor(observation_or_state))

    @abstractmethod
    def _predict(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Predicts the action given the policy and current preprocessed state.

        Args:
            state (torch.Tensor):
                The preprocessed state by interacting with the environment

            deterministic (bool):
                Decides if the action should be selected according to the deterministic exploration strategy (:= True)
                or the non-deterministic exploration strategy (:= False)

        Returns:
            torch.Tensor:
                The selected action
        """
        pass

    def predict(self, observation: np.ndarray, deterministic: bool) -> torch.Tensor:
        """
        Predict the action given the policy network, exploration strategy and the current observation.

        Args:
            observation (np.ndarray):
                Observation extracted by interacting with the environment

            deterministic (bool):
                Decides if the action should be selected according to the deterministic exploration strategy (:= True)
                or the Non-deterministic exploration strategy (:= False)

        Returns:
            torch.Tensor:
                Selected action as Pytorch Tensor
        """
        self.train(False)
        with torch.no_grad():
            actions = self._predict(obs_to_tensor(observation), deterministic)
        return actions

    def update_strategy(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Updates the non-deterministic strategy by the given Transition (s, a, r, s', done)

        Args:
            state (np.ndarray):
                The current state s

            action (int):
                The taken action a

            reward (float):
                The reward r by taking action a from state s

            next_state (np.ndarray):
                The next state s' by taking action a from state s

            done (bool):
                Signalized whether the end state is reached
        """
        self.non_deterministic_strategy.update(state, action, reward, next_state, done)

    def __str__(self):
        header = f"{self.__class__.__name__}("
        model_line = f"(model): {self.model.__str__()},"
        non_deterministic_strategy_line = f"(non_deterministic_strategy): {self.non_deterministic_strategy.__str__()},"
        deterministic_strategy_line = f"(deterministic_strategy): {self.deterministic_strategy.__str__()}"
        end = ")"
        return "\n".join([header, model_line, non_deterministic_strategy_line, deterministic_strategy_line, end])

    def __getstate__(self) -> dict:
        """ Magic function to save a custom class as yaml file. """
        state = super().__getstate__()
        state["non_deterministic_strategy"] = self.non_deterministic_strategy
        state["deterministic_strategy"] = self.deterministic_strategy
        return state

    def __setstate__(self, state: dict):
        """ Magic function to load a custom class from yaml file. """
        super().__setstate__(state)
        self.non_deterministic_strategy = state["non_deterministic_strategy"]
        self.deterministic_strategy = state["deterministic_strategy"]


class StochasticPolicy(nn.Module, ABC):
    pass


class ActorCriticPolicy(nn.Module, ABC):
    """
    An abstract class representing a stochastic policy.

    A Policy is a fundamental component in reinforcement learning that defines how an agent selects actions
    in response to states from an environment. This abstract class provides a common interface for various policy
    implementations.

    Attributes:
        model (ActorCriticNetwork):
            The PyTorch network to sample actions (~actor) and to compute baseline values (~critic)
    """

    def __init__(
            self,
            model: ActorCriticNetwork,
            **kwargs
    ):
        super().__init__()
        self.model = model

    def forward(
            self,
            observation_or_state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
    ) -> Union[tuple[Distribution, torch.Tensor], tuple[Distribution, torch.Tensor, torch.Tensor]]:
        """
        The forward pass of the policy.

        Args:
            observation_or_state (np.ndarray | torch.Tensor):
                The observation (np.ndarray) or preprocessed state (torch.Tensor) of the environment

            action (np.ndarray | torch.Tensor):
                The selected actions

        Returns:
            tuple[Distribution, Optional[torch.Tensor], Tensor]:
                Output of the policy
        """
        if not isinstance(observation_or_state, torch.Tensor):
            observation_or_state = obs_to_tensor(observation_or_state)
        if not isinstance(action, torch.Tensor):
            action = obs_to_tensor(action)
        return self.model.forward(observation_or_state, action)

    def _predict(
            self,
            observation: torch.Tensor
    ) -> tuple[Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the probability distribution over all actions
        pi = self.model.actor.distribution(observation)

        # Sample the next action
        action = pi.sample()

        # Compute the log probability pi(a | s)
        log_prob = self.model.actor.log_prob(pi, action)

        # Compute the state value V(s)
        value = self.model.critic(observation)

        return pi, action, log_prob, value

    def predict(
            self,
            observation: np.ndarray,
            return_all: bool = False,
    ) -> Union[torch.Tensor, tuple[Distribution, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Predict the action given the policy network and the current observation.

        If return_all is False then it will only return the action.

        Args:
            observation (np.ndarray):
                Observation extracted by interacting with the environment

            return_all (bool):
                Controls whether to only return the action or other necessary information

        Returns:
            Union[torch.Tensor, tuple[Distribution, torch.Tensor, torch.Tensor, torch.Tensor]]:
                pi (Distribution):
                    The probability distribution over all actions

                action (torch.Tensor):
                    The selected action a

                log_prob (torch.Tensor):
                    The log probability pi(a | s)

                value (torch.Tensor):
                    The state-value V(s)
        """
        with torch.no_grad():
            pi, action, log_prob, value = self._predict(obs_to_tensor(observation))

        if return_all:
            return pi, action, log_prob, value
        return action

    def __str__(self):
        header = f"{self.__class__.__name__}("
        model_line = f"(model): {self.model.__str__()},"
        end = ")"
        return "\n".join([header, model_line, end])

    def __getstate__(self) -> dict:
        """ Magic function to save a custom class as yaml file. """
        state = super().__getstate__()
        return state

    def __setstate__(self, state: dict):
        """ Magic function to load a custom class from yaml file. """
        super().__setstate__(state)
