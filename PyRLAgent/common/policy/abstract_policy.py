from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch.nn as nn
import torch

from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.util.observation import obs_to_tensor


class Policy(nn.Module, ABC):
    """
    An abstract class representing a policy.

    A Policy is a fundamental component in reinforcement learning that defines how an agent selects actions
    in response to environmental states. This abstract class provides a common interface for various policy
    implementations.
    """

    def __init__(
            self,
            model: nn.Module,
            non_deterministic_strategy: Strategy,
            deterministic_strategy: Strategy,
    ):
        super().__init__()
        self.model = model
        self.non_deterministic_strategy = non_deterministic_strategy
        self.deterministic_strategy = deterministic_strategy

    def freeze(self):
        """
        Freeze the models parameters (:= requires_grad = False).
        """
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, observation_or_state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        The Forward-Pass of the Policy Network.

        Args:
            observation_or_state (Union[np.ndarray, torch.Tensor]):
                Either the observation extracted by interacting with the environment or the preprocessed state

        Returns:
            torch.Tensor:
                Output of the Policy Network
        """
        if isinstance(observation_or_state, torch.Tensor):
            # Case: State is given
            return self.model.forward(observation_or_state)
        else:
            # Case: Observation is given - transform to state
            return self.model.forward(obs_to_tensor(observation_or_state))

    @abstractmethod
    def _predict(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
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
        Updates the Non-deterministic strategy by the given Transition (s, a, r, s', done)

        Args:
            state (np.ndarray):
                Current state s

            action (int):
                Taken action a

            reward (float):
                Reward r by taking action a from state s

            next_state (np.ndarray):
                Next state s' by taking action a from state s

            done (bool):
                Is next state s' a terminal state or not
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
