from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class Strategy(ABC):
    """
    An abstract base class representing a strategy for making decisions or selecting actions.

    This class defines the interface for various decision-making strategies used in reinforcement learning
    and related applications. Concrete implementations of strategies should inherit from this base class
    and implement the required methods.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def choose_action(self, state: Union[np.ndarray, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        """
        Returns the actions according to the given strategy and state-value function (q_values).

        Args:
            state (np.ndarray | torch.Tensor):
                The Current state

            output (torch.Tensor):
                The output of a policy

        Returns:
            torch.Tensor:
                The selected action according to the strategy
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Updates the strategy based on the received transition (s, a, r, s', done).

        Args:
            state (np.ndarray):
                The current state s

            action (int):
                The taken action a

            reward (float):
                The received reward r

            next_state (np.ndarray):
                The next state s'

            done (bool):
                Signalizes whether the end state is reached
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __getstate__(self) -> dict:
        """ Magic function to save a custom class as yaml file. """
        pass

    @abstractmethod
    def __setstate__(self, state: dict):
        """ Magic function to load a custom class from yaml file. """
        pass
