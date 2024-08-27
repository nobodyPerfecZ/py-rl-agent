from abc import ABC, abstractmethod
from typing import Union, Optional

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
    def update(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Union[np.ndarray, torch.Tensor],
            reward: Union[np.ndarray, torch.Tensor],
            next_state: Union[np.ndarray, torch.Tensor],
            done: Union[np.ndarray, torch.Tensor],
            log_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
            value: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Updates the strategy based on the received transition (s_t, a_t, r_t, s_t+1, done).

        Args:
            state (np.ndarray | torch.Tensor):
                The current state s_t

            action (np.ndarray | torch.Tensor):
                The taken action a_t

            reward (np.ndarray | torch.Tensor):
                The received reward r_t

            next_state (np.ndarray | torch.Tensor):
                The next state s_t+1

            done (np.ndarray | torch.Tensor):
                Signalizes whether the end state is reached

            log_prob(np.ndarray | torch.Tensor, optional):
                The log probability p(a_t | s_t)

            value(np.ndarray | torch.Tensor, optional):
                The state-value function V(s_t)
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
