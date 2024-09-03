from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch

from pyrlagent.torch.experience import Trajectory


class Buffer(ABC):
    """Base class for a buffer to store and sample transitions in RL."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        """Resets the buffer."""

    @abstractmethod
    def push(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: Union[np.ndarray, torch.Tensor],
        next_state: Union[np.ndarray, torch.Tensor],
        done: Union[np.ndarray, torch.Tensor],
        log_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
        value: Optional[Union[np.ndarray, torch.Tensor]] = None,
        next_value: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Adds a transition (s_t, a_t, r_t, s_t+1, done, log_prob, value, next_value) to the replay buffer.

        Args:
            state (np.ndarray | torch.Tensor):
                The current state s_t

            action (np.ndarray | torch.Tensor):
                The action a_t

            reward (np.ndarray | torch.Tensor):
                The reward r_t

            next_state (np.ndarray | torch.Tensor):
                The next state s_t+1

            done (np.ndarray | torch.Tensor):
                Whether the end state is reached

            log_prob (np.ndarray | torch.Tensor, optional):
                The log probability of p(a_t | s_t)

            value (np.ndarray | torch.Tensor, optional):
                The state value V(s_t)

            next_value (np.ndarray | torch.Tensor, optional):
                The state value V(s_t+1)
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Trajectory:
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int):
                The size of the batch

        Returns:
            Trajectory:
                The sampled transitions as a single transition
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __getstate__(self) -> dict:
        """Magic function to save a custom class as yaml file."""
        pass

    @abstractmethod
    def __setstate__(self, state: dict):
        """Magic function to load a custom class from yaml file."""
        pass
