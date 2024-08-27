from abc import ABC, abstractmethod
from typing import Any, Optional

from pyrlagent.torch.experience.trajectory import Trajectory


class AbstractBuffer(ABC):
    """Abstract class representing a buffer for storing and sampling transitions in RL."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        """Resets the buffer."""

    @abstractmethod
    def push(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
        log_prob: Optional[Any] = None,
        value: Optional[Any] = None,
        next_value: Optional[Any] = None,
    ):
        """
        Adds a transition (s_t, a_t, r_t, s_t+1, done, log_prob, value, next_value) to the replay buffer.

        Args:
            state (Any):
                The current state s_t

            action (Any):
                The action a_t

            reward (Any):
                The reward r_t

            next_state (Any):
                The next state s_t+1

            done (Any):
                Whether the end state is reached

            log_prob (Any, optional):
                The log probability of p(a_t | s_t)

            value (np.ndarray, optional):
                The state value V(s_t)

            next_value (np.ndarray, optional):
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
