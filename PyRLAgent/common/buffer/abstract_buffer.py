from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Union

import numpy as np
import torch


class Transition(NamedTuple):
    """
    A class for representing a Transition in a reinforcement learning environment.

    A Transition (s, a, r, s', done, log_prob) is used to capture the key elements of an agent interaction with an
    environment:
        - s := current state
        - a := taken action
        - r := reward for doing action a from state s
        - s' := next state after taking action a from state s
        - done := Is state s' a terminal state or not
        - log_prob := log probability of p(a | s)
        - value := state-value V(s)

    Attributes:
        state (Union[np.ndarray, torch.Tensor]):
            The current state s

        action (Union[np.ndarray, torch.Tensor]):
            The taken action a

        reward (Union[np.ndarray, torch.Tensor]):
            The reward r for doing action a from state s

        next_state (Union[np.ndarray, torch.Tensor]):
            The next state s' after taking action a from state s

        done (Union[bool, torch.Tensor]):
            Signalizes whether the end state is reached

        log_prob(Union[float, torch.Tensor], optional):
            The log probability p(a | s)

        value(Union[float, torch.Tensor], optional):
            The state-value V(s)
    """
    state: Union[np.ndarray, torch.Tensor]
    action: Union[int, torch.Tensor]
    reward: Union[float, torch.Tensor]
    next_state: Union[np.ndarray, torch.Tensor]
    done: Union[bool, torch.Tensor]
    log_prob: Optional[Union[float, torch.Tensor]] = None
    value: Optional[Union[float, torch.Tensor]] = None


class Buffer(ABC):
    """
    An abstract class representing a buffer for storing and sampling data.

    A buffer is used to collect and manage data samples, such as transitions, for training reinforcement learning
    agents.

    Concrete implementations of buffers should inherit from this base class and implement the required methods.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the buffer.
        """
        pass

    @abstractmethod
    def full(self) -> bool:
        """
        Returns True if the maximum size of the replay buffer is reached.

        Returns:
            bool:
                True if maximum size of replay buffer is reached
        """
        pass

    @abstractmethod
    def filled(self, min_size: int) -> bool:
        """
        Returns True if the replay buffer has at least min_size transitions stored.

        Args:
            min_size (int):
                The minimum size of Transitions

        Returns:
            bool:
                True if replay buffer has at least min_size transitions stored
        """
        pass

    @abstractmethod
    def _push(self, transition: Transition):
        """
        Pushes a Transition (s, a, r, s', done, log_prob, value) to the replay buffer.

        Args:
            transition (Transition):
                The transition to push to the replay buffer
        """
        pass

    def push(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            log_prob: Optional[Union[float, torch.Tensor]] = None,
            value: Optional[Union[float, torch.Tensor]] = None,
    ):
        """
        Pushes a Transition (s, a, r, s', done, log_prob, value) to the replay buffer.

        Args:
            state (np.ndarray):
                The current state s

            action (int):
                The taken action a

            reward (float):
                The reward r for taken action a from state s

            next_state (np.ndarray):
                The next state s' for taken action a from state s

            done (bool):
                Signalizes whether the end state is reached
            
            log_prob (Union[float, torch.Tensor], optional):
                The log probability of p(a | s)

            value (Union[float, torch.Tensor], optional):
                The state-value V(s)
        """
        return self._push(Transition(state, action, reward, next_state, done, log_prob, value))

    @abstractmethod
    def _get(self, batch_size: int) -> list[Transition]:
        """
        Returns the first [0, ..., batch_size - 1] transitions from the replay buffer.

        Args:
            batch_size (int):
                The number of transitions to consider

        Returns:
            list[Transition]:
                The first [0, ..., batch_size - 1] transitions
        """
        pass

    def get(self, batch_size: int) -> Transition:
        """
        Returns the first [0, ..., batch_size - 1] transitions from the replay buffer.

        Args:
            batch_size (int):
                The number of transitions to consider

        Returns:
            Transition:
                The transition as batch of PyTorch Tensors
        """
        samples = self._get(batch_size)

        # Convert list[Transition] to a single Transition with matrices
        states = torch.tensor(np.stack([t.state for t in samples], axis=1)).to(dtype=torch.float32)
        actions = torch.tensor(np.stack([t.action for t in samples], axis=1)).to(dtype=torch.int64)
        rewards = torch.tensor(np.stack([t.reward for t in samples], axis=1)).to(dtype=torch.float32)
        next_states = torch.tensor(np.stack([t.next_state for t in samples], axis=1)).to(dtype=torch.float32)
        dones = torch.tensor(np.stack([t.done for t in samples], axis=1)).to(dtype=torch.bool)

        log_probs = None
        if samples[0].log_prob is not None:
            log_probs = torch.tensor(np.stack([t.log_prob for t in samples], axis=1)).to(dtype=torch.float32)

        values = None
        if samples[0].value is not None:
            values = torch.tensor(np.stack([t.value for t in samples], axis=1)).to(dtype=torch.float32)

        return Transition(states, actions, rewards, next_states, dones, log_probs, values)

    @abstractmethod
    def _sample(self, batch_size: int) -> list[Transition]:
        """
        Samples a list of transitions from the replay buffer.

        Args:
            batch_size (int):
                The number of transitions to sample

        Returns:
            list[Transition]:
                The list of sampled Transitions
        """
        pass

    def sample(self, batch_size: int) -> Transition:
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int):
                The number of transitions to sample

        Returns:
            Transition:
                The sampled transition as batches of PyTorch Tensors
        """
        samples = self._sample(batch_size)

        # Convert list[Transition] to a single Transition with matrices
        states = torch.tensor(np.array([t.state for t in samples])).to(dtype=torch.float32)
        actions = torch.tensor(np.array([t.action for t in samples])).to(dtype=torch.int64)
        rewards = torch.tensor(np.array([t.reward for t in samples])).to(dtype=torch.float32)
        next_states = torch.tensor(np.array([t.next_state for t in samples])).to(dtype=torch.float32)
        dones = torch.tensor(np.array([t.done for t in samples])).to(dtype=torch.bool)

        log_probs = None
        if samples[0].log_prob is not None:
            log_probs = torch.tensor([t.log_prob for t in samples]).to(dtype=torch.float32)

        values = None
        if samples[0].value is not None:
            values = torch.tensor([t.value for t in samples]).to(dtype=torch.float32)

        return Transition(states, actions, rewards, next_states, dones, log_probs, values)

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
