from typing import NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import torch


class Transition(NamedTuple):
    """
    A class for representing a Transition in a reinforcement learning environment.

    A Transition (s, a, r, s', done) is used to capture the key elements of an agent interaction with an environment:
        - s := current state
        - a := taken action
        - r := reward for doing action a from state s
        - s' := next state after taking action a from state s
        - done := Is state s' a terminal state or not
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class TransitionSample(NamedTuple):
    """
    A class for representing a sample of transitions in a reinforcement learning environment.

    A TransitionSample (states, actions, rewards, next_states, dones) is used to capture the key elements of an agent
    interaction with an environment as Tensor sample:
        - states := current states
        - actions := taken actions
        - rewards := rewards for doing actions a from states s
        - next_states := next states after taking actions a from states s
        - dones := Are states s' terminal states or not
    """
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class Buffer(ABC):
    """
    An abstract class representing a buffer for storing and sampling data.

    A buffer is used to collect and manage data samples, such as transitions, for training reinforcement learning agents.
    Concrete implementations of buffers should inherit from this base class and implement the required methods.
    """

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
    def filled(self, minimum_size: int) -> bool:
        """
        Returns True if the replay buffer has at least minimum_size transitions stored.

        Returns:
            bool:
                True if replay buffer has at least minimum_size transitions stored
        """
        pass

    @abstractmethod
    def _push(self, transition: Transition):
        pass

    def push(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool
    ):
        """
        Push a Transition (s, a, r, s', done) to the replay buffer.

        Args:
            state (np.ndarray):
                Current state s

            action (int):
                Taken action a

            reward (float):
                Reward r for taken action a from state s

            next_state (np.ndarray):
                Next state s' for taken action a from state s

            done (bool):
                Is next state s' a terminal state or not
        """
        return self._push(Transition(state, action, reward, next_state, done))

    @abstractmethod
    def _sample(self, batch_size: int) -> list[Transition]:
        pass

    def sample(self, batch_size: int) -> TransitionSample:
        """
        Samples a list of transitions from the replay buffer.

        Args:
            batch_size (int):
                Number of transitions to sample

        Returns:
            TransitionSample:
                Sampled transitions as pytorch tensors
        """
        samples = self._sample(batch_size)

        # Convert list[Transition] to TransitionSample
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in samples:
            states += [t.state]
            actions += [t.action]
            rewards += [t.reward]
            next_states += [t.next_state]
            dones += [t.done]

        states = torch.from_numpy(np.array(states)).to(dtype=torch.float32)
        actions = torch.from_numpy(np.array(actions)).to(dtype=torch.int64)
        rewards = torch.from_numpy(np.array(rewards)).to(dtype=torch.float32)
        next_states = torch.from_numpy(np.array(next_states)).to(dtype=torch.float32)
        dones = torch.from_numpy(np.array(dones)).to(dtype=torch.bool)
        return TransitionSample(states, actions, rewards, next_states, dones)

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
