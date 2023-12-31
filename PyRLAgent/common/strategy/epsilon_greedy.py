from abc import ABC, abstractmethod
import numpy as np
import torch

from PyRLAgent.common.strategy.abstract_strategy import Strategy


class EpsilonGreedy(Strategy, ABC):
    """
    An abstract class representing an epsilon-greedy exploration strategy.

    The epsilon-greedy strategy balances exploration and exploitation from a reinforcement learning agent.
    With probability epsilon, it explores by selecting a random action, and with probability 1 - epsilon,
    it exploits by selecting the action with the highest estimated state-value function (q_values),
    given the current observation.
    """

    def __init__(self, epsilon_min: float, epsilon_max: float):
        if 0.0 > epsilon_min or 1.0 < epsilon_min:
            raise ValueError(
                "Illegal epsilon_min!"
                "The argument should be in between of 0.0 (inclusive) and 1.0 (inclusive)!"
            )
        if 0.0 > epsilon_max or 1.0 < epsilon_max:
            raise ValueError(
                "Illegal epsilon_max!"
                "The argument should be in between of 0.0 (inclusive) and 1.0 (inclusive)!"
            )
        if epsilon_min > epsilon_max:
            raise ValueError(
                "Illegal epsilon_min!"
                "The argument should be lower equal to epsilon_max!"
            )
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def choose_action(self, state: np.ndarray, output: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.epsilon:
            # Case: Choose a random action
            size = () if output.dim() == 1 else (output.size()[0],)
            actions = torch.randint(0, len(output), size=size)
        else:
            # Case: Choose the best actions that maximizes the Q-values
            actions = torch.argmax(output, dim=-1)
        return actions

    def __eq__(self, other) -> bool:
        if isinstance(other, EpsilonGreedy):
            return self.epsilon_min == other.epsilon_min and \
                   self.epsilon_max == other.epsilon_max and \
                   self.epsilon == other.epsilon
        raise NotImplementedError

    def __getstate__(self) -> dict:
        return {
            "epsilon_min": self.epsilon_min,
            "epsilon_max": self.epsilon_max,
            "epsilon": self.epsilon,
        }

    def __setstate__(self, state: dict):
        self.epsilon_min = state["epsilon_min"]
        self.epsilon_max = state["epsilon_max"]
        self.epsilon = state["epsilon"]


class LinearDecayEpsilonGreedy(EpsilonGreedy):
    """
    A class representing an epsilon-greedy exploration strategy with linear decay.

    The LinearDecayEpsilonGreedy strategy starts with a high exploration rate (epsilon) and
    linearly decays it over time. It balances exploration and exploitation from a reinforcement learning agent.
    With a high initial epsilon, it explores by selecting random actions,
    and over time, it reduces exploration to favor exploitation.
    """

    def __init__(self, epsilon_min: float, epsilon_max: float, steps: int):
        super().__init__(epsilon_min, epsilon_max)
        self.steps = steps
        self.momentum = (self.epsilon_max - self.epsilon_min) / self.steps

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.epsilon = max(self.epsilon - self.momentum, self.epsilon_min)

    def __str__(self) -> str:
        return f"LinearDecayEpsilonGreedy(epsilon_min={self.epsilon_min}, epsilon_max={self.epsilon_max}, epsilon={self.epsilon}, steps={self.steps})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, LinearDecayEpsilonGreedy):
            is_eq = super().__eq__(other)
            return is_eq and \
                   self.steps == other.steps and \
                   self.momentum == other.momentum
        raise NotImplementedError

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["steps"] = self.steps
        state["momentum"] = self.momentum
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self.steps = state["steps"]
        self.momentum = state["momentum"]


class ExponentialDecayEpsilonGreedy(EpsilonGreedy):
    """
    A class representing an epsilon-greedy exploration strategy with exponential decay.

    The ExponentialDecayEpsilonGreedy strategy starts with a high exploration rate (epsilon) and
    exponentially decays it over time. It balances exploration and exploitation from a reinforcement learning agent.
    With a high initial epsilon, it explores by selecting random actions,
    and over time, it reduces exploration exponentially to favor exploitation.
    """

    def __init__(self, epsilon_min: float, epsilon_max: float, decay_factor: float):
        super().__init__(epsilon_min, epsilon_max)
        self.decay_factor = decay_factor

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.epsilon = max(self.epsilon * self.decay_factor, self.epsilon_min)

    def __str__(self) -> str:
        return f"ExponentialDecayEpsilonGreedy(epsilon_min={self.epsilon_min}, epsilon_max={self.epsilon_max}, epsilon={self.epsilon}, decay_factor={self.decay_factor})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        print("Geht hier rein!")
        if isinstance(other, ExponentialDecayEpsilonGreedy):
            is_eq = super().__eq__(other)
            return is_eq and \
                   self.decay_factor == other.decay_factor
        raise NotImplementedError

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["decay_factor"] = self.decay_factor
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self.decay_factor = state["decay_factor"]
