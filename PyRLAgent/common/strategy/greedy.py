import numpy as np
import torch

from PyRLAgent.common.strategy.abstract_strategy import Strategy


class Greedy(Strategy):
    """
    A class representing a greedy exploration strategy.

    The Greedy strategy always selects the action with the highest estimated Q-values based on the current state,
    without exploration. It's a deterministic strategy used when the agent aims to exploit its current knowledge.
    """

    def __init__(self):
        pass

    def choose_action(self, state: np.ndarray, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, dim=-1)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        pass

    def __str__(self) -> str:
        return "Greedy()"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, Greedy):
            return True
        raise NotImplementedError

    def __getstate__(self) -> dict:
        pass

    def __setstate__(self, state: dict):
        pass
