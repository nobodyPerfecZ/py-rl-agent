import numpy as np
import torch

from PyRLAgent.common.strategy.abstract_strategy import Strategy


class Random(Strategy):
    """
    A concrete implementation of a random exploration strategy.

    The Random strategy selects actions completely at random without considering value estimates or
    exploration-exploitation trade-offs. It's useful for exploration in the early stages of learning and as a baseline
    for comparison with more sophisticated strategies.
    """

    def __init__(self):
        pass

    def choose_action(self, state: np.ndarray, output: torch.Tensor) -> torch.Tensor:
        size = () if output.dim() == 1 else (output.size()[0],)
        return torch.randint(0, len(output), size=size)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        pass

    def __str__(self) -> str:
        return "Random()"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, Random):
            return True
        raise NotImplementedError

    def __getstate__(self) -> dict:
        pass

    def __setstate__(self, state: dict):
        pass
