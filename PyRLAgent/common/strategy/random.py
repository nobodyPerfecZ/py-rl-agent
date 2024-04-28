from typing import Union, Optional

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

    def choose_action(self, state: Union[np.ndarray, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        n_actions = output.shape[-1]
        return torch.randint(0, n_actions, size=output.shape[:-1])

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
