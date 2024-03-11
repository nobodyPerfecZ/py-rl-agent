from abc import ABC, abstractmethod
from typing import Union

import torch


class Algorithm(ABC):
    """
    Abstract base class for reinforcement learning algorithms.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train(self):
        """
        Trains the algorithm.
        """
        pass

    @abstractmethod
    def compute_loss(self, **kwargs) -> torch.Tensor:
        """
        Computes the loss for the algorithm.

        Args:
            **kwargs:
                Arbitrary keyword arguments

        Returns:
            torch.Tensor:
                The computed loss
        """
        pass

    @abstractmethod
    def predict(self, **kwargs) -> torch.Tensor:
        """
        Predicts the next action a given the input.

        Args:
            **kwargs:
                Arbitrary keyword arguments

        Returns:
            torch.Tensor:
                The predicted action
        """
        pass

    @abstractmethod
    def fit(self, n_timesteps: Union[float, int]) -> list[float]:
        """
        Fits the algorithm for a certain number of timesteps.

        Args:
            n_timesteps (float | int):
                The number of timesteps to fit the algorithm.

        Returns:
            list[float]:
                The accumulated rewards for each epoch
        """
        pass

    @abstractmethod
    def eval(self, n_timesteps: Union[float, int]) -> list[float]:
        """
        Evaluates the algorithm for a certain number of timesteps.

        Args:
            n_timesteps (float | int):
                The number of timesteps to evaluate the algorithm.

        Returns:
            list[float]:
                The accumulated rewards for each epoch
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __getstate__(self) -> dict:
        """ Magic function to save a custom class as yaml file. """
        pass

    @abstractmethod
    def __setstate__(self, state: dict):
        """ Magic function to load a custom class from yaml file. """
        pass
