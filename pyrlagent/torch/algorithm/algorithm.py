from abc import ABC, abstractmethod


class AbstractRLAlgorithm(ABC):
    """Abstract class for Reinforcement Learning (RL) algorithms."""

    @abstractmethod
    def fit(self, num_timesteps: int) -> list[float]:
        """
        Fits the algorithm for a certain number of timesteps.

        Args:
            num_timesteps (int):
                The number of timesteps to fit the algorithm.

        Returns:
            list[float]:
                The accumulated rewards for each epoch
        """
        pass

    @abstractmethod
    def eval(self, num_timesteps: int) -> list[float]:
        """
        Evaluates the algorithm for a certain number of timesteps.

        Args:
            num_timesteps (int):
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
        """Magic function to save a custom class as yaml file."""
        pass

    @abstractmethod
    def __setstate__(self, state: dict):
        """Magic function to load a custom class from yaml file."""
        pass
