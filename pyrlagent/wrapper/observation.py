from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType


class NormalizeObservationWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper class to normalize the observation of a Gymnasium environment.

    This wrapper normalize the observations after the Welford algorithm, by iteratively update the mean and variance
    of observed values.

    Attributes:
        env (Gymnasium.Environment):
            The Gymnasium environment

        eps (float):
            The added value to the standard deviation
    """

    def __init__(self, env: gym.Env, eps: float = 1e-4):
        super().__init__(env)
        self.mean = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)
        self.var = np.ones(env.observation_space.shape, dtype=env.observation_space.dtype)
        self.count = 0
        self.eps = eps

    def step(
            self,
            action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Perform a step on the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update mean and variance after the Welford algorithm
        delta = observation - self.mean
        self.mean += delta / (self.count + 1)
        self.var += delta * (observation - self.mean) / (self.count + 1)
        self.count += 1

        # Normalize the observation
        normalized_observation = (observation - self.mean) / (np.sqrt(self.var) + self.eps)

        # Store non-normalized observation
        info["observation"] = observation

        return normalized_observation, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[WrapperObsType, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.mean = np.zeros(self.env.observation_space.shape, dtype=self.env.observation_space.dtype)
        self.var = np.ones(self.env.observation_space.shape, dtype=self.env.observation_space.dtype)
        self.count = 0

        # Update mean and variance after the Welford algorithm
        delta = observation - self.mean
        self.mean += delta / (self.count + 1)
        self.var += delta * (observation - self.mean) / (self.count + 1)
        self.count += 1

        # Normalize the observation
        normalized_observation = (observation - self.mean) / (np.sqrt(self.var) + self.eps)

        # Store non-normalized observation
        info["observation"] = observation

        return normalized_observation, info
