from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType


class NormalizeRewardWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper class to normalize the rewards of a Gymnasium environment.

    This wrapper normalize the rewards after the Welford algorithm, by iteratively update the mean and variance
    of observed values.

    Attributes:
        env (Gymnasium.Environment):
            The Gymnasium environment

        eps (float):
            The added value to the standard deviation
    """

    def __init__(self, env: gym.Env, eps: float = 1e-4):
        super().__init__(env)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = eps

    def step(
            self,
            action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Perform a step on the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update mean and variance after the Welford algorithm
        delta = reward - self.mean
        self.mean += delta / (self.count + 1)
        self.var += delta * (reward - self.mean) / (self.count + 1)
        self.count += 1

        # Normalize the reward
        normalized_reward = (reward - self.mean) / (np.sqrt(self.var) + self.eps)

        # Store non-normalized reward
        info["reward"] = reward

        return observation, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[WrapperObsType, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

        # Store non-normalized reward
        info["reward"] = None

        return observation, info
