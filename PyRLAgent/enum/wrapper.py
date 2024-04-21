from enum import Enum
from typing import Optional, Any

import gymnasium as gym

from PyRLAgent.wrapper.observation import NormalizeObservationWrapper
from PyRLAgent.wrapper.reward import NormalizeRewardWrapper


class GymWrapperEnum(str, Enum):
    """
    An enumeration of supported wrapper types.
    """
    NONE = "none"
    NORMALIZE_OBSERVATION = "normalize-observation"
    NORMALIZE_REWARD = "normalize-reward"

    @classmethod
    def wrapper(cls) -> dict[Enum, Any]:
        """
        Returns the wrapper dictionary, where
            - `key` represents the wrapper type as enum
            - `value` represents the wrapper class

        Returns:
            dict[Enum, Any]:
                The wrapper of GymWrapperEnum
        """
        return {
            cls.NONE: None,
            cls.NORMALIZE_OBSERVATION: NormalizeObservationWrapper,
            cls.NORMALIZE_REWARD: NormalizeRewardWrapper,
        }

    def to_wrapper(self, env: gym.Env, **wrapper_kwargs) -> Optional[gym.Env]:
        """
        Initialize the wrapped gymnasium environment given the arguments.

        Args:
            env (gym.Env):
                The gymnasium environment

            **wrapper_kwargs:
                Additional arguments for the wrapper class

        Returns:
            gym.Env | None:
                The wrapped gymnasium environment
        """
        if self == GymWrapperEnum.NONE:
            return None
        return GymWrapperEnum.wrapper()[self](env, **wrapper_kwargs)
