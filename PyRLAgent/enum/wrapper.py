from enum import Enum
from typing import Optional, Type, Union

import gymnasium as gym

from PyRLAgent.enum.abstract_enum import AbstractStrEnum
from PyRLAgent.util.environment import get_env, transform_env, get_vector_env
from PyRLAgent.wrapper.observation import NormalizeObservationWrapper
from PyRLAgent.wrapper.reward import NormalizeRewardWrapper


class GymWrapperEnum(AbstractStrEnum):
    """
    An enumeration of supported wrapper types.
    """
    NONE = "none"
    NORMALIZE_OBSERVATION = "normalize-observation"
    NORMALIZE_REWARD = "normalize-reward"

    @classmethod
    def wrapper(cls) -> dict[Enum, Type[gym.Wrapper]]:
        return {
            cls.NONE: None,
            cls.NORMALIZE_OBSERVATION: NormalizeObservationWrapper,
            cls.NORMALIZE_REWARD: NormalizeRewardWrapper,
        }

    def to(self, env: gym.Env, **wrapper_kwargs) -> Optional[gym.Env]:
        if self == GymWrapperEnum.NONE:
            return None
        return GymWrapperEnum.wrapper()[self](env=env, **wrapper_kwargs)

    @staticmethod
    def create_env(
            name: str,
            wrappers: Union[list[str], list["GymWrapperEnum"]],
            render_mode: Optional[str] = None
    ) -> gym.Env:
        # TODO: Add documentation
        # Convert str to enums
        wrappers = [GymWrapperEnum(wrapper) for wrapper in wrappers]

        # Convert enum to the classes
        wrappers = [GymWrapperEnum.wrapper()[wrapper] for wrapper in wrappers]

        # Remove all nones from the wrappers
        wrappers = [wrapper for wrapper in wrappers if wrapper is not None]

        return transform_env(get_env(name, render_mode=render_mode), wrappers)

    @staticmethod
    def create_vector_env(
            name: str,
            num_envs: int,
            wrappers: Union[list[str], list["GymWrapperEnum"]],
            render_mode: Optional[str] = None
    ):
        # TODO: Add documentation
        # Convert str to enums
        wrappers = [GymWrapperEnum(wrapper) for wrapper in wrappers]

        # Convert enum to the classes
        wrappers = [GymWrapperEnum.wrapper()[wrapper] for wrapper in wrappers]

        # Remove all nones from the wrappers
        wrappers = [wrapper for wrapper in wrappers if wrapper is not None]

        return transform_env(get_vector_env(name, num_envs, render_mode=render_mode), wrappers)
