from enum import Enum
from typing import Type

from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.common.strategy.epsilon_greedy import LinearDecayEpsilonGreedy, ExponentialDecayEpsilonGreedy
from PyRLAgent.common.strategy.greedy import Greedy
from PyRLAgent.common.strategy.random import Random
from PyRLAgent.enum.abstract_enum import AbstractStrEnum


class StrategyEnum(AbstractStrEnum):
    """
    An enumeration of supported strategy types.
    """
    RANDOM = "random"
    GREEDY = "greedy"
    LINEAR_EPSILON = "linear-epsilon"
    EXP_EPSILON = "exp-epsilon"

    @classmethod
    def wrapper(cls) -> dict[Enum, Type[Strategy]]:
        return {
            cls.RANDOM: Random,
            cls.GREEDY: Greedy,
            cls.LINEAR_EPSILON: LinearDecayEpsilonGreedy,
            cls.EXP_EPSILON: ExponentialDecayEpsilonGreedy,
        }

    def to(self, **strategy_kwargs) -> Strategy:
        return StrategyEnum.wrapper()[self](**strategy_kwargs)
