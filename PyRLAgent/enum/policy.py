from enum import Enum
from typing import Type

from gymnasium import spaces

from PyRLAgent.algorithm.policy import QNetwork, QDuelingNetwork, QProbNetwork, ActorCriticNetwork
from PyRLAgent.common.policy.abstract_policy import Policy
from PyRLAgent.enum.abstract_enum import AbstractStrEnum


class PolicyEnum(AbstractStrEnum):
    """
    An enumeration of supported policy types.
    """
    Q_NET = "q-net"
    Q_DUELING_NET = "q-dueling-net"
    Q_PROB_NET = "q-prob-net"
    ACTOR_CRITIC_NET = "actor-critic-net"

    @classmethod
    def wrapper(cls) -> dict[Enum, Type[Policy]]:
        return {
            cls.Q_NET: QNetwork,
            cls.Q_DUELING_NET: QDuelingNetwork,
            cls.Q_PROB_NET: QProbNetwork,
            cls.ACTOR_CRITIC_NET: ActorCriticNetwork,
        }

    def to(self, observation_space: spaces.Space, action_space: spaces.Space, **policy_kwargs) -> Policy:
        return PolicyEnum.wrapper()[self](
            observation_space=observation_space,
            action_space=action_space,
            **policy_kwargs
        )
