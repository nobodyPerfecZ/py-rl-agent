import unittest

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv

from PyRLAgent.enum.wrapper import GymWrapperEnum
from PyRLAgent.wrapper.observation import NormalizeObservationWrapper
from PyRLAgent.wrapper.reward import NormalizeRewardWrapper


class TestGymWrapperEnum(unittest.TestCase):
    """
    Tests the enum class GymWrapperEnum.
    """

    def setUp(self):
        self.env = gym.make("CartPole-v1", render_mode=None)

        self.wrapper = {
            GymWrapperEnum.NONE: None,
            GymWrapperEnum.NORMALIZE_OBSERVATION: NormalizeObservationWrapper,
            GymWrapperEnum.NORMALIZE_REWARD: NormalizeRewardWrapper,
        }

        self.wrapper_kwargs1 = {}
        self.wrapper_kwargs2 = {}
        self.wrapper_kwargs3 = {}

    def test_wrapper(self):
        """
        Tests the method test_wrapper().
        """
        self.assertDictEqual(self.wrapper, GymWrapperEnum.wrapper())

    def test_to(self):
        """
        Tests the method to().
        """
        env1 = GymWrapperEnum.NONE.to(env=self.env, **self.wrapper_kwargs1)
        env2 = GymWrapperEnum.NORMALIZE_OBSERVATION.to(env=self.env, **self.wrapper_kwargs1)
        env3 = GymWrapperEnum.NORMALIZE_REWARD.to(env=self.env, **self.wrapper_kwargs1)

        self.assertIsNone(env1)
        self.assertIsInstance(env2, NormalizeObservationWrapper)
        self.assertIsInstance(env3, NormalizeRewardWrapper)

    def test_create_env(self):
        """
        Tests the static method create_env().
        """
        env1 = GymWrapperEnum.create_env(
            name="CartPole-v1",
            wrappers=["none", "normalize-observation", "normalize-reward"],
            render_mode=None
        )

        env2 = GymWrapperEnum.create_env(
            name="CartPole-v1",
            wrappers=["none"],
            render_mode=None
        )

        env3 = GymWrapperEnum.create_env(
            name="CartPole-v1",
            wrappers=["normalize-observation"],
            render_mode=None
        )

        self.assertIsInstance(env1, NormalizeRewardWrapper)
        self.assertIsNotNone(env2, CartPoleEnv)
        self.assertIsInstance(env3, NormalizeObservationWrapper)


if __name__ == '__main__':
    unittest.main()
