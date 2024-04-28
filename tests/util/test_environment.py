import unittest

import gymnasium as gym

from PyRLAgent.util.environment import get_env, get_vector_env, transform_env
from PyRLAgent.wrapper.observation import NormalizeObservationWrapper
from PyRLAgent.wrapper.reward import NormalizeRewardWrapper


class TestEnvironment(unittest.TestCase):
    """
    Tests all methods from util.environment.
    """

    def setUp(self):
        self.env_name = "CartPole-v1"
        self.num_envs = 10
        self.wrappers = [NormalizeObservationWrapper, NormalizeRewardWrapper]

    def test_get_env(self):
        """
        Tests the method get_env().
        """
        env = get_env(self.env_name, render_mode=None)

        # Check if both environments are the right ones
        self.assertEqual(self.env_name, env.spec.id)

        env.close()

    def test_get_vector_env(self):
        """
        Tests the method get_vector_env().
        """
        envs = get_vector_env(self.env_name, 1, render_mode=None)

        self.assertIsInstance(envs, gym.experimental.vector.VectorEnv)
        envs.close()

    def test_transform_env(self):
        """
        Tests the method transform_env().
        """
        env = get_env(self.env_name, render_mode=None)
        env = transform_env(env, self.wrappers)

        self.assertIsInstance(env, NormalizeRewardWrapper)

        env.close()


if __name__ == '__main__':
    unittest.main()
