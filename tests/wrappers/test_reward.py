import unittest

import gymnasium as gym
import numpy as np

from PyRLAgent.wrapper.reward import NormalizeRewardWrapper


class TestNormalizeRewardWrapper(unittest.TestCase):
    """
    Tests the class NormalizeRewardWrapper.
    """

    def setUp(self):
        self.env = NormalizeRewardWrapper(env=gym.make("CartPole-v1", render_mode=None))

    def tearDown(self):
        self.env.close()

    def test_step(self):
        """
        Tests the method step().
        """
        self.env.reset()

        observation, reward, terminated, truncated, info = self.env.step(0)

        self.assertFalse(np.allclose(info["reward"], reward))

    def test_reset(self):
        """
        Tests the method reset().
        """
        _, info = self.env.reset()

        self.assertIsNone(info["reward"])


if __name__ == '__main__':
    unittest.main()
