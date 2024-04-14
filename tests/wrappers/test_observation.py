import unittest

import gymnasium as gym
import numpy as np

from PyRLAgent.wrapper.observation import NormalizeObservationWrapper


class TestNormalizeObservationWrapper(unittest.TestCase):
    """
    Tests the class NormalizeObservationWrapper.
    """

    def setUp(self):
        self.env = NormalizeObservationWrapper(env=gym.make("CartPole-v1", render_mode=None))

    def tearDown(self):
        self.env.close()

    def test_step(self):
        """
        Tests the method step().
        """
        self.env.reset()

        observation, reward, terminated, truncated, info = self.env.step(0)

        self.assertFalse(np.allclose(info["observation"], observation))

    def test_reset(self):
        """
        Tests the method reset().
        """
        observation, info = self.env.reset()

        np.testing.assert_almost_equal(np.array([0., 0., 0., 0.]), observation)
        self.assertFalse(np.allclose(info["observation"], observation))


if __name__ == '__main__':
    unittest.main()
