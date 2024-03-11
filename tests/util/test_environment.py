import unittest

from PyRLAgent.util.environment import get_env


class TestEnvironment(unittest.TestCase):
    """
    Tests all methods from util.environment.
    """

    def setUp(self):
        self.env_name = "CartPole-v1"

    def test_get_env(self):
        """
        Tests the method get_env().
        """
        env, render_env = get_env(self.env_name, return_render=True)

        # Check if both environments are the right ones
        self.assertEqual(self.env_name, env.spec.id)
        self.assertEqual(self.env_name, render_env.spec.id)


if __name__ == '__main__':
    unittest.main()
