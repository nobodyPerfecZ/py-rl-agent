import unittest

import gymnasium as gym

from pyrlagent.torch.config import EnvConfig, create_env_eval, create_env_train


class TestEnvConfig(unittest.TestCase):
    """Tests the EnvConfig class."""

    def setUp(self):
        self.id = "CartPole-v1"
        self.kwargs = {}
        self.device = "cpu"

    def test_init(self):
        """Tests the __init__() method."""
        env_config = EnvConfig(id=self.id, kwargs=self.kwargs)
        self.assertEqual(env_config.id, self.id)
        self.assertEqual(env_config.kwargs, self.kwargs)

    def test_create_env_train(self):
        """Tests the create_env_train() method."""
        env_train = create_env_train(
            env_config=EnvConfig(id=self.id, kwargs=self.kwargs),
            num_envs=1,
            device=self.device,
        )
        self.assertIsInstance(env_train, gym.vector.VectorEnv)

    def test_create_env_eval(self):
        """Tests the create_env_eval() method."""
        env_eval = create_env_eval(
            env_config=EnvConfig(id=self.id, kwargs=self.kwargs),
            device=self.device,
        )
        self.assertIsInstance(env_eval, gym.Env)


if __name__ == "__main__":
    unittest.main()
