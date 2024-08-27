import unittest

import torch
import torch.nn as nn

from pyrlagent.torch.network.critic import MLPCriticNetwork
from pyrlagent.torch.util.env import get_vector_env


class TestMLPCriticNetwork(unittest.TestCase):
    """Tests the class MLPCriticNetwork."""

    def setUp(self):
        self.num_envs = 10
        self.env = get_vector_env(
            env_id="CartPole-v1", num_envs=self.num_envs, render_mode=None
        )
        self.network = MLPCriticNetwork(
            obs_dim=self.env.single_observation_space.shape[0],
            hidden_features=[16, 16],
            activation=nn.ReLU,
        )

    def tearDown(self) -> None:
        return self.env.close()

    def test_critic_value(self):
        """Tests the critic_value() method."""
        obs, _ = self.env.reset()
        value = self.network.critic_value(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual((self.num_envs,), value.shape)

    def test_forward(self):
        """Tests the forward() method."""
        obs, _ = self.env.reset()
        value = self.network.critic_value(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual((self.num_envs,), value.shape)


class TestCNNCriticNetwork(unittest.TestCase):
    """Tests the class CNNCriticNetwork."""

    # TODO: Implement here
    pass


if __name__ == "__main__":
    unittest.main()
