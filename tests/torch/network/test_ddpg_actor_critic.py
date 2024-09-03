import unittest

import torch
import torch.nn as nn

from pyrlagent.torch.network import MLPContinuousDDPGActorCriticNetwork
from pyrlagent.torch.util import get_vector_env


class TestMLPContinuousDDPGActorCriticNetwork(unittest.TestCase):
    """Tests the class MLPContinuousDDPGActorCriticNetwork."""

    def setUp(self):
        self.num_envs = 10
        self.env = get_vector_env(
            env_id="Ant-v5",
            num_envs=self.num_envs,
            device="cpu",
            render_mode=None,
        )
        self.noise_scale = 0.1
        self.low_action = -1.0
        self.high_action = 1.0
        self.network = MLPContinuousDDPGActorCriticNetwork(
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=self.env.single_action_space.shape[0],
            hidden_features=[16, 16],
            activation=nn.ReLU,
            noise_scale=self.noise_scale,
            low_action=self.low_action,
            high_action=self.high_action,
        )

    def tearDown(self) -> None:
        return self.env.close()

    def test_action(self):
        """Tests the action() method."""
        obs, _ = self.env.reset()
        action = self.network.action(obs)

        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), action.shape
        )

    def test_q_value(self):
        """Tests the q_value() method."""
        obs, _ = self.env.reset()
        q_value = self.network.q_value(obs, self.network.action(obs))

        self.assertIsInstance(q_value, torch.Tensor)
        self.assertEqual((self.num_envs,), q_value.shape)

    def test_forward(self):
        """Tests the forward() method."""
        obs, _ = self.env.reset()
        action, q_value = self.network.forward(obs)

        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), action.shape
        )

        self.assertIsInstance(q_value, torch.Tensor)
        self.assertEqual((self.num_envs,), q_value.shape)


if __name__ == "__main__":
    unittest.main()
