import unittest

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from pyrlagent.torch.network.actor_critic import *
from pyrlagent.torch.util.env import get_vector_env


class TestMLPCategoricalActorCriticNetwork(unittest.TestCase):
    """Tests the class MLPCategoricalActorCriticNetwork."""

    def setUp(self):
        self.num_envs = 10
        self.env = get_vector_env(
            env_id="CartPole-v1", num_envs=self.num_envs, render_mode=None
        )
        self.network = MLPCategoricalActorCriticNetwork(
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=self.env.single_action_space.n,
            hidden_features=[16, 16],
            activation=nn.ReLU,
        )

    def tearDown(self) -> None:
        return self.env.close()

    def test_distribution(self):
        """Tests the distribution() method."""
        obs, _ = self.env.reset()
        pi = self.network.distribution(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(pi, Categorical)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.n), pi.param_shape
        )

    def test_log_prob(self):
        """Tests the log_prob() method."""
        obs, _ = self.env.reset()
        pi = self.network.distribution(torch.from_numpy(obs).to(torch.float32))
        log_prob = self.network.log_prob(
            pi, torch.from_numpy(self.env.action_space.sample())
        )

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual((self.num_envs,), log_prob.shape)

    def test_critic_value(self):
        """Tests the critic_value() method."""
        obs, _ = self.env.reset()
        value = self.network.critic_value(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual((self.num_envs,), value.shape)

    def test_forward(self):
        """Tests the forward() method."""
        obs, _ = self.env.reset()
        pi, value = self.network.forward(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(pi, Categorical)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.n), pi.param_shape
        )

        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual((self.num_envs,), value.shape)


class TestCNNCategoricalActorCriticNetwork(unittest.TestCase):
    """Tests the class CNNCategoricalActorNetwork."""

    # TODO: Implement here
    pass


class TestMLPGaussianActorCriticNetwork(unittest.TestCase):
    """Tests the class MLPGaussianActorCriticNetwork."""

    def setUp(self):
        self.num_envs = 10
        self.env = get_vector_env(
            env_id="Ant-v4", num_envs=self.num_envs, render_mode=None
        )
        self.network = MLPGaussianActorCriticNetwork(
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=self.env.single_action_space.shape[0],
            hidden_features=[16, 16],
            activation=nn.ReLU,
        )

    def tearDown(self) -> None:
        return self.env.close()

    def test_distribution(self):
        """Tests the distribution() method."""
        obs, _ = self.env.reset()
        pi = self.network.distribution(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(pi, Normal)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), pi.loc.shape
        )
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), pi.scale.shape
        )

    def test_log_prob(self):
        """Tests the log_prob() method."""
        obs, _ = self.env.reset()
        pi = self.network.distribution(torch.from_numpy(obs).to(torch.float32))
        log_prob = self.network.log_prob(
            pi, torch.from_numpy(self.env.action_space.sample())
        )

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual((self.num_envs,), log_prob.shape)

    def test_forward(self):
        """Tests the forward() method."""
        obs, _ = self.env.reset()
        pi, value = self.network.forward(torch.from_numpy(obs).to(torch.float32))

        self.assertIsInstance(pi, Normal)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), pi.loc.shape
        )
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), pi.scale.shape
        )

        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual((self.num_envs,), value.shape)


class TestCNNGaussianActorCriticNetwork(unittest.TestCase):
    """Tests the class CNNGaussianActorCriticNetwork."""

    # TODO: Implement here
    pass


if __name__ == "__main__":
    unittest.main()
