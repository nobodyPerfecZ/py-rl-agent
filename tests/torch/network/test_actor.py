import unittest

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from pyrlagent.torch.network import (
    MLPCategoricalActorNetwork,
    MLPGaussianActorNetwork,
)
from pyrlagent.torch.util import get_vector_env


class TestMLPCategoricalActorNetwork(unittest.TestCase):
    """Tests the class MLPCategoricalActorNetwork."""

    def setUp(self):
        self.num_envs = 10
        self.env = get_vector_env(
            env_id="CartPole-v1",
            num_envs=self.num_envs,
            device="cpu",
            render_mode=None,
        )
        self.network = MLPCategoricalActorNetwork(
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
        pi = self.network.distribution(obs)

        self.assertIsInstance(pi, Categorical)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.n), pi.param_shape
        )

    def test_log_prob(self):
        """Tests the log_prob() method."""
        obs, _ = self.env.reset()
        pi = self.network.distribution(obs)
        log_prob = self.network.log_prob(pi, pi.sample())

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual((self.num_envs,), log_prob.shape)

    def test_forward(self):
        """Tests the forward() method."""
        obs, _ = self.env.reset()
        pi = self.network.forward(obs)

        self.assertIsInstance(pi, Categorical)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.n), pi.param_shape
        )


class TestCNNCategoricalActorNetwork(unittest.TestCase):
    """Tests the class CNNCategoricalActorNetwork."""

    # TODO: Implement here
    pass


class TestMLPGaussianActorNetwork(unittest.TestCase):
    """Tests the class MLPGaussianActorNetwork."""

    def setUp(self):
        self.num_envs = 10
        self.env = get_vector_env(
            env_id="Ant-v5",
            num_envs=self.num_envs,
            device="cpu",
            render_mode=None,
        )
        self.network = MLPGaussianActorNetwork(
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
        pi = self.network.distribution(obs)

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
        pi = self.network.distribution(obs)
        log_prob = self.network.log_prob(pi, pi.sample())

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual((self.num_envs,), log_prob.shape)

    def test_forward(self):
        """Tests the forward() method."""
        obs, _ = self.env.reset()
        pi = self.network.forward(obs)

        self.assertIsInstance(pi, Normal)
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), pi.loc.shape
        )
        self.assertEqual(
            (self.num_envs, self.env.single_action_space.shape[0]), pi.scale.shape
        )


class TestCNNGaussianActorNetwork(unittest.TestCase):
    """Tests the class CNNGaussianActorNetwork."""

    # TODO: Implement here
    pass


if __name__ == "__main__":
    unittest.main()
