import unittest

import gymnasium as gym
from torch import nn

from PyRLAgent.algorithm.policy import QNetwork, QDuelingNetwork, QProbNetwork, ActorCriticNetwork
from PyRLAgent.enum.policy import PolicyEnum


class TestPolicy(unittest.TestCase):
    """
    Tests the enum class Policy
    """

    def setUp(self):
        self.env = gym.make("CartPole-v1", render_mode=None)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.wrapper = {
            PolicyEnum.Q_NET: QNetwork,
            PolicyEnum.Q_DUELING_NET: QDuelingNetwork,
            PolicyEnum.Q_PROB_NET: QProbNetwork,
            PolicyEnum.ACTOR_CRITIC_NET: ActorCriticNetwork,
        }

        self.policy_kwargs1 = {
            "architecture": [64],
            "activation_fn": nn.Tanh(),
            "output_activation_fn": None,
            "bias": True,
            "strategy_type": "exp-epsilon",
            "strategy_kwargs": {"epsilon_min": 0.1, "epsilon_max": 1.0, "decay_factor": 0.95},
        }
        self.policy_kwargs2 = {
            "feature_architecture": [64],
            "feature_activation_fn": None,
            "feature_output_activation_fn": nn.Tanh(),
            "value_architecture": [64],
            "value_activation_fn": nn.Tanh(),
            "value_output_activation_fn": None,
            "advantage_architecture": [64, 64],
            "advantage_activation_fn": nn.Tanh(),
            "advantage_output_activation_fn": None,
            "bias": True,
            "strategy_type": "exp-epsilon",
            "strategy_kwargs": {"epsilon_min": 0.1, "epsilon_max": 1.0, "decay_factor": 0.95},
        }
        self.policy_kwargs3 = {
            "Q_min": -10,
            "Q_max": 10,
            "num_atoms": 51,
            "architecture": [64],
            "activation_fn": nn.Tanh(),
            "output_activation_fn": None,
            "bias": True,
            "strategy_type": "exp-epsilon",
            "strategy_kwargs": {"epsilon_min": 0.1, "epsilon_max": 1.0, "decay_factor": 0.95},
        }
        self.policy_kwargs4 = {
            "actor_architecture": [128],
            "actor_activation_fn": nn.Tanh(),
            "actor_output_activation_fn": None,
            "critic_architecture": [128],
            "critic_activation_fn": nn.Tanh(),
            "critic_output_activation_fn": None,
            "bias": True
        }

    def test_wrapper(self):
        """
        Tests the class method wrapper().
        """
        self.assertDictEqual(self.wrapper, PolicyEnum.wrapper())

    def test_to(self):
        """
        Tests the method to().
        """
        policy1 = PolicyEnum.Q_NET.to(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **self.policy_kwargs1
        )
        policy2 = PolicyEnum.Q_DUELING_NET.to(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **self.policy_kwargs2
        )
        policy3 = PolicyEnum.Q_PROB_NET.to(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **self.policy_kwargs3
        )
        policy4 = PolicyEnum.ACTOR_CRITIC_NET.to(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **self.policy_kwargs4
        )

        self.assertIsInstance(policy1, QNetwork)
        self.assertIsInstance(policy2, QDuelingNetwork)
        self.assertIsInstance(policy3, QProbNetwork)
        self.assertIsInstance(policy4, ActorCriticNetwork)


if __name__ == '__main__':
    unittest.main()
