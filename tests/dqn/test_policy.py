import unittest
import torch.nn as nn
import numpy as np
import gymnasium
import torch
import yaml

from PyRLAgent.common.strategy.epsilon_greedy import EpsilonGreedy
from PyRLAgent.dqn.policy import QNetwork


class TestQNetwork(unittest.TestCase):
    """
    Tests the class QNetwork.
    """

    def setUp(self):
        env = gymnasium.make("CartPole-v1")
        self.model = QNetwork(
            observation_space=env.observation_space,
            action_space=env.action_space,
            architecture=[16, 32],
            activation_fn=nn.ReLU(),
            bias=True,
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 3},
        )

    def test_forward(self):
        """
        Tests the method forward().
        """
        result = self.model.forward(np.random.random(size=(4,)).astype(dtype=np.float32))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual((2,), result.size())

    def test_predict_deterministic(self):
        """
        Tests the method predict() with deterministic=True.
        """
        state = np.random.random(size=(4,)).astype(dtype=np.float32)
        q_values = self.model.forward(state)
        action = self.model.predict(state, deterministic=True)

        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(torch.argmax(q_values), action)

    def test_predict_non_deterministic(self):
        """
        Tests the method predict() with deterministic=False.
        """
        state = np.random.random(size=(4,)).astype(dtype=np.float32)
        action = self.model.predict(state, deterministic=False)

        self.assertIsInstance(action, torch.Tensor)

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.model, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            model = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal overriding __eq__ causes much slower training
        self.assertEqual(str(model), str(self.model))


if __name__ == '__main__':
    unittest.main()
