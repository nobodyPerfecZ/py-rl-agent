import unittest

import numpy as np
import torch
import yaml

from PyRLAgent.common.strategy.epsilon_greedy import LinearDecayEpsilonGreedy, ExponentialDecayEpsilonGreedy


class TestLinearDecayEpsilonGreedy(unittest.TestCase):
    """
    Tests the class LinearDecayEpsilonGreedy.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.strategy = LinearDecayEpsilonGreedy(epsilon_min=0.1, epsilon_max=1.0, steps=3)
        self.q_values = torch.tensor([
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
        ])
        self.state = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ])
        self.action = np.array([0, 0, 0, 0, 0])
        self.reward = np.array([1, 1, 1, 1, 1])
        self.next_state = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        self.done = np.array([False, False, False, False, False])

    def test_choose_action(self):
        """
        Tests the method choose_action().
        """
        # Takes a random action
        action = self.strategy.choose_action(self.state, self.q_values)
        self.assertTrue(np.array_equal(torch.tensor([4, 3, 0, 3, 4]), action))

    def test_update(self):
        """
        Tests the method update().
        """
        self.assertEqual(self.strategy.epsilon_max, self.strategy.epsilon)

        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)

        self.assertEqual(self.strategy.epsilon_min, self.strategy.epsilon)

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.strategy, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            strategy = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(strategy, self.strategy)


class TestExponentialDecayEpsilonGreedy(unittest.TestCase):
    """
    Tests the class ExponentialDecayEpsilonGreedy.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.strategy = ExponentialDecayEpsilonGreedy(epsilon_min=0.1, epsilon_max=1.0, decay_factor=0.5)
        self.q_values = torch.tensor([
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
            [-0.9641, 0.9936, 0.3008, -0.5313, -0.7851],
        ])
        self.state = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ])
        self.action = np.array([0, 0, 0, 0, 0])
        self.reward = np.array([1, 1, 1, 1, 1])
        self.next_state = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        self.done = np.array([False, False, False, False, False])

    def test_choose_action(self):
        """
        Tests the method choose_action().
        """
        action = self.strategy.choose_action(self.state, self.q_values)
        self.assertTrue(np.array_equal(torch.tensor([4, 3, 0, 3, 4]), action))

    def test_update(self):
        """
        Tests the method update().
        """
        self.assertEqual(self.strategy.epsilon_max, self.strategy.epsilon)

        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, self.action, self.reward, self.next_state, self.done)

        self.assertEqual(self.strategy.epsilon_min, self.strategy.epsilon)

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.strategy, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            strategy = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(strategy, self.strategy)


if __name__ == '__main__':
    unittest.main()
