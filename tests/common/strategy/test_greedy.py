import unittest

import numpy as np
import torch
import yaml

from PyRLAgent.common.strategy.greedy import Greedy


class TestGreedy(unittest.TestCase):
    """
    Tests the class Greedy.
    """

    def setUp(self):
        self.strategy = Greedy()
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

    def test_choose_action(self):
        """
        Tests the method choose_action().
        """
        action = self.strategy.choose_action(self.state, self.q_values)

        self.assertTrue(np.array_equal(torch.tensor([1, 1, 1, 1, 1]), action))

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
