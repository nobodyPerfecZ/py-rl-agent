import unittest
import numpy as np
import torch
import yaml

from PyRLAgent.common.strategy.ucb import UCB


class TestUCB(unittest.TestCase):
    """
    Tests the class UCB.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.strategy = UCB(c=1.0)
        self.state = np.array([0, 1, 2, 3])
        self.next_state = np.array([1, 2, 3, 4])
        self.reward = 1
        self.done = False
        self.q_values = torch.tensor([-0.9641, 0.9936, 0.3008, -0.5313, -0.7851])

    def test_choose_action(self):
        """
        Tests the method choose_action().
        """
        action = self.strategy.choose_action(self.state, self.q_values)

        self.assertTrue(np.array_equal(torch.tensor(1), action))

    def test_update(self):
        """
        Tests the method update().
        """
        action = self.strategy.choose_action(self.state, self.q_values)
        self.strategy.update(self.state, 0, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 1, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 2, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 3, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 4, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 0, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 1, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 1, self.reward, self.next_state, self.done)

        self.assertTrue(np.array_equal(torch.tensor(9), self.strategy.timestep))
        self.assertTrue(np.array_equal(torch.tensor([2, 3, 1, 1, 1]), self.strategy.counter[self.state.tobytes()]))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Lets add some values to the counter
        action = self.strategy.choose_action(self.state, self.q_values)
        self.strategy.update(self.state, 0, self.reward, self.next_state, self.done)
        self.strategy.update(self.state, 1, self.reward, self.next_state, self.done)

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
