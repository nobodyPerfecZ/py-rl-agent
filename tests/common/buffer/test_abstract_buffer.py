import unittest

import numpy as np
import torch

from PyRLAgent.common.buffer.abstract_buffer import Transition


class TestTransition(unittest.TestCase):
    """
    Tests the dataclass Transition.
    """
    def setUp(self):
        self.state = np.array([1, 2, 3, 4, 5])
        self.action = np.array([0])
        self.reward = np.array([1])
        self.next_state = np.array([2, 3, 4, 5, 6])
        self.done = np.array([True])
        self.log_prob = np.array([0.23456])
        self.value = np.array([0])

        self.transitions = [
            Transition(self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value),
            Transition(self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value),
            Transition(self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value),
        ]

    def test_create(self):
        """
        Tests the static method create().
        """
        size = len(self.transitions)
        transition = Transition.create(self.transitions)

        self.assertIsInstance(transition.state, torch.Tensor)
        self.assertIsInstance(transition.action, torch.Tensor)
        self.assertIsInstance(transition.reward, torch.Tensor)
        self.assertIsInstance(transition.next_state, torch.Tensor)
        self.assertIsInstance(transition.done, torch.Tensor)

        self.assertEqual((size, 5), transition.state.shape)
        self.assertEqual((size, 1), transition.action.shape)
        self.assertEqual((size, 1), transition.reward.shape)
        self.assertEqual((size, 5), transition.next_state.shape)
        self.assertEqual((size, 1), transition.done.shape)


if __name__ == '__main__':
    unittest.main()
