import unittest
import numpy as np
import torch
import yaml

from PyRLAgent.common.buffer.ring_buffer import RingBuffer


class TestRingBuffer(unittest.TestCase):
    """
    Tests the class RingBuffer.
    """

    def setUp(self):
        self.max_size = 5
        self.replay_buffer = RingBuffer(max_size=self.max_size)

    def test_reset(self):
        """
        Tests the method reset().
        """
        # Fill values
        for i in range(self.max_size):
            self.replay_buffer.push(np.array([i]), i, i, np.array([i + 1]), False)
        self.replay_buffer.reset()

        self.assertEqual(0, len(self.replay_buffer))

    def test_full(self):
        """
        Tests the method full().
        """
        self.assertFalse(self.replay_buffer.full())

        # Fill values
        for i in range(self.max_size):
            self.replay_buffer.push(np.array([i]), i, i, np.array([i + 1]), False)

        self.assertTrue(self.replay_buffer.full())

    def test_filled(self):
        """
        Tests the method filled().
        """
        size = self.max_size - 2
        for i in range(size):
            self.replay_buffer.push(np.array([i]), i, i, np.array([i + 1]), False)

        self.assertTrue(self.replay_buffer.filled(size))
        self.assertFalse(self.replay_buffer.filled(self.max_size))

    def test_push(self):
        """
        Tests the method push().
        """
        self.replay_buffer.push(np.array([0]), 0, 0, np.array([0 + 1]), False)

        self.assertEqual(1, len(self.replay_buffer))

    def test_sample(self):
        """
        Tests the method sample().
        """
        size = self.max_size - 2
        for i in range(self.max_size):
            self.replay_buffer.push(np.array([i]), i, i, np.array([i + 1]), False)

        samples = self.replay_buffer.sample(batch_size=size)

        self.assertIsInstance(samples.states, torch.Tensor)
        self.assertIsInstance(samples.actions, torch.Tensor)
        self.assertIsInstance(samples.rewards, torch.Tensor)
        self.assertIsInstance(samples.next_states, torch.Tensor)
        self.assertIsInstance(samples.dones, torch.Tensor)

        self.assertEqual(size, len(samples.states))
        self.assertEqual(size, len(samples.actions))
        self.assertEqual(size, len(samples.rewards))
        self.assertEqual(size, len(samples.next_states))
        self.assertEqual(size, len(samples.dones))

    def test_len(self):
        """
        Tests the magic function __len__.
        """
        self.assertEqual(0, len(self.replay_buffer))

        # Fill values
        for i in range(self.max_size):
            self.replay_buffer.push(np.array([i]), i, i, np.array([i + 1]), False)

        self.assertEqual(self.max_size, len(self.replay_buffer))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Lets push some values to the replay buffer
        self.replay_buffer.push(np.array([0]), 0, 0, np.array([1]), False)
        self.replay_buffer.push(np.array([1]), 1, 1, np.array([2]), False)

        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.replay_buffer, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            replay_buffer = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(replay_buffer, self.replay_buffer)


if __name__ == '__main__':
    unittest.main()
