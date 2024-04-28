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
        self.state = np.array([1, 2, 3, 4, 5])
        self.action = np.array([0])
        self.reward = np.array([1])
        self.next_state = np.array([2, 3, 4, 5, 6])
        self.done = np.array([True])
        self.log_prob = np.array([0.23456])
        self.value = np.array([0])

    def test_reset(self):
        """
        Tests the method reset().
        """
        # Fill values
        for i in range(self.max_size):
            self.replay_buffer.push(
                self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
            )
        self.replay_buffer.reset()

        # Check if no transition is in the ring buffer
        self.assertEqual(0, len(self.replay_buffer))

    def test_full(self):
        """
        Tests the method full().
        """
        # Check if the ring buffer is empty (full=False)
        self.assertFalse(self.replay_buffer.full())

        # Fill values
        for i in range(self.max_size):
            self.replay_buffer.push(
                self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
            )

        # Check if the ring buffer is full (full=True)
        self.assertTrue(self.replay_buffer.full())

    def test_filled(self):
        """
        Tests the method filled().
        """
        # Fill 3 values
        size = self.max_size - 2
        for i in range(size):
            self.replay_buffer.push(
                self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
            )

        # Check if ring buffer is filled with 3 (filled=True) and 5 values (filled=False)
        self.assertTrue(self.replay_buffer.filled(size))
        self.assertFalse(self.replay_buffer.filled(self.max_size))

    def test_push(self):
        """
        Tests the method push().
        """
        self.replay_buffer.push(
            self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
        )

        self.assertEqual(1, len(self.replay_buffer))

    def test_get(self):
        """
        Tests the method get().
        """
        size = self.max_size - 2
        for i in range(self.max_size):
            self.replay_buffer.push(
                self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
            )

        samples = self.replay_buffer.get(batch_size=size)

        self.assertIsInstance(samples.state, torch.Tensor)
        self.assertIsInstance(samples.action, torch.Tensor)
        self.assertIsInstance(samples.reward, torch.Tensor)
        self.assertIsInstance(samples.next_state, torch.Tensor)
        self.assertIsInstance(samples.done, torch.Tensor)

        self.assertEqual((size, 5), samples.state.shape)
        self.assertEqual((size, 1), samples.action.shape)
        self.assertEqual((size, 1), samples.reward.shape)
        self.assertEqual((size, 5), samples.next_state.shape)
        self.assertEqual((size, 1), samples.done.shape)

    def test_sample(self):
        """
        Tests the method sample().
        """
        size = self.max_size - 2
        for i in range(self.max_size):
            self.replay_buffer.push(
                self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
            )

        samples = self.replay_buffer.sample(batch_size=size)

        self.assertIsInstance(samples.state, torch.Tensor)
        self.assertIsInstance(samples.action, torch.Tensor)
        self.assertIsInstance(samples.reward, torch.Tensor)
        self.assertIsInstance(samples.next_state, torch.Tensor)
        self.assertIsInstance(samples.done, torch.Tensor)

        self.assertEqual((size, 5), samples.state.shape)
        self.assertEqual((size, 1), samples.action.shape)
        self.assertEqual((size, 1), samples.reward.shape)
        self.assertEqual((size, 5), samples.next_state.shape)
        self.assertEqual((size, 1), samples.done.shape)

    def test_len(self):
        """
        Tests the magic function __len__.
        """
        self.assertEqual(0, len(self.replay_buffer))

        # Fill values
        for i in range(self.max_size):
            self.replay_buffer.push(
                self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
            )

        self.assertEqual(self.max_size, len(self.replay_buffer))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Let push some values to the replay buffer
        self.replay_buffer.push(
            self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
        )
        self.replay_buffer.push(
            self.state, self.action, self.reward, self.next_state, self.done, self.log_prob, self.value
        )

        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.replay_buffer, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            replay_buffer = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(self.replay_buffer.max_size, replay_buffer.max_size)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[0].state, replay_buffer.memory[0].state)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[0].action, replay_buffer.memory[0].action)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[0].reward, replay_buffer.memory[0].reward)
        np.testing.assert_array_almost_equal(
            self.replay_buffer.memory[0].next_state,
            replay_buffer.memory[0].next_state
        )
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[0].done, replay_buffer.memory[0].done)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[0].log_prob, replay_buffer.memory[0].log_prob)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[0].value, replay_buffer.memory[0].value)

        np.testing.assert_array_almost_equal(self.replay_buffer.memory[1].state, replay_buffer.memory[1].state)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[1].action, replay_buffer.memory[1].action)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[1].reward, replay_buffer.memory[1].reward)
        np.testing.assert_array_almost_equal(
            self.replay_buffer.memory[1].next_state,
            replay_buffer.memory[1].next_state
        )
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[1].done, replay_buffer.memory[1].done)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[1].log_prob, replay_buffer.memory[1].log_prob)
        np.testing.assert_array_almost_equal(self.replay_buffer.memory[1].value, replay_buffer.memory[1].value)


if __name__ == '__main__':
    unittest.main()
