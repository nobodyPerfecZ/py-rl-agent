import unittest

import numpy as np
import torch

from pyrlagent.torch.experience.trajectory import Trajectory


class TestTrajectory(unittest.TestCase):
    """Tests the class Trajectory."""

    def setUp(self):
        self.state = np.array([[1, 2, 3], [4, 5, 6]])
        self.action = np.array([[0], [1]])
        self.reward = np.array([1, 1])
        self.next_state = np.array([[4, 5, 6], [7, 8, 9]])
        self.done = np.array([False, False])
        self.log_prob = np.array([0.1234, 0.5678])
        self.value = np.array([0.5, 1.0])
        self.next_value = np.array([1.0, 1.5])

    def test_numpy_init(self):
        """Tests the __init__() method with numpy arrays."""
        transition1 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
        )
        np.testing.assert_allclose(self.state, transition1.state)
        np.testing.assert_allclose(self.action, transition1.action)
        np.testing.assert_allclose(self.reward, transition1.reward)
        np.testing.assert_allclose(self.next_state, transition1.next_state)
        np.testing.assert_allclose(self.done, transition1.done)
        self.assertIsNone(transition1.log_prob)
        self.assertIsNone(transition1.value)
        self.assertIsNone(transition1.next_value)

        transition2 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        np.testing.assert_allclose(self.state, transition2.state)
        np.testing.assert_allclose(self.action, transition2.action)
        np.testing.assert_allclose(self.reward, transition2.reward)
        np.testing.assert_allclose(self.next_state, transition2.next_state)
        np.testing.assert_allclose(self.done, transition2.done)
        np.testing.assert_allclose(self.log_prob, transition2.log_prob)
        np.testing.assert_allclose(self.value, transition2.value)
        np.testing.assert_allclose(self.next_value, transition2.next_value)

    def test_torch_init(self):
        """Tests the __init__() method with torch tensors."""
        self.state = torch.tensor(self.state)
        self.action = torch.tensor(self.action)
        self.reward = torch.tensor(self.reward)
        self.next_state = torch.tensor(self.next_state)
        self.done = torch.tensor(self.done)
        self.log_prob = torch.tensor(self.log_prob)
        self.value = torch.tensor(self.value)
        self.next_value = torch.tensor(self.next_value)

        transition1 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
        )
        torch.testing.assert_close(self.state, transition1.state)
        torch.testing.assert_close(self.action, transition1.action)
        torch.testing.assert_close(self.reward, transition1.reward)
        torch.testing.assert_close(self.next_state, transition1.next_state)
        torch.testing.assert_close(self.done, transition1.done)
        self.assertIsNone(transition1.log_prob)
        self.assertIsNone(transition1.value)
        self.assertIsNone(transition1.next_value)

        transition2 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        torch.testing.assert_close(self.state, transition2.state)
        torch.testing.assert_close(self.action, transition2.action)
        torch.testing.assert_close(self.reward, transition2.reward)
        torch.testing.assert_close(self.next_state, transition2.next_state)
        torch.testing.assert_close(self.done, transition2.done)
        torch.testing.assert_close(self.log_prob, transition2.log_prob)
        torch.testing.assert_close(self.value, transition2.value)
        torch.testing.assert_close(self.next_value, transition2.next_value)

    def test_numpy_getitem(self):
        """Tests the __getitem__() method with numpy arrays."""
        transition1 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
        )
        transition1 = transition1[0]
        np.testing.assert_allclose(self.state[0], transition1.state)
        np.testing.assert_allclose(self.action[0], transition1.action)
        np.testing.assert_allclose(self.reward[0], transition1.reward)
        np.testing.assert_allclose(self.next_state[0], transition1.next_state)
        np.testing.assert_allclose(self.done[0], transition1.done)
        self.assertIsNone(transition1.log_prob)
        self.assertIsNone(transition1.value)
        self.assertIsNone(transition1.next_value)

        transition2 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        transition2 = transition2[0]
        np.testing.assert_allclose(self.state[0], transition2.state)
        np.testing.assert_allclose(self.action[0], transition2.action)
        np.testing.assert_allclose(self.reward[0], transition2.reward)
        np.testing.assert_allclose(self.next_state[0], transition2.next_state)
        np.testing.assert_allclose(self.done[0], transition2.done)
        np.testing.assert_allclose(self.log_prob[0], transition2.log_prob)
        np.testing.assert_allclose(self.value[0], transition2.value)
        np.testing.assert_allclose(self.next_value[0], transition2.next_value)

    def test_torch_getitem(self):
        """Tests the __getitem__() method with torch tensors."""
        self.state = torch.tensor(self.state)
        self.action = torch.tensor(self.action)
        self.reward = torch.tensor(self.reward)
        self.next_state = torch.tensor(self.next_state)
        self.done = torch.tensor(self.done)
        self.log_prob = torch.tensor(self.log_prob)
        self.value = torch.tensor(self.value)
        self.next_value = torch.tensor(self.next_value)

        transition1 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
        )
        transition1 = transition1[0]
        torch.testing.assert_close(self.state[0], transition1.state)
        torch.testing.assert_close(self.action[0], transition1.action)
        torch.testing.assert_close(self.reward[0], transition1.reward)
        torch.testing.assert_close(self.next_state[0], transition1.next_state)
        torch.testing.assert_close(self.done[0], transition1.done)
        self.assertIsNone(transition1.log_prob)
        self.assertIsNone(transition1.value)
        self.assertIsNone(transition1.next_value)

        transition2 = Trajectory(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        transition2 = transition2[0]
        torch.testing.assert_close(self.state[0], transition2.state)
        torch.testing.assert_close(self.action[0], transition2.action)
        torch.testing.assert_close(self.reward[0], transition2.reward)
        torch.testing.assert_close(self.next_state[0], transition2.next_state)
        torch.testing.assert_close(self.done[0], transition2.done)
        torch.testing.assert_close(self.log_prob[0], transition2.log_prob)
        torch.testing.assert_close(self.value[0], transition2.value)
        torch.testing.assert_close(self.next_value[0], transition2.next_value)


if __name__ == "__main__":
    unittest.main()
