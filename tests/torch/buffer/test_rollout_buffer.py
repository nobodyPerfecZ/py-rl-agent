import unittest

import torch

from pyrlagent.torch.buffer import RolloutBuffer


class TestRolloutBuffer(unittest.TestCase):
    """Tests the class RolloutBuffer()."""

    def setUp(self):
        self.obs_dim = 3
        self.act_dim = 1
        self.env_dim = 1
        self.max_size = 3
        self.device = "cpu"
        self.trajectory = RolloutBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            env_dim=self.env_dim,
            max_size=self.max_size,
            device=self.device,
        )

        self.state = torch.tensor([[1.0, 2.0, 3.0]], device=self.device)
        self.action = torch.tensor([0.0], device=self.device)
        self.reward = torch.tensor([1.0], device=self.device)
        self.next_state = torch.tensor([[4.0, 5.0, 6.0]], device=self.device)
        self.done = torch.tensor([0.0], device=self.device)
        self.log_prob = torch.tensor([0.1234], device=self.device)
        self.value = torch.tensor([0.5], device=self.device)
        self.next_value = torch.tensor([1.0], device=self.device)

    def test_reset(self):
        """Tests the reset() method."""
        # Push a transition
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )

        # Reset the trajectory
        self.trajectory.reset()

        self.assertEqual(self.trajectory.ptr, -1)
        self.assertEqual(len(self.trajectory), 0)
        torch.testing.assert_close(
            torch.zeros(
                (self.max_size, self.env_dim, self.obs_dim), device=self.device
            ),
            self.trajectory.trajectory.state,
        )
        torch.testing.assert_close(
            torch.zeros((self.max_size, self.env_dim), device=self.device),
            self.trajectory.trajectory.action,
        )
        torch.testing.assert_close(
            torch.zeros((self.max_size, self.env_dim), device=self.device),
            self.trajectory.trajectory.reward,
        )
        torch.testing.assert_close(
            torch.zeros(
                (self.max_size, self.env_dim, self.obs_dim), device=self.device
            ),
            self.trajectory.trajectory.next_state,
        )
        torch.testing.assert_close(
            torch.zeros((self.max_size, self.env_dim), device=self.device),
            self.trajectory.trajectory.done,
        )
        torch.testing.assert_close(
            torch.zeros((self.max_size, self.env_dim), device=self.device),
            self.trajectory.trajectory.log_prob,
        )
        torch.testing.assert_close(
            torch.zeros((self.max_size, self.env_dim), device=self.device),
            self.trajectory.trajectory.value,
        )
        torch.testing.assert_close(
            torch.zeros((self.max_size, self.env_dim), device=self.device),
            self.trajectory.trajectory.next_value,
        )

    def test_push(self):
        """Tests the push() method."""
        # Push a transition
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        self.assertEqual(self.trajectory.ptr, 0)
        self.assertEqual(len(self.trajectory), 1)
        torch.testing.assert_close(self.state, self.trajectory.trajectory.state[0])
        torch.testing.assert_close(self.action, self.trajectory.trajectory.action[0])
        torch.testing.assert_close(self.reward, self.trajectory.trajectory.reward[0])
        torch.testing.assert_close(
            self.next_state, self.trajectory.trajectory.next_state[0]
        )
        torch.testing.assert_close(self.done, self.trajectory.trajectory.done[0])
        torch.testing.assert_close(
            self.log_prob, self.trajectory.trajectory.log_prob[0]
        )
        torch.testing.assert_close(self.value, self.trajectory.trajectory.value[0])
        torch.testing.assert_close(
            self.next_value, self.trajectory.trajectory.next_value[0]
        )

        # Push another transition
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        self.assertEqual(self.trajectory.ptr, 1)
        self.assertEqual(len(self.trajectory), 2)
        torch.testing.assert_close(self.state, self.trajectory.trajectory.state[1])
        torch.testing.assert_close(self.action, self.trajectory.trajectory.action[1])
        torch.testing.assert_close(self.reward, self.trajectory.trajectory.reward[1])
        torch.testing.assert_close(
            self.next_state, self.trajectory.trajectory.next_state[1]
        )
        torch.testing.assert_close(self.done, self.trajectory.trajectory.done[1])
        torch.testing.assert_close(
            self.log_prob, self.trajectory.trajectory.log_prob[1]
        )
        torch.testing.assert_close(self.value, self.trajectory.trajectory.value[1])
        torch.testing.assert_close(
            self.next_value, self.trajectory.trajectory.next_value[1]
        )

        # Push another transition
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        self.assertEqual(self.trajectory.ptr, 2)
        self.assertEqual(len(self.trajectory), 3)
        torch.testing.assert_close(self.state, self.trajectory.trajectory.state[2])
        torch.testing.assert_close(self.action, self.trajectory.trajectory.action[2])
        torch.testing.assert_close(self.reward, self.trajectory.trajectory.reward[2])
        torch.testing.assert_close(
            self.next_state, self.trajectory.trajectory.next_state[2]
        )
        torch.testing.assert_close(self.done, self.trajectory.trajectory.done[2])
        torch.testing.assert_close(
            self.log_prob, self.trajectory.trajectory.log_prob[2]
        )
        torch.testing.assert_close(self.value, self.trajectory.trajectory.value[2])
        torch.testing.assert_close(
            self.next_value, self.trajectory.trajectory.next_value[2]
        )

        # Push another transition
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        self.assertEqual(self.trajectory.ptr, 0)
        self.assertEqual(len(self.trajectory), 1)
        torch.testing.assert_close(self.state, self.trajectory.trajectory.state[0])
        torch.testing.assert_close(self.action, self.trajectory.trajectory.action[0])
        torch.testing.assert_close(self.reward, self.trajectory.trajectory.reward[0])
        torch.testing.assert_close(
            self.next_state, self.trajectory.trajectory.next_state[0]
        )
        torch.testing.assert_close(self.done, self.trajectory.trajectory.done[0])
        torch.testing.assert_close(
            self.log_prob, self.trajectory.trajectory.log_prob[0]
        )
        torch.testing.assert_close(self.value, self.trajectory.trajectory.value[0])
        torch.testing.assert_close(
            self.next_value, self.trajectory.trajectory.next_value[0]
        )

    def test_sample(self):
        """Tests the sample() method."""
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )
        self.trajectory.push(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            done=self.done,
            log_prob=self.log_prob,
            value=self.value,
            next_value=self.next_value,
        )

        # Get the trajectory
        sample = self.trajectory.sample(3)
        torch.testing.assert_close(
            torch.unsqueeze(torch.vstack((self.state, self.state, self.state)), dim=1),
            sample.state,
        )
        torch.testing.assert_close(
            torch.vstack((self.action, self.action, self.action)),
            sample.action,
        )
        torch.testing.assert_close(
            torch.vstack((self.reward, self.reward, self.reward)),
            sample.reward,
        )
        torch.testing.assert_close(
            torch.unsqueeze(
                torch.vstack((self.next_state, self.next_state, self.next_state)),
                axis=1,
            ),
            sample.next_state,
        )
        torch.testing.assert_close(
            torch.vstack((self.done, self.done, self.done)),
            sample.done,
        )
        torch.testing.assert_close(
            torch.vstack((self.log_prob, self.log_prob, self.log_prob)),
            sample.log_prob,
        )
        torch.testing.assert_close(
            torch.vstack((self.value, self.value, self.value)),
            sample.value,
        )
        torch.testing.assert_close(
            torch.vstack((self.next_value, self.next_value, self.next_value)),
            sample.next_value,
        )


if __name__ == "__main__":
    unittest.main()
