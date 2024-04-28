import unittest

import torch
from torch import nn

from PyRLAgent.algorithm.ddqn import ClippedDDQN, DDQN


class TestDDQN(unittest.TestCase):
    """
    Tests the class DDQN.
    """

    def setUp(self) -> None:
        self.agent = DDQN(
            env_type="CartPole-v1",
            env_wrappers="none",
            policy_type="q-net",
            policy_kwargs={
                "architecture": [128],
                "activation_fn": nn.Tanh(),
                "output_activation_fn": None,
                "bias": True
            },
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring-buffer",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            lr_scheduler_type="linear-lr",
            lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.3, "total_iters": 1000},
            loss_type="huber",
            loss_kwargs={},
            max_gradient_norm=100,
            num_envs=8,
            steps_per_trajectory=64,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        loss, loss_info = self.agent.compute_loss(
            states=torch.rand(size=(8, 64, 4)),
            actions=torch.randint(low=0, high=2, size=(8, 64,)),
            rewards=torch.rand(size=(8, 64,)),
            next_states=torch.rand(size=(8, 64, 4)),
            dones=torch.randint(low=0, high=2, size=(8, 64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(loss_info, dict)


class TestClippedDDQN(unittest.TestCase):
    """
    Tests the class ClippedDDQN.
    """

    def setUp(self) -> None:
        self.agent = ClippedDDQN(
            env_type="CartPole-v1",
            env_wrappers="none",
            policy_type="q-net",
            policy_kwargs={
                "architecture": [128],
                "activation_fn": nn.Tanh(),
                "output_activation_fn": None,
                "bias": True
            },
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring-buffer",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            lr_scheduler_type="linear-lr",
            lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.3, "total_iters": 1000},
            loss_type="huber",
            loss_kwargs={},
            max_gradient_norm=100,
            num_envs=8,
            steps_per_trajectory=64,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        loss, loss_info = self.agent.compute_loss(
            states=torch.rand(size=(8, 64, 4)),
            actions=torch.randint(low=0, high=2, size=(8, 64,)),
            rewards=torch.rand(size=(8, 64,)),
            next_states=torch.rand(size=(8, 64, 4)),
            dones=torch.randint(low=0, high=2, size=(8, 64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(loss_info, dict)


if __name__ == '__main__':
    unittest.main()
