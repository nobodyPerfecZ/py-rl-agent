import unittest

import torch
from torch import nn

from PyRLAgent.algorithm.c51 import C51


class TestC51(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = C51(
            env_type="CartPole-v1",
            env_wrappers=None,
            policy_type="q-prob-net",
            policy_kwargs={
                "Q_min": -1,
                "Q_max": 1,
                "num_atoms": 51,
                "architecture": [128],
                "activation_fn": nn.Tanh(),
                "output_activation_fn": None,
                "bias": True
            },
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            lr_scheduler_type="linear-lr",
            lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.3, "total_iters": 1000},
            loss_type="ce-logits",
            loss_kwargs={},
            max_gradient_norm=100,
            batch_size=64,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            train_freq=1,
            render_freq=50,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        loss, loss_info = self.agent.compute_loss(
            states=torch.rand(size=(64, 4)),
            actions=torch.randint(low=0, high=2, size=(64,)),
            rewards=torch.rand(size=(64,)),
            next_states=torch.rand(size=(64, 4)),
            dones=torch.randint(low=0, high=2, size=(64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(loss_info, dict)


if __name__ == '__main__':
    unittest.main()
