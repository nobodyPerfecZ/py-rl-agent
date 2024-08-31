import unittest

import torch
import torch.nn as nn

from pyrlagent.torch.config import LRSchedulerConfig, create_lr_scheduler


class TestLRSchedulerConfig(unittest.TestCase):
    """Tests the LRSchedulerConfig class."""

    def setUp(self):
        self.network = nn.Sequential(*[nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)])
        self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=0.01)

        self.step_lr_id = "step"
        self.step_lr_kwargs = {"step_size": 10, "gamma": 0.1}

        self.exponential_lr_id = "exponential"
        self.exponential_lr_kwargs = {"gamma": 0.1}

    def test_init(self):
        """Tests the __init__() method."""
        step_lr_config = LRSchedulerConfig(
            id=self.step_lr_id, kwargs=self.step_lr_kwargs
        )
        self.assertEqual(step_lr_config.id, self.step_lr_id)
        self.assertEqual(step_lr_config.kwargs, self.step_lr_kwargs)

        exponential_lr_config = LRSchedulerConfig(
            id=self.exponential_lr_id, kwargs=self.exponential_lr_kwargs
        )
        self.assertEqual(exponential_lr_config.id, self.exponential_lr_id)
        self.assertEqual(exponential_lr_config.kwargs, self.exponential_lr_kwargs)

    def test_create_lr_scheduler(self):
        """Tests the create_lr_scheduler() method."""
        # Create an invalid optimizer
        with self.assertRaises(ValueError):
            create_lr_scheduler(
                lr_scheduler_config=LRSchedulerConfig(id="invalid_id", kwargs={}),
                optimizer=self.optimizer,
            )

        # Create an step learning rate scheduler
        step_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.step_lr_id, kwargs=self.step_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(step_lr, torch.optim.lr_scheduler.StepLR)

        # Create an Adam optimizer
        exponential_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.exponential_lr_id, kwargs=self.exponential_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(exponential_lr, torch.optim.lr_scheduler.ExponentialLR)


if __name__ == "__main__":
    unittest.main()
