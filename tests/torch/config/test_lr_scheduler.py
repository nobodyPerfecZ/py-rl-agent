import unittest

import torch
import torch.nn as nn

from pyrlagent.torch.config import LRSchedulerConfig, create_lr_scheduler


class TestLRSchedulerConfig(unittest.TestCase):
    """Tests the LRSchedulerConfig class."""

    def setUp(self):
        self.network = nn.Sequential(*[nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)])
        self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=0.01)

        self.lambda_lr_id = "lambda"
        self.lambda_lr_kwargs = {"lr_lambda": lambda epoch: 0.95**epoch}

        self.multiplicative_lr_id = "multiplicative"
        self.multiplicative_lr_kwargs = {"lr_lambda": lambda epoch: 0.95**epoch}

        self.step_lr_id = "step"
        self.step_lr_kwargs = {"step_size": 10, "gamma": 0.1}

        self.multi_step_lr_id = "multi_step"
        self.multi_step_lr_kwargs = {"milestones": [30, 80], "gamma": 0.1}

        self.constant_lr_id = "constant"
        self.constant_lr_kwargs = {"factor": 0.9}

        self.linear_lr_id = "linear"
        self.linear_lr_kwargs = {"start_factor": 1.0, "end_factor": 0.5}

        self.exponential_lr_id = "exponential"
        self.exponential_lr_kwargs = {"gamma": 0.1}

        self.polynomial_lr_id = "polynomial"
        self.polynomial_lr_kwargs = {"power": 2}

        self.cosine_annealing_lr_id = "cosine_annealing"
        self.cosine_annealing_lr_kwargs = {"T_max": 100}

        self.reduce_on_plateau_lr_id = "reduce_on_plateau"
        self.reduce_on_plateau_lr_kwargs = {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
        }

        self.cyclic_lr_id = "cyclic"
        self.cyclic_lr_kwargs = {"base_lr": 0.001, "max_lr": 0.1, "step_size_up": 200}

        self.one_cycle_lr_id = "one_cycle"
        self.one_cycle_lr_kwargs = {"max_lr": 0.1, "steps_per_epoch": 100, "epochs": 10}

        self.cosine_warm_restarts_lr_id = "cosine_warm_restarts"
        self.cosine_warm_restarts_lr_kwargs = {"T_0": 100}

    def test_init(self):
        """Tests the __init__() method."""
        lambda_lr_config = LRSchedulerConfig(
            id=self.lambda_lr_id, kwargs=self.lambda_lr_kwargs
        )
        self.assertEqual(lambda_lr_config.id, self.lambda_lr_id)
        self.assertEqual(lambda_lr_config.kwargs, self.lambda_lr_kwargs)

        multiplicative_lr_config = LRSchedulerConfig(
            id=self.multiplicative_lr_id, kwargs=self.multiplicative_lr_kwargs
        )
        self.assertEqual(multiplicative_lr_config.id, self.multiplicative_lr_id)
        self.assertEqual(multiplicative_lr_config.kwargs, self.multiplicative_lr_kwargs)

        step_lr_config = LRSchedulerConfig(
            id=self.step_lr_id, kwargs=self.step_lr_kwargs
        )
        self.assertEqual(step_lr_config.id, self.step_lr_id)
        self.assertEqual(step_lr_config.kwargs, self.step_lr_kwargs)

        multi_step_lr_config = LRSchedulerConfig(
            id=self.multi_step_lr_id, kwargs=self.multi_step_lr_kwargs
        )
        self.assertEqual(multi_step_lr_config.id, self.multi_step_lr_id)
        self.assertEqual(multi_step_lr_config.kwargs, self.multi_step_lr_kwargs)

        constant_lr_config = LRSchedulerConfig(
            id=self.constant_lr_id, kwargs=self.constant_lr_kwargs
        )
        self.assertEqual(constant_lr_config.id, self.constant_lr_id)
        self.assertEqual(constant_lr_config.kwargs, self.constant_lr_kwargs)

        linear_lr_config = LRSchedulerConfig(
            id=self.linear_lr_id, kwargs=self.linear_lr_kwargs
        )
        self.assertEqual(linear_lr_config.id, self.linear_lr_id)
        self.assertEqual(linear_lr_config.kwargs, self.linear_lr_kwargs)

        exponential_lr_config = LRSchedulerConfig(
            id=self.exponential_lr_id, kwargs=self.exponential_lr_kwargs
        )
        self.assertEqual(exponential_lr_config.id, self.exponential_lr_id)
        self.assertEqual(exponential_lr_config.kwargs, self.exponential_lr_kwargs)

        polynomial_lr_config = LRSchedulerConfig(
            id=self.polynomial_lr_id, kwargs=self.polynomial_lr_kwargs
        )
        self.assertEqual(polynomial_lr_config.id, self.polynomial_lr_id)
        self.assertEqual(polynomial_lr_config.kwargs, self.polynomial_lr_kwargs)

        cosine_annealing_lr_config = LRSchedulerConfig(
            id=self.cosine_annealing_lr_id, kwargs=self.cosine_annealing_lr_kwargs
        )
        self.assertEqual(cosine_annealing_lr_config.id, self.cosine_annealing_lr_id)
        self.assertEqual(
            cosine_annealing_lr_config.kwargs, self.cosine_annealing_lr_kwargs
        )

        reduce_on_plateau_lr_config = LRSchedulerConfig(
            id=self.reduce_on_plateau_lr_id, kwargs=self.reduce_on_plateau_lr_kwargs
        )
        self.assertEqual(reduce_on_plateau_lr_config.id, self.reduce_on_plateau_lr_id)
        self.assertEqual(
            reduce_on_plateau_lr_config.kwargs, self.reduce_on_plateau_lr_kwargs
        )

        cyclic_lr_config = LRSchedulerConfig(
            id=self.cyclic_lr_id, kwargs=self.cyclic_lr_kwargs
        )
        self.assertEqual(cyclic_lr_config.id, self.cyclic_lr_id)
        self.assertEqual(cyclic_lr_config.kwargs, self.cyclic_lr_kwargs)

        one_cycle_lr_config = LRSchedulerConfig(
            id=self.one_cycle_lr_id, kwargs=self.one_cycle_lr_kwargs
        )
        self.assertEqual(one_cycle_lr_config.id, self.one_cycle_lr_id)
        self.assertEqual(one_cycle_lr_config.kwargs, self.one_cycle_lr_kwargs)

        cosine_warm_restarts_lr_config = LRSchedulerConfig(
            id=self.cosine_warm_restarts_lr_id,
            kwargs=self.cosine_warm_restarts_lr_kwargs,
        )
        self.assertEqual(
            cosine_warm_restarts_lr_config.id, self.cosine_warm_restarts_lr_id
        )
        self.assertEqual(
            cosine_warm_restarts_lr_config.kwargs, self.cosine_warm_restarts_lr_kwargs
        )

    def test_create_lr_scheduler(self):
        """Tests the create_lr_scheduler() method."""
        # Create an invalid optimizer
        with self.assertRaises(ValueError):
            create_lr_scheduler(
                lr_scheduler_config=LRSchedulerConfig(id="invalid_id", kwargs={}),
                optimizer=self.optimizer,
            )

        # Create a lambda learning rate scheduler
        lambda_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.lambda_lr_id, kwargs=self.lambda_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(lambda_lr, torch.optim.lr_scheduler.LambdaLR)

        # Create a multiplicative learning rate scheduler
        multiplicative_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.multiplicative_lr_id, kwargs=self.multiplicative_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(
            multiplicative_lr, torch.optim.lr_scheduler.MultiplicativeLR
        )

        # Create an step learning rate scheduler
        step_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.step_lr_id, kwargs=self.step_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(step_lr, torch.optim.lr_scheduler.StepLR)

        # Create an multi-step learning rate scheduler
        multi_step_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.multi_step_lr_id, kwargs=self.multi_step_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(multi_step_lr, torch.optim.lr_scheduler.MultiStepLR)

        # Create an constant learning rate scheduler
        constant_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.constant_lr_id, kwargs=self.constant_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(constant_lr, torch.optim.lr_scheduler.ConstantLR)

        # Create an linear learning rate scheduler
        linear_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.linear_lr_id, kwargs=self.linear_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(linear_lr, torch.optim.lr_scheduler.LinearLR)

        # Create an exponential learning rate scheduler
        exponential_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.exponential_lr_id, kwargs=self.exponential_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(exponential_lr, torch.optim.lr_scheduler.ExponentialLR)

        # Create an polynomial learning rate scheduler
        polynomial_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.polynomial_lr_id, kwargs=self.polynomial_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(polynomial_lr, torch.optim.lr_scheduler.PolynomialLR)

        # Create an cosine annealing learning rate scheduler
        cosine_annealing_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.cosine_annealing_lr_id, kwargs=self.cosine_annealing_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(
            cosine_annealing_lr, torch.optim.lr_scheduler.CosineAnnealingLR
        )

        # Create an reduce on plateau learning rate scheduler
        reduce_on_plateau_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.reduce_on_plateau_lr_id, kwargs=self.reduce_on_plateau_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(
            reduce_on_plateau_lr, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        # Create an cyclic learning rate scheduler
        cyclic_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.cyclic_lr_id, kwargs=self.cyclic_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(cyclic_lr, torch.optim.lr_scheduler.CyclicLR)

        # Create an one cycle learning rate scheduler
        one_cycle_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.one_cycle_lr_id, kwargs=self.one_cycle_lr_kwargs
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(one_cycle_lr, torch.optim.lr_scheduler.OneCycleLR)

        # Create an cosine warm restarts learning rate scheduler
        cosine_warm_restarts_lr = create_lr_scheduler(
            lr_scheduler_config=LRSchedulerConfig(
                id=self.cosine_warm_restarts_lr_id,
                kwargs=self.cosine_warm_restarts_lr_kwargs,
            ),
            optimizer=self.optimizer,
        )
        self.assertIsInstance(
            cosine_warm_restarts_lr,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        )


if __name__ == "__main__":
    unittest.main()
