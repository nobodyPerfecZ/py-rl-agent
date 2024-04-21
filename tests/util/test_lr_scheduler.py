import unittest

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    LinearLR,
    StepLR,
    LambdaLR,
    MultiplicativeLR,
    MultiStepLR,
    ConstantLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)

from PyRLAgent.util.lr_scheduler import LRSchedulerEnum


class TestLRSchedulerEnum(unittest.TestCase):
    """
    Tests the enum class LRSchedulerEnum
    """

    def setUp(self):
        self.theta1 = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.theta2 = nn.Parameter(data=torch.tensor(2.0), requires_grad=True)
        self.optimizer = SGD(params=[self.theta1, self.theta2], lr=1e-3, momentum=1e-4, weight_decay=1e-5)

        self.wrapper = {
            LRSchedulerEnum.NONE: None,
            LRSchedulerEnum.LAMBDA_LR: LambdaLR,
            LRSchedulerEnum.MULTIPLICATIVE_LR: MultiplicativeLR,
            LRSchedulerEnum.STEP_LR: StepLR,
            LRSchedulerEnum.MULTISTEP_LR: MultiStepLR,
            LRSchedulerEnum.CONSTANT_LR: ConstantLR,
            LRSchedulerEnum.LINEAR_LR: LinearLR,
            LRSchedulerEnum.EXPONENTIAL_LR: ExponentialLR,
            LRSchedulerEnum.POLYNOMIAL_LR: PolynomialLR,
            LRSchedulerEnum.COSINE_ANNEALING_LR: CosineAnnealingLR,
            LRSchedulerEnum.REDUCE_LR_ON_PLATEAU: ReduceLROnPlateau,
            LRSchedulerEnum.CYCLIC_LR: CyclicLR,
            LRSchedulerEnum.ONE_CYCLE_LR: OneCycleLR,
            LRSchedulerEnum.COSINE_ANNEALING_WARM_RESTARTS: CosineAnnealingWarmRestarts,
        }

        self.lr_scheduler_kwargs1 = {"lr_lambda": lambda epoch: 0.95 ** epoch}
        self.lr_scheduler_kwargs2 = {"lr_lambda": lambda epoch: 0.95 ** epoch}
        self.lr_scheduler_kwargs3 = {"step_size": 30}
        self.lr_scheduler_kwargs4 = {"milestones": [30, 80]}
        self.lr_scheduler_kwargs5 = {}
        self.lr_scheduler_kwargs6 = {}
        self.lr_scheduler_kwargs7 = {"gamma": 0.95}
        self.lr_scheduler_kwargs8 = {}
        self.lr_scheduler_kwargs9 = {"T_max": 30}
        self.lr_scheduler_kwargs10 = {}
        self.lr_scheduler_kwargs11 = {"base_lr": 0.01, "max_lr": 0.1}
        self.lr_scheduler_kwargs12 = {"max_lr": 0.1, "total_steps": 30}
        self.lr_scheduler_kwargs13 = {"T_0": 30}

    def test_wrapper(self):
        """
        Tests the class method wrapper().
        """
        self.assertDictEqual(self.wrapper, LRSchedulerEnum.wrapper())

    def test_to_lr_scheduler(self):
        """
        Tests the method to_lr_scheduler().
        """
        lr_scheduler1 = LRSchedulerEnum.LAMBDA_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs1
        )
        lr_scheduler2 = LRSchedulerEnum.MULTIPLICATIVE_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs2
        )
        lr_scheduler3 = LRSchedulerEnum.STEP_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs3
        )
        lr_scheduler4 = LRSchedulerEnum.MULTISTEP_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs4
        )
        lr_scheduler5 = LRSchedulerEnum.CONSTANT_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs5
        )
        lr_scheduler6 = LRSchedulerEnum.LINEAR_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs6
        )
        lr_scheduler7 = LRSchedulerEnum.EXPONENTIAL_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs7
        )
        lr_scheduler8 = LRSchedulerEnum.POLYNOMIAL_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs8
        )
        lr_scheduler9 = LRSchedulerEnum.COSINE_ANNEALING_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs9
        )
        lr_scheduler10 = LRSchedulerEnum.REDUCE_LR_ON_PLATEAU.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs10
        )
        lr_scheduler11 = LRSchedulerEnum.CYCLIC_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs11
        )
        lr_scheduler12 = LRSchedulerEnum.ONE_CYCLE_LR.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs12
        )
        lr_scheduler13 = LRSchedulerEnum.COSINE_ANNEALING_WARM_RESTARTS.to_lr_scheduler(
            self.optimizer,
            **self.lr_scheduler_kwargs13
        )

        self.assertIsInstance(lr_scheduler1, LambdaLR)
        self.assertIsInstance(lr_scheduler2, MultiplicativeLR)
        self.assertIsInstance(lr_scheduler3, StepLR)
        self.assertIsInstance(lr_scheduler4, MultiStepLR)
        self.assertIsInstance(lr_scheduler5, ConstantLR)
        self.assertIsInstance(lr_scheduler6, LinearLR)
        self.assertIsInstance(lr_scheduler7, ExponentialLR)
        self.assertIsInstance(lr_scheduler8, PolynomialLR)
        self.assertIsInstance(lr_scheduler9, CosineAnnealingLR)
        self.assertIsInstance(lr_scheduler10, ReduceLROnPlateau)
        self.assertIsInstance(lr_scheduler11, CyclicLR)
        self.assertIsInstance(lr_scheduler12, OneCycleLR)
        self.assertIsInstance(lr_scheduler13, CosineAnnealingWarmRestarts)


if __name__ == "__main__":
    unittest.main()
