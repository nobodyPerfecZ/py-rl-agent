from enum import Enum
from typing import Optional, Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)

from PyRLAgent.enum.abstract_enum import AbstractStrEnum


class LRSchedulerEnum(AbstractStrEnum):
    """
    An enumeration of supported learning rate scheduler types in PyTorch.

    The list of supported learning rate scheduler in PyTorch can be found here:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    NONE = "none"
    LAMBDA_LR = "lambda-lr"
    MULTIPLICATIVE_LR = "multiplicative-lr"
    STEP_LR = "step-lr"
    MULTISTEP_LR = "multistep-lr"
    CONSTANT_LR = "constant-lr"
    LINEAR_LR = "linear-lr"
    EXPONENTIAL_LR = "exponential-lr"
    POLYNOMIAL_LR = "polynomial-lr"
    COSINE_ANNEALING_LR = "cosine-annealing-lr"
    REDUCE_LR_ON_PLATEAU = "reduce-lr-on-plateau"
    CYCLIC_LR = "cyclic-lr"
    ONE_CYCLE_LR = "one-cycle-lr"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine-annealing-warm-restarts"

    @classmethod
    def wrapper(cls) -> dict[Enum, Any]:
        return {
            cls.NONE: None,
            cls.LAMBDA_LR: LambdaLR,
            cls.MULTIPLICATIVE_LR: MultiplicativeLR,
            cls.STEP_LR: StepLR,
            cls.MULTISTEP_LR: MultiStepLR,
            cls.CONSTANT_LR: ConstantLR,
            cls.LINEAR_LR: LinearLR,
            cls.EXPONENTIAL_LR: ExponentialLR,
            cls.POLYNOMIAL_LR: PolynomialLR,
            cls.COSINE_ANNEALING_LR: CosineAnnealingLR,
            cls.REDUCE_LR_ON_PLATEAU: ReduceLROnPlateau,
            cls.CYCLIC_LR: CyclicLR,
            cls.ONE_CYCLE_LR: OneCycleLR,
            cls.COSINE_ANNEALING_WARM_RESTARTS: CosineAnnealingWarmRestarts,
        }

    def to(self, optimizer: Optimizer, **lr_scheduler_kwargs) -> Optional[LRScheduler]:
        if self == LRSchedulerEnum.NONE:
            return None
        return LRSchedulerEnum.wrapper()[self](optimizer=optimizer, **lr_scheduler_kwargs)
