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


class LRSchedulerEnum(str, Enum):
    """
    An enumeration of supported learning rate scheduler types in PyTorch.

    The list of supported learning rate scheduler in PyTorch can be found here:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    NONE = "none"
    LAMBDA_LR = "lambda_lr"
    MULTIPLICATIVE_LR = "multiplicative_lr"
    STEP_LR = "step_lr"
    MULTISTEP_LR = "multistep_lr"
    CONSTANT_LR = "constant_lr"
    LINEAR_LR = "linear_lr"
    EXPONENTIAL_LR = "exponential_lr"
    POLYNOMIAL_LR = "polynomial_lr"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    CYCLIC_LR = "cyclic_lr"
    ONE_CYCLE_LR = "one_cycle_lr"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"

    @classmethod
    def wrapper(cls) -> dict[str, Any]:
        """
        Returns the wrapper dictionary, where
            - `key` represents the learning rate scheduler type as enum
            - `value` represents the learning rate scheduler class

        Returns:
            dict[str, Any]:
                The wrapper of LRSchedulerEnum
        """
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

    def to_lr_scheduler(self, optimizer: Optimizer, **lr_scheduler_kwargs) -> Optional[LRScheduler]:
        """
        Initialize the learning rate scheduler given the arguments.

        Args:
            optimizer (Optimizer):
                The used optimizer to schedule the learning rate of it

            **lr_scheduler_kwargs:
                Additional arguments for the lr_scheduler class

        Returns:
            LRScheduler | None:
                The initialized lr_scheduler
        """
        if self == LRSchedulerEnum.NONE:
            return None
        return LRSchedulerEnum.wrapper()[self](optimizer=optimizer, **lr_scheduler_kwargs)
