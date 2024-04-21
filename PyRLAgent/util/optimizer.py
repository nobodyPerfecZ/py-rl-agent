from enum import Enum
from typing import Iterable, Any

from torch import nn
from torch.optim import (
    Optimizer,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD
)


class OptimizerEnum(str, Enum):
    """
    An enumeration of supported optimizer types in PyTorch.

    The list of supported optimizer in PyTorch can be found here:
    https://pytorch.org/docs/stable/optim.html#algorithms
    """
    ADADELTA = "adadelta"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMW = "adamw"
    SPARSEADAM = "sparseadam"
    ADAMAX = "adamax"
    ASGD = "asgd"
    LBFGS = "lbfgs"
    NADAM = "nadam"
    RADAM = "radam"
    RMSPROP = "rmsprop"
    RPROP = "rprop"
    SGD = "sgd"

    @classmethod
    def wrapper(cls) -> dict[str, Any]:
        """
        Returns the wrapper dictionary, where
            - `key` represents optimizer type as enum
            - `value` represents the optimizer class

        Returns:
            dict[str, Any]:
                The wrapper of OptimizerEnum
        """
        return {
            cls.ADADELTA: Adadelta,
            cls.ADAGRAD: Adagrad,
            cls.ADAM: Adam,
            cls.ADAMW: AdamW,
            cls.SPARSEADAM: SparseAdam,
            cls.ADAMAX: Adamax,
            cls.ASGD: ASGD,
            cls.LBFGS: LBFGS,
            cls.NADAM: NAdam,
            cls.RADAM: RAdam,
            cls.RMSPROP: RMSprop,
            cls.RPROP: Rprop,
            cls.SGD: SGD,
        }

    def to_optimizer(self, params: Iterable[nn.Parameter], lr: float, **optimizer_kwargs) -> Optimizer:
        """
        Initializes a new optimizer given the arguments.

        Args:
            params (Iterable[nn.Parameter]):
                The parameters to be optimized

            lr (float):
                The initial learning rate

            **optimizer_kwargs:
                Additional arguments for the optimizer class

        Returns:
            Optimizer:
                The initialized optimizer
        """
        return OptimizerEnum.wrapper()[self](params=params, lr=lr, **optimizer_kwargs)
