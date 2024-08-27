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

from PyRLAgent.enum.abstract_enum import AbstractStrEnum


class OptimizerEnum(AbstractStrEnum):
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
    def wrapper(cls) -> dict[Enum, Any]:
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

    def to(self, params: Iterable[nn.Parameter], lr: float, **optimizer_kwargs) -> Optimizer:
        return OptimizerEnum.wrapper()[self](params=params, lr=lr, **optimizer_kwargs)
