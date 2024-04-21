from enum import Enum

import torch.nn.functional as F

from PyRLAgent.enum.abstract_enum import AbstractStrEnum


class LossEnum(AbstractStrEnum):
    """
    An enumeration of supported loss functions.
    """
    MAE = "mae"
    MSE = "mse"
    HUBER = "huber"
    CE_LOGITS = "ce-logits"
    KL_LOGITS = "kl-logits"

    @classmethod
    def wrapper(cls) -> dict[Enum, F]:
        return {
            cls.MAE: F.l1_loss,
            cls.MSE: F.mse_loss,
            cls.HUBER: F.huber_loss,
            cls.CE_LOGITS: F.cross_entropy,
            cls.KL_LOGITS: F.kl_div,
        }

    def to(self) -> F:
        return LossEnum.wrapper()[self]
