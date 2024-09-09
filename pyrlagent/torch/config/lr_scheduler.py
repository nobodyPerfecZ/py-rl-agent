from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LRSchedulerConfig:
    """Configuration of the learning rate scheduler."""

    id: str
    kwargs: dict[str, Any]


def create_lr_scheduler(
    lr_scheduler_config: LRSchedulerConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Creates the learning rate scheduler for the optimizer.

    Args:
        lr_scheduler_config (LRSchedulerConfig):
            The configuration of the learning rate scheduler

        optimizer (torch.optim.Optimizer):
            The optimizer for which to create the learning rate scheduler

    Returns:
        torch.optim.lr_scheduler.LRScheduler:
            The learning rate scheduler
    """
    if lr_scheduler_config.id == "lambda":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "multiplicative":
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "multi_step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "polynomial":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "cosine_annealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "cyclic":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "one_cycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "cosine_warm_restarts":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    else:
        raise ValueError(
            f"Invalid learning rate scheduler type: {lr_scheduler_config.id}."
        )
    return lr_scheduler
