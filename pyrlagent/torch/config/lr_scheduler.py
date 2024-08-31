from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LRSchedulerConfig:
    """Configuration of the learning rate scheduler."""

    id: str
    kwargs: dict[str, Any]


def create_lr_scheduler(
    lr_scheduler_config: LRSchedulerConfig, optimizer: torch.optim.Optimizer
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
    if lr_scheduler_config.id == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    elif lr_scheduler_config.id == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, **lr_scheduler_config.kwargs
        )
    else:
        raise ValueError(
            f"Invalid learning rate scheduler type: {lr_scheduler_config.id}."
        )
    return lr_scheduler
