from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    """Configuration of the optimizer."""

    id: str
    kwargs: dict[str, Any]


def create_optimizer(
    optimizer_config: OptimizerConfig, network: nn.Module
) -> torch.optim.Optimizer:
    """
    Create the optimizer for training the neural network.

    Args:
        optimizer_config (OptimizerConfig):
            The configuration of the optimizer

        network (nn.Module):
            The neural network to optimize

    Returns:
        torch.optim.Optimizer:
            The optimizer
    """
    if optimizer_config.id == "sgd":
        optimizer = torch.optim.SGD(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "adam":
        optimizer = torch.optim.Adam(
            params=network.parameters(), **optimizer_config.kwargs
        )
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_config.id}.")

    return optimizer
