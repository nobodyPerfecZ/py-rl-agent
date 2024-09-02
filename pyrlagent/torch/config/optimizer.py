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
    if optimizer_config.id == "adadelta":
        optimizer = torch.optim.Adadelta(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "adagrad":
        optimizer = torch.optim.Adagrad(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "adam":
        optimizer = torch.optim.Adam(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "adamw":
        optimizer = torch.optim.AdamW(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "sparse_adam":
        optimizer = torch.optim.SparseAdam(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "adamax":
        optimizer = torch.optim.Adamax(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "asgd":
        optimizer = torch.optim.ASGD(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "nadam":
        optimizer = torch.optim.NAdam(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "radam":
        optimizer = torch.optim.RAdam(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "rprop":
        optimizer = torch.optim.Rprop(
            params=network.parameters(), **optimizer_config.kwargs
        )
    elif optimizer_config.id == "sgd":
        optimizer = torch.optim.SGD(
            params=network.parameters(), **optimizer_config.kwargs
        )
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_config.id}.")

    return optimizer
