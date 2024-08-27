import functools
from typing import Any

import numpy as np
import torch


@functools.singledispatch
def non_discounted_return(rewards: Any, dones: Any) -> Any:
    """
    Calculate the non-discounted returns of a trajectory of length T.

    Args:
        rewards (Any):
            The rewards of shape (T, env_dim)

        dones (np.ndarray):
            The completed flag of shape (T, env_dim)

    Returns:
        np.ndarray:
            The non-discounted returns of shape (T, env_dim)
    """
    raise TypeError(
        f"No known data type ({type(rewards)}) to calculate non-discounted return registered."
    )


@non_discounted_return.register(np.ndarray)
def _numpy_non_discounted_return(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + (1 - dones[t]) * returns[t + 1]
    return returns


@non_discounted_return.register(torch.Tensor)
def _torch_non_discounted_return(
    rewards: torch.Tensor, dones: torch.Tensor
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + (1 - dones[t]) * returns[t + 1]
    return returns


@functools.singledispatch
def discounted_return(rewards: Any, dones: Any, gamma: float) -> Any:
    """
    Calculate the discounted returns of a trajectory of length T.

    Args:
        rewards (Any):
            The rewards of shape (T, env_dim)

        dones (np.ndarray):
            The completed flag of shape (T, env_dim)

        gamma (float):
            The discount factor

    Returns:
        np.ndarray:
            The discounted returns of shape (T, env_dim)
    """
    raise TypeError(
        f"No known data type ({type(rewards)}) to calculate discounted return registered."
    )


@discounted_return.register(np.ndarray)
def _numpy_discounted_return(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + gamma * (1 - dones[t]) * returns[t + 1]
    return returns


@discounted_return.register(torch.Tensor)
def _torch_discounted_return(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + gamma * (1 - dones[t]) * returns[t + 1]
    return returns


@functools.singledispatch
def gae(
    rewards: Any,
    values: Any,
    next_values: Any,
    dones: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[Any, Any]:
    """
    Calculate the advantages and the target values using the generalized advantage estimation (GAE).

    Args:
        rewards (Any):
            The rewards of shape (T, env_dim)

        values (Any):
            The value estimates of shape (T, env_dim)

        next_values (Any):
            The next value estimates of shape (T, env_dim)

        dones (Any):
            The completed flag of shape (T, env_dim)

        gamma (float):
            The discount factor

        gae_lambda (float):
            The lambda factor for GAE

    Returns:
        tuple:
            advantages (Any):
                The advantages of shape (T, env_dim)

            target_values (Any):
                The target values of shape (T, env_dim)
    """
    raise TypeError(
        f"No known data type ({type(rewards)}) to calculate generalized advantage estimation registered."
    )


@gae.register(np.ndarray)
def _numpy_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Calculate deltas
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # Reverse deltas for lfilter
    advantages = np.zeros_like(rewards)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            advantages[t] = deltas[t]
        else:
            advantages[t] = (
                deltas[t] + gamma * gae_lambda * (1 - dones[t]) * advantages[t + 1]
            )

    return advantages, advantages + values


@gae.register(torch.Tensor)
def _torch_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculate deltas
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # Reverse deltas for lfilter
    advantages = torch.zeros_like(rewards)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            advantages[t] = deltas[t]
        else:
            advantages[t] = (
                deltas[t] + gamma * gae_lambda * (1 - dones[t]) * advantages[t + 1]
            )

    return advantages, advantages + values
