from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import torch
import torch.nn as nn

from pyrlagent.torch.config import (
    EnvConfig,
    LRSchedulerConfig,
    NetworkConfig,
    OptimizerConfig,
    create_env_eval,
    create_env_train,
    create_lr_scheduler,
    create_network,
    create_network_id,
    create_optimizer,
)
from pyrlagent.torch.util import get_obs_act_dims, get_obs_act_space


@dataclass
class RLTrainState:
    """State of the training of a neural network in RL."""

    network_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    lr_scheduler_state: dict[str, Any]


@dataclass
class RLTrainConfig:
    """Configuration for the training of a neural network in RL."""

    env_config: EnvConfig
    network_config: NetworkConfig
    optimizer_config: OptimizerConfig
    lr_scheduler_config: LRSchedulerConfig


def create_rl_components_train(
    train_config: RLTrainConfig,
    train_state: Optional[RLTrainState] = None,
    num_envs: int = 1,
    device: str = "cpu",
) -> tuple[
    gym.vector.VectorEnv,
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """
    Create the components for training a neural network in RL.
    The components include
        - the environment
        - the neural network
        - the optimizer
        - the learning rate scheduler

    Args:
        train_config (RLTrainConfig):
            The configuration for training a neural network in RL

        train_state (RLTrainState, optional):
            The state of the training of a neural network in RL

        num_envs (int):
            The number of parallel environments for the training

        device (str):
            The device to run the PyTorch computation

    Returns:
        tuple[gym.vector.VectorEnv, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
            env (gym.vector.VectorEnv):
                The environment for training the agent

            network (nn.Module):
                The neural network for the agent

            optimizer (torch.optim.Optimizer):
                The optimizer for the neural network

            lr_scheduler (torch.optim.lr_scheduler.LRScheduler):
                The learning rate scheduler for the optimizer
    """
    # Create the environment
    env = create_env_train(
        env_config=train_config.env_config,
        num_envs=num_envs,
        device=device,
    )

    # Create the network
    obs_space, act_space = get_obs_act_space(env)
    obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
    train_config.network_config.id = create_network_id(
        train_config.network_config.method, obs_space, act_space
    )

    network = create_network(
        network_config=train_config.network_config,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )
    if train_state is not None:
        network.load_state_dict(train_state.network_state)
    network.to(device=device)

    # Create the optimizer
    optimizer = create_optimizer(
        optimizer_config=train_config.optimizer_config,
        network=network,
    )
    if train_state is not None:
        optimizer.load_state_dict(train_state.optimizer_state)

    # Create the learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        lr_scheduler_config=train_config.lr_scheduler_config,
        optimizer=optimizer,
    )
    if train_state is not None:
        lr_scheduler.load_state_dict(train_state.lr_scheduler_state)

    return env, network, optimizer, lr_scheduler


def create_rl_components_eval(
    train_config: RLTrainConfig,
    train_state: Optional[RLTrainState] = None,
    device: str = "cpu",
) -> tuple[gym.Env, nn.Module]:
    """
    Create the components for evaluating a neural network in RL.
    The components include
        - the environment
        - the neural network

    Args:
        train_config (RLTrainConfig):
            The configuration for training a neural network in RL

        train_state (RLTrainState, optional):
            The state of the training of a neural network in RL

        device (str):
            The device to run the PyTorch computation

    Returns:
        tuple[gym.Env, nn.Module]:
            env (gym.Env):
                The environment for training the agent

            network (nn.Module):
                The neural network for the agent
    """
    # Create the environment
    env = create_env_eval(env_config=train_config.env_config, device=device)

    # Create the network
    obs_space, act_space = get_obs_act_space(env)
    obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
    train_config.network_config.id = create_network_id(
        train_config.network_config.method, obs_space, act_space
    )

    network = create_network(
        network_config=train_config.network_config, obs_dim=obs_dim, act_dim=act_dim
    )
    if train_state is not None:
        network.load_state_dict(train_state.network_state)
    network.to(device=device)

    return env, network
