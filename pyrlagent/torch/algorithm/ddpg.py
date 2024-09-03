import copy
from typing import Optional

import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm

from pyrlagent.torch.algorithm import RLAlgorithm
from pyrlagent.torch.config import (
    BufferConfig,
    RLTrainConfig,
    RLTrainState,
    create_buffer,
    create_rl_components_eval,
    create_rl_components_train,
)
from pyrlagent.torch.util import get_device


class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient (DDPG).

    The corresponding paper can be found here:
    https://arxiv.org/abs/1509.02971

    Attributes:
        train_config (RLTrainConfig):
            The configuration of the RL training

        max_gradient_norm (float):
            The maximum gradient norm for gradient clipping

        num_envs (int):
            The number of different environments used for training

        steps_per_update (int):
            The number of timesteps T per update

        max_size (int):
            The maximum size of the replay buffer

        gamma (float):
            The discount factor of the return

        polyak (float):
            The weight of the target network update

        vf_coef (float):
            The weight of the critic loss

        update_steps (int):
            The number of gradient steps per update
    """

    def __init__(
        self,
        train_config: RLTrainConfig,
        max_gradient_norm: float,
        num_envs: int,
        steps_per_update: int,
        max_size: int,
        gamma: float,
        polyak: float,
        vf_coef: float,
        update_steps: int,
        device: str = "auto",
    ):
        self.train_config = train_config
        self.train_config.network_config.method = "ddpg"
        self.buffer_config = BufferConfig(id="replay", kwargs={})
        self.max_gradient_norm = max_gradient_norm
        self.num_envs = num_envs
        self.steps_per_update = steps_per_update
        self.max_size = int(max_size)
        self.gamma = gamma
        self.polyak = polyak
        self.vf_coef = vf_coef
        self.update_steps = update_steps
        self.device = get_device(device)

        self.env = None
        self.replay_buffer = None
        self.target_network = None
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None

    def _setup_fit(self, train_state: Optional[RLTrainState] = None):
        """Set up the necessary components for the DDPG algorithm."""
        # Create the env, network, optimizer, and lr_scheduler
        self.env, self.network, self.optimizer, self.lr_scheduler = (
            create_rl_components_train(
                train_config=self.train_config,
                num_envs=self.num_envs,
                train_state=train_state,
                device=self.device,
            )
        )
        # Create the target network
        self.target_network = copy.deepcopy(self.network)
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.target_network.to(device=self.device)

        # Create the replay buffer
        self.replay_buffer = create_buffer(
            buffer_config=self.buffer_config,
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=(
                1
                if isinstance(self.env.single_action_space, gym.spaces.Discrete)
                else self.env.single_action_space.shape[0]
            ),
            env_dim=self.num_envs,
            max_size=self.max_size,
            device=self.device,
        )

    def _setup_eval(self, train_state: Optional[RLTrainState] = None):
        """Set up the necessary components for the DDPG algorithm."""
        # Create the env and network
        self.env, self.network = create_rl_components_eval(
            train_config=self.train_config,
            train_state=train_state,
            device=self.device,
        )

    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the DDPG loss."""
        # Compute the actor loss
        loss_actor = -self.network.q_value(states, self.network.action(states)).mean()

        # Compute the critic loss
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - dones) * self.target_network.q_value(
                next_states, self.target_network.action(next_states)
            )
        prediction = self.network.q_value(states, actions)
        loss_critic = ((prediction - targets) ** 2).mean()

        total_loss = loss_actor + self.vf_coef * loss_critic
        return total_loss

    def state_dict(self) -> RLTrainState:
        return RLTrainState(
            network_state=self.network.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            lr_scheduler_state=self.lr_scheduler.state_dict(),
        )

    def update(self):
        # Get the trajectories
        trajectory = self.replay_buffer.sample(self.steps_per_update)

        for _ in range(self.update_steps):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute the PPO Loss (actor loss + critic loss)
            loss = self._compute_loss(
                states=trajectory.state,
                actions=trajectory.action,
                rewards=trajectory.reward,
                next_states=trajectory.next_state,
                dones=trajectory.done,
            )

            # Perform the backward propagation
            loss.backward()

            # Clip the gradients
            nn.utils.clip_grad_norm_(
                parameters=self.network.parameters(),
                max_norm=self.max_gradient_norm,
            )

            # Perform the gradient update
            self.optimizer.step()

        # Perform a learning rate scheduler update
        self.lr_scheduler.step()

        # Perform a polyak update of the target networks
        with torch.no_grad():
            for params, params_target in zip(
                self.network.parameters(), self.target_network.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                params_target.data.mul_(self.polyak)
                params_target.data.add_((1 - self.polyak) * params.data)

    def fit(
        self,
        num_timesteps: int,
        train_state: Optional[RLTrainState] = None,
    ) -> list[float]:
        # Create the training environment
        self._setup_fit(train_state=train_state)

        # Reset parameters
        self.network.train()
        progressbar = tqdm(total=int(num_timesteps))
        progressbar.update(0)

        # Reset the environment
        state, _ = self.env.reset()
        for _ in range(0, int(num_timesteps), self.num_envs):
            # Get the next action
            action = self.network.action(state).detach()

            # Do a single step on the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated).detach()

            # Update the replay buffer by pushing the given transition
            self.replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=None,
                value=None,
                next_value=None,
            )

            # Update the state
            state = next_state

            if len(self.replay_buffer) >= self.steps_per_update:
                # Case: Update the actor critic network
                self.update()

            progressbar.update(self.num_envs)

        # Close all necessary objects
        self.env.close()
        progressbar.close()
        return None

    def eval(
        self,
        num_timesteps: int,
        train_state: Optional[RLTrainState] = None,
    ) -> list[float]:
        # Create the training environment
        self._setup_eval(train_state=train_state)

        # Reset parameters
        self.network.eval()
        progressbar = tqdm(total=int(num_timesteps))

        # Reset the environment
        state, _ = self.env.reset()
        for _ in range(0, int(num_timesteps)):
            # Get the next action
            action = self.network.action(state).detach()

            # Do a single step on the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Update the progressbar
            progressbar.update(1)

            # Update the state
            state = next_state

        # Close all necessary objects
        self.env.close()
        return None

    def __str__(self) -> str:
        header = f"{self.__class__.__name__}("
        max_gradient_norm = f"max_gradient_norm={self.max_gradient_norm},"
        num_envs = f"num_envs={self.num_envs},"
        steps_per_update = f"steps_per_update={self.steps_per_update},"
        batch_size = f"batch_size={self.max_size},"
        gamma = f"gamma={self.gamma},"
        polyak = f"polyak={self.polyak},"
        vf_coef = f"vf_coef={self.vf_coef},"
        update_steps = f"update_steps={self.update_steps},"
        end = ")"

        return "\n".join(
            [
                header,
                max_gradient_norm,
                num_envs,
                steps_per_update,
                batch_size,
                gamma,
                polyak,
                vf_coef,
                update_steps,
                end,
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> dict:
        return {
            "train_config": self.train_config,
            "max_gradient_norm": self.max_gradient_norm,
            "num_envs": self.num_envs,
            "steps_per_update": self.steps_per_update,
            "max_size": self.max_size,
            "gamma": self.gamma,
            "polyak": self.polyak,
            "vf_coef": self.vf_coef,
            "update_steps": self.update_steps,
            "device": self.device,
        }

    def __setstate__(self, state: dict):
        self.train_config = state["train_config"]
        self.max_gradient_norm = state["max_gradient_norm"]
        self.num_envs = state["num_envs"]
        self.steps_per_update = state["steps_per_update"]
        self.max_size = state["max_size"]
        self.gamma = state["gamma"]
        self.polyak = state["polyak"]
        self.vf_coef = state["vf_coef"]
        self.update_steps = state["update_steps"]
        self.device = state["device"]
