from typing import Optional

import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm

from pyrlagent.torch.algorithm import RLAlgorithm
from pyrlagent.torch.config import (
    BufferConfig,
    create_buffer,
    RLTrainConfig,
    RLTrainState,
    create_rl_components_eval,
    create_rl_components_train,
)
from pyrlagent.torch.experience import gae
from pyrlagent.torch.util import get_device


class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization (PPO).

    The corresponding paper can be found here:
    https://arxiv.org/abs/1707.06347

    Attributes:
        train_config (RLTrainConfig):
            The configuration of the RL training

        max_gradient_norm (float):
            The maximum gradient norm for gradient clipping

        num_envs (int):
            The number of different environments used for training

        steps_per_update (int):
            The number of timesteps T per update

        clip_ratio (float):
            The ratio of the trust region

        gamma (float):
            The discount factor of the return

        gae_lambda (float):
            The lambda weight of GAE

        vf_coef (float):
            The weight of the critic loss

        ent_coef (float):
            The weight of the entropy bonus

        update_steps (int):
            The number of gradient steps per update
    """

    def __init__(
        self,
        train_config: RLTrainConfig,
        max_gradient_norm: float,
        num_envs: int,
        steps_per_update: int,
        clip_ratio: float,
        gamma: float,
        gae_lambda: float,
        vf_coef: float,
        ent_coef: float,
        update_steps: int,
        device: str = "auto",
    ):
        self.train_config = train_config
        self.train_config.network_config.method = "pg"
        self.buffer_config = BufferConfig(id="rollout", kwargs={})
        self.max_gradient_norm = max_gradient_norm
        self.num_envs = num_envs
        self.steps_per_update = steps_per_update
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.update_steps = update_steps
        self.device = get_device(device)

        self.env = None
        self.rollout_buffer = None
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None

    def _setup_fit(self, train_state: Optional[RLTrainState] = None):
        """Set up the necessary components for the PPO algorithm."""
        # Create the env, network, optimizer, and lr_scheduler
        self.env, self.network, self.optimizer, self.lr_scheduler = (
            create_rl_components_train(
                train_config=self.train_config,
                num_envs=self.num_envs,
                train_state=train_state,
                device=self.device,
            )
        )

        # Create the replay buffer
        self.rollout_buffer = create_buffer(
            buffer_config=self.buffer_config,
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=(
                1
                if isinstance(self.env.single_action_space, gym.spaces.Discrete)
                else self.env.single_action_space.shape[0]
            ),
            env_dim=self.num_envs,
            max_size=self.steps_per_update,
            device=self.device,
        )

    def _setup_eval(self, train_state: Optional[RLTrainState] = None):
        """Set up the necessary components for the PPO algorithm."""
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
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PPO loss."""
        # Compute the probability distribution and critic values
        next_pi, next_values = self.network(states)

        # Compute the log probabilities
        next_log_prob = self.network.log_prob(next_pi, actions)

        # Actor loss
        ratio = torch.exp(next_log_prob - log_probs)
        loss_actor = -(
            torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages,
            )
        ).mean()

        # Critic loss
        targets_clipped = values + torch.clamp(
            next_values - values, -self.clip_ratio, self.clip_ratio
        )
        loss_critic = torch.square(next_values - targets)
        loss_critic_clipped = torch.square(targets_clipped - targets)
        loss_critic = 0.5 * torch.max(loss_critic, loss_critic_clipped).mean()

        # Entropy bonus
        entropy = next_pi.entropy().mean()

        # Total loss
        total_loss = loss_actor + self.vf_coef * loss_critic - self.ent_coef * entropy

        return total_loss

    def state_dict(self) -> RLTrainState:
        return RLTrainState(
            network_state=self.network.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            lr_scheduler_state=self.lr_scheduler.state_dict(),
        )

    def update(self):
        # Get the trajectories
        trajectory = self.rollout_buffer.sample(self.steps_per_update)

        # Calculate the advantages, targets according to Generalized Advantage Estimation (GAE)
        advantages, targets = gae(
            trajectory.reward,
            trajectory.value,
            trajectory.next_value,
            trajectory.done,
            self.gamma,
            self.gae_lambda,
        )

        for _ in range(self.update_steps):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute the PPO Loss (actor loss + critic loss)
            loss = self._compute_loss(
                states=trajectory.state,
                actions=trajectory.action,
                log_probs=trajectory.log_prob,
                advantages=advantages,
                values=trajectory.value,
                targets=targets,
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

        # Reset the buffer
        self.rollout_buffer.reset()

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
            # Get the next action and its log probability
            pi, value = self.network(state)
            action = pi.sample()
            log_prob = self.network.log_prob(pi, action)

            # Remove the gradients
            value = value.detach()
            action = action.detach()
            log_prob = log_prob.detach()

            # Do a single step on the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated).detach()

            _, next_value = self.network(next_state)
            next_value = next_value.detach()

            # Update the replay buffer by pushing the given transition
            self.rollout_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
                next_value=next_value,
            )

            # Update the state
            state = next_state

            if len(self.rollout_buffer) == self.steps_per_update:
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
            pi, _ = self.network(state)
            action = pi.sample().detach()

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
        clip_ratio = f"clip_ratio={self.clip_ratio},"
        gamma = f"gamma={self.gamma},"
        gae_lambda = f"gae_lambda={self.gae_lambda},"
        vf_coef = f"vf_coef={self.vf_coef},"
        ent_coef = f"ent_coef={self.ent_coef},"
        update_steps = f"update_steps={self.update_steps},"
        end = ")"

        return "\n".join(
            [
                header,
                max_gradient_norm,
                num_envs,
                steps_per_update,
                clip_ratio,
                gamma,
                gae_lambda,
                vf_coef,
                ent_coef,
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
            "clip_ratio": self.clip_ratio,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "update_steps": self.update_steps,
            "device": self.device,
        }

    def __setstate__(self, state: dict):
        self.train_config = state["train_config"]
        self.max_gradient_norm = state["max_gradient_norm"]
        self.num_envs = state["num_envs"]
        self.steps_per_update = state["steps_per_update"]
        self.clip_ratio = state["clip_ratio"]
        self.gamma = state["gamma"]
        self.gae_lambda = state["gae_lambda"]
        self.vf_coef = state["vf_coef"]
        self.ent_coef = state["ent_coef"]
        self.update_steps = state["update_steps"]
        self.device = state["device"]
