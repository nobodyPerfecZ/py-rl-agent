from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from PyRLAgent.algorithm.abstract_algorithm import Algorithm
from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.common.policy.abstract_policy import ActorCriticPolicy
from PyRLAgent.enum.lr_scheduler import LRSchedulerEnum
from PyRLAgent.enum.optimizer import OptimizerEnum
from PyRLAgent.enum.policy import PolicyEnum
from PyRLAgent.enum.wrapper import GymWrapperEnum
from PyRLAgent.util.environment import get_env


class PPO(Algorithm):
    """
    Proximal Policy Optimization (PPO) agent for reinforcement learning.

    The corresponding paper can be found here:
    https://arxiv.org/abs/1707.06347

    Attributes:
        env_type (str):
            The environment where we want to optimize our agent.

        env_wrappers (str | list[str] | GymWrapperEnum | list[GymWrapperEnum]):
            The list of used Gymnasium Wrapper to transform the environment.

        policy_type (str | PolicyEnum):
            The type of the policy, that we want to use.
            Either the name or the enum itself can be given.

        policy_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the policy.

        optimizer_type (str | OptimizerEnum):
            The type of optimizer used for training the policy.
            Either the name or the enum itself can be given.

        optimizer_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the optimizer.

        lr_scheduler_type (str, LRSchedulerEnum):
            The type of the learning rate scheduler used for training the policy.
            Either the name or the enum itself can be given.

        lr_scheduler_kwargs(Dict[str, Any]):
            Keyword arguments for initializing the learning rate scheduler.

        max_gradient_norm (float | None):
            The maximum gradient norm for gradient clipping.
            If the value is None, then no gradient clipping is used.

        num_envs (int):
            The number of different actors N.

        steps_per_trajectory (int):
            The number of timesteps T.

        clip_ratio (float):
            The ratio of the trust region.

        gamma (float):
            The discount factor of the return.

        gae_lambda (float):
            The discount factor of the advantages.

        target_kl (float, optional):
            The target value of the KL divergence.
            If the KL divergence of the policy exceeds 1.5 times the target value, the training process will be early
            stopped.
            If no target value is given, then no early stopping will be performed.

        vf_coef (float):
            The weight of the critic loss.

        ent_coef (float):
            The weight of the entropy bonus.

        gradient_steps (int):
            The number of gradient updates per training step.
            For each training steps, N gradient steps will be performed.
    """

    def __init__(
            self,
            env_type: str,
            env_wrappers: Union[str, list[str], GymWrapperEnum, list[GymWrapperEnum]],
            policy_type: Union[str, Type[ActorCriticPolicy]],
            policy_kwargs: Dict[str, Any],
            optimizer_type: Union[str, OptimizerEnum],
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_type: Union[str, LRSchedulerEnum],
            lr_scheduler_kwargs: Dict[str, Any],
            max_gradient_norm: Optional[float],
            num_envs: int,
            steps_per_trajectory: int,
            clip_ratio: float,
            gamma: float,
            gae_lambda: float,
            target_kl: Optional[float],
            vf_coef: float,
            ent_coef: float,
            gradient_steps: int
    ):
        self.env_type = env_type
        self.env_wrappers = [env_wrappers] if isinstance(env_wrappers, (str, GymWrapperEnum)) else env_wrappers
        self.env = None

        self.policy_type = policy_type
        self.policy_kwargs = policy_kwargs
        self.policy = PolicyEnum(self.policy_type).to(
            observation_space=get_env(self.env_type, render_mode=None).observation_space,
            action_space=get_env(self.env_type, render_mode=None).action_space,
            **self.policy_kwargs,
        )

        self.num_envs = num_envs
        self.steps_per_trajectory = steps_per_trajectory
        self.replay_buffer = RingBuffer(max_size=self.num_envs * self.steps_per_trajectory)

        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = OptimizerEnum(self.optimizer_type).to(
            params=self.policy.parameters(),
            **self.optimizer_kwargs,
        )

        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler = LRSchedulerEnum(self.lr_scheduler_type).to(
            optimizer=self.optimizer,
            **self.lr_scheduler_kwargs,
        )

        self.max_gradient_norm = max_gradient_norm
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gradient_steps = gradient_steps

    def _apply_gradient_norm(self):
        """
        Applies gradient clipping on the policy weights based on the given maximum gradient norm.
        If the maximum gradient norm is not given, then no gradient norm will be performed.
        """
        if self.max_gradient_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_gradient_norm)

    def _early_stopping(self, kl: float) -> bool:
        """
        Applies early stopping on the policy updates based on the given target_kl.
        If target_kl is not given, then no early stopping will be performed.

        Args:
            kl (float):
                The current kl divergence value

        Returns:
            bool:
                True if early stopping should be performed, otherwise False
        """
        if self.target_kl is None:
            # Case: No early stopping should be performed
            return False
        return kl > 1.5 * self.target_kl

    def train(self):
        # Get the trajectories
        samples = self.replay_buffer.get(self.steps_per_trajectory)

        # Compute the next values
        _, _, _, next_values = self.policy.predict(samples.next_state, return_all=True)

        # Calculate the advantages, targets according to Generalized Advantage Estimation (GAE)
        advantages, targets = self.compute_gae(
            rewards=samples.reward,
            dones=samples.done,
            values=samples.value,
            next_values=next_values,
        )

        for i in range(self.gradient_steps):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute the PPO Loss (actor loss + critic loss)
            loss, loss_info = self.compute_loss(
                states=samples.state,
                actions=samples.action,
                log_probs=samples.log_prob,
                advantages=advantages,
                values=samples.value,
                targets=targets,
            )

            kl = loss_info["kl"]
            if self._early_stopping(kl):
                # Case: Perform early stopping
                break

            # Perform the backward propagation
            loss.backward()

            # Clip the gradients
            self._apply_gradient_norm()

            # Perform the gradient update
            self.optimizer.step()

        # Perform a learning rate scheduler update
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Reset the buffer
        self.replay_buffer.reset()

    def compute_gae(
            self,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the advantages after the Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor):
                Minibatch of rewards with shape (NUM_STEPS, NUM_ENVS)

            dones (torch.Tensor):
                Minibatch of dones with shape (NUM_STEPS, NUM_ENVS)

            values (torch.Tensor):
                Minibatch of values V(s) with shape (NUM_STEPS, NUM_ENVS)

            next_values (torch.Tensor):
                Minibatch of next values V(s_t+1) with shape (NUM_STEPS, NUM_ENVS)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                advantages (torch.Tensor):
                    Minibatch of computed advantages with shape (NUM_STEPS, NUM_ENVS)

                targets (torch.Tensor):
                    Minibatch of computed targets with shape (NUM_STEPS, NUM_ENVS)
        """
        # Compute temporal difference errors (deltas)
        # delta_t = r_t + gamma * (1-dones) * V(s_t+1) - V(s_t)
        deltas = rewards + self.gamma * ~dones * values - next_values

        # Compute advantages using Generalized Advantage Estimation (GAE)
        # A_t = delta_t + gamma * gae_lambda * (1-dones) * A_t+1
        advantages = torch.zeros(rewards.shape)
        advantages[-1, :] = deltas[-1, :]
        for t in reversed(range(self.steps_per_trajectory - 1)):
            advantages[t, :] = deltas[t, :] + self.gamma * self.gae_lambda * ~dones[t, :] * advantages[t + 1, :]

        # Normalization of the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, advantages + values

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            log_probs: torch.Tensor,
            advantages: torch.Tensor,
            values: torch.Tensor,
            targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Get the new log probabilities pi(a | s) and new values V(s)
        pi, new_log_probs, new_values = self.policy.forward(states, actions)

        # Actor loss
        ratio = torch.exp(new_log_probs - log_probs)
        loss_actor = -(torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        )).mean()

        # Critic loss
        loss_critic = 0.5 * (torch.max(
            ((new_values - targets) ** 2),
            ((torch.clamp(new_values - values, -self.clip_ratio, +self.clip_ratio) - targets) ** 2),
        )).mean()

        # Entropy bonus
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.vf_coef * loss_critic
            - self.ent_coef * entropy
        )

        # Compute extra information
        loss_info = {"kl": (log_probs - new_log_probs).mean().item()}
        return total_loss, loss_info

    def fit(self, n_timesteps: Union[float, int]) -> list[float]:
        # Reset parameters
        self.policy.train()
        rewards = []
        acc_reward = 0.0
        progressbar = tqdm(total=n_timesteps)

        # Create the training environment
        self.env = GymWrapperEnum.create_vector_env(
            name=self.env_type,
            num_envs=self.num_envs,
            wrappers=self.env_wrappers,
            render_mode=None
        )

        # Reset the environment
        state, info = self.env.reset()
        for _ in range(0, int(n_timesteps), self.num_envs):
            # Get the next action
            _, action, log_prob, value = self.policy.predict(state, return_all=True)
            action = action.numpy()
            log_prob = log_prob.numpy()
            value = value.numpy()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = np.logical_or(terminated,  truncated)
            acc_reward += np.mean(reward)

            # Update the replay buffer by pushing the given transition
            self.replay_buffer.push(state, action, reward, next_state, done, log_prob, value)

            # Update the state
            state = next_state

            if self.replay_buffer.filled(self.steps_per_trajectory):
                # Case: Update the Actor-Critic Network
                self.train()

            # Update the progressbar
            progressbar.update(self.num_envs)

        # Close all necessary objects
        self.env.close()
        progressbar.close()
        return rewards

    def eval(self, n_timesteps: Union[float, int]) -> list[float]:
        # Reset parameters
        self.policy.eval()
        rewards = []
        acc_reward = 0.0

        # Create the environment
        self.env = GymWrapperEnum.create_env(name=self.env_type, wrappers=self.env_wrappers, render_mode="human")

        # Reset the environment
        state, info = self.env.reset()
        for _ in tqdm(range(int(n_timesteps))):
            # Get the next action
            action = self.policy.predict(state, return_all=False).item()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            acc_reward += reward

            # Update the state
            state = next_state

            if done:
                # Append accumulated rewards to the list of rewards for each episode
                rewards += [acc_reward]

                # Reset the environment
                state, info = self.env.reset()
                acc_reward = 0
                continue

        # Close all necessary objects
        self.env.close()
        return rewards

    def __str__(self) -> str:
        header = f"{self.__class__.__name__}("
        env = f"env={self.env_type},"
        policy = f"policy={self.policy},"
        replay_buffer = f"replay_buffer={self.replay_buffer},"
        optimizer = f"optimizer={self.optimizer},"
        steps_per_trajectory = f"steps_per_trajectory={self.steps_per_trajectory}"
        clip_ratio = f"clip_ratio={self.clip_ratio}"
        gamma = f"gamma={self.gamma},"
        gae_lambda = f"gae_lambda={self.gae_lambda}"
        vf_coef = f"vf_coef={self.vf_coef}"
        ent_coef = f"ent_coef={self.ent_coef}"
        gradient_steps = f"gradient_steps={self.gradient_steps},"
        end = ")"
        return "\n".join(
            [
                header, env, policy, replay_buffer, optimizer, steps_per_trajectory, clip_ratio,
                gamma, gae_lambda, vf_coef, ent_coef, gradient_steps, end
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> dict:
        return {
            "env_type": self.env_type,
            "env_wrappers": self.env_wrappers,
            "policy_type": self.policy_type,
            "policy_kwargs": self.policy_kwargs,
            "policy": self.policy,
            "replay_buffer": self.replay_buffer,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "max_gradient_norm": self.max_gradient_norm,
            "num_envs": self.num_envs,
            "steps_per_trajectory": self.steps_per_trajectory,
            "clip_ratio": self.clip_ratio,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "target_kl": self.target_kl,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "gradient_steps": self.gradient_steps,
        }

    def __setstate__(self, state: dict):
        self.env_type = state["env_type"]
        self.env_wrappers = state["env_wrappers"]
        self.env = None
        self.policy_type = state["policy_type"]
        self.policy_kwargs = state["policy_kwargs"]
        self.policy = state["policy"]
        self.replay_buffer = state["replay_buffer"]
        self.optimizer_type = state["optimizer_type"]
        self.optimizer_kwargs = state["optimizer_kwargs"]
        self.optimizer = OptimizerEnum(self.optimizer_type).to(
            params=self.policy.parameters(),
            **self.optimizer_kwargs,
        )
        self.lr_scheduler_type = state["lr_scheduler_type"]
        self.lr_scheduler_kwargs = state["lr_scheduler_kwargs"]
        self.lr_scheduler = LRSchedulerEnum(self.lr_scheduler_type).to(
            optimizer=self.optimizer,
            **self.lr_scheduler_kwargs,
        )
        self.max_gradient_norm = state["max_gradient_norm"]
        self.num_envs = state["num_envs"]
        self.steps_per_trajectory = state["steps_per_trajectory"]
        self.clip_ratio = state["clip_ratio"]
        self.gamma = state["gamma"]
        self.gae_lambda = state["gae_lambda"]
        self.target_kl = state["target_kl"]
        self.vf_coef = state["vf_coef"]
        self.ent_coef = state["ent_coef"]
        self.gradient_steps = state["gradient_steps"]
