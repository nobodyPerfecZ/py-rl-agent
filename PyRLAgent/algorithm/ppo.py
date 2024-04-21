from typing import Any, Dict, Type, Union

import gymnasium as gym
import torch
from tqdm import tqdm

from PyRLAgent.algorithm.abstract_algorithm import Algorithm
from PyRLAgent.common.buffer.abstract_buffer import Buffer
from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.common.policy.abstract_policy import ActorCriticPolicy
from PyRLAgent.enum.lr_scheduler import LRSchedulerEnum
from PyRLAgent.enum.optimizer import OptimizerEnum
from PyRLAgent.enum.policy import PolicyEnum
from PyRLAgent.enum.wrapper import GymWrapperEnum
from PyRLAgent.util.environment import get_env, transform_env
from PyRLAgent.util.interval_counter import IntervalCounter
from PyRLAgent.util.mapping import get_values_enum


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

        batch_size (int):
            The batch size (number of different actors) N.

        steps_per_trajectory (int):
            The number of timesteps T.

        clip_ratio (float):
            The ratio of the trust region.

        gamma (float):
            The discount factor of the return.

        gae_lambda (float):
            The discount factor of the advantages.

        vf_coef (float):
            The weight of the critic loss.

        ent_coef (float):
            The weight of the entropy bonus.

        render_freq (int):
            The frequency of rendering the environment.
            After N episodes the environment gets rendered.

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
            batch_size: int,
            steps_per_trajectory: int,
            clip_ratio: float,
            gamma: float,
            gae_lambda: float,
            target_kl: float,
            vf_coef: float,
            ent_coef: float,
            render_freq: int,
            gradient_steps: int
    ):
        self.env_type = env_type
        if isinstance(env_wrappers, (str, GymWrapperEnum)):
            self.env_wrappers = GymWrapperEnum(env_wrappers)
        else:
            self.env_wrappers = [GymWrapperEnum(env_wrapper) for env_wrapper in env_wrappers]
            print(self.env_wrappers)
        self.env = None

        self.policy_type = policy_type
        self.model = PolicyEnum(self.policy_type).to(
            observation_space=get_env(self.env_type, render_mode=None).observation_space,
            action_space=get_env(self.env_type, render_mode=None).action_space,
            **policy_kwargs,
        )

        self.replay_buffer: Buffer = RingBuffer(max_size=batch_size * steps_per_trajectory)

        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = OptimizerEnum(self.optimizer_type).to(
            params=self.model.parameters(),
            **self.optimizer_kwargs,
        )

        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler = LRSchedulerEnum(self.lr_scheduler_type).to(
            optimizer=self.optimizer,
            **self.lr_scheduler_kwargs,
        )

        self.batch_size = batch_size
        self.steps_per_trajectory = steps_per_trajectory
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.render_freq = render_freq
        self.gradient_steps = gradient_steps

        # Counter
        self.render_count = IntervalCounter(initial_value=0, modulo=self.render_freq)

    def _reset(self):
        """
        Resets the all used counters.
        """
        # Reset the counters
        self.render_count.reset()

    def create_env(self, render_mode: str = None) -> gym.Env:
        if isinstance(self.env_wrappers, (str, GymWrapperEnum)):
            # Case: Single wrapper is given
            wrappers = [get_values_enum(GymWrapperEnum.wrapper(), self.env_wrappers)]
        else:
            # Case: Multiple wrappers are given
            wrappers = get_values_enum(GymWrapperEnum.wrapper(), self.env_wrappers)

        # Remove all none occurrences
        wrappers = [wrapper for wrapper in wrappers if wrapper is not None]

        if wrappers:
            # Case: Wrappers are given
            return transform_env(get_env(self.env_type, render_mode=render_mode), wrappers)
        else:
            # Case: No wrappers are given
            return get_env(self.env_type, render_mode=render_mode)

    def train(self):
        # Get the trajectories
        samples = self.replay_buffer.get(self.batch_size * self.steps_per_trajectory)

        # Calculate the advantages, targets according to Generalized Advantage Estimation (GAE)
        advantages, targets = self.compute_gae(
            rewards=samples.reward,
            dones=samples.done,
            values=samples.value
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
            if kl > 1.5 * self.target_kl:
                # Case: Early stopping technique
                break

            # Perform the backward propagation
            loss.backward()

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
            values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the advantages after the Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor):
                Tensor of shape [batch_size *  num_steps]
                Minibatch of rewards r

            dones (torch.Tensor):
                Tensor of shape [batch_size * num_steps]
                Minibatch of dones

            values (torch.Tensor):
                Tensor of shape [batch_size * num_steps]
                Minibatch of values V(s)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:

                advantages (torch.Tensor):
                    Tensor of shape [batch_size * num_steps]
                    Minibatch of computed advantages

                targets (torch.Tensor):
                    Tensor of shape [batch_size * num_steps]
                    Minibatch of computed targets
        """
        # Reshape rewards, dones and values to (batch_size, num_steps)
        rewards = rewards.reshape(self.batch_size, self.steps_per_trajectory)
        dones = dones.reshape(self.batch_size, self.steps_per_trajectory)
        values = values.reshape(self.batch_size, self.steps_per_trajectory)

        # Compute temporal difference errors (deltas)
        delta = torch.zeros(rewards.shape)

        # Extreme case T := delta_T = r_T - V(s_T)
        delta[:, -1] = rewards[:, -1] - values[:, -1]

        # All other cases t := delta_t = r_t + gamma * (1-dones) * V(s_t+1) - V(s_t)
        delta[:, :-1] = rewards[:, :-1] + self.gamma * values[:, 1:] * ~dones[:, :-1] - values[:, :-1]

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantage = torch.zeros(rewards.shape)

        # Extreme case T := A_T = delta_T
        advantage[:, -1] = delta[:, -1]

        # All other cases t := A_t = delta_t + gamma * gae_lambda * (1-dones) * A_t+1
        for t in reversed(range(self.steps_per_trajectory - 1)):
            advantage[:, t] = delta[:, t] + self.gamma * self.gae_lambda * ~dones[:, t] * advantage[:, t + 1]

        # Normalization of the advantages
        advantage = (advantage - advantage.mean(dim=0)) / (advantage.std(dim=0) + 1e-8)

        # Compute the targets
        targets = advantage + values

        # Reshape advantage and targets back to (batch_size * num_steps)
        advantage = advantage.reshape(-1)
        targets = advantage.reshape(-1)

        return advantage, targets

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
        pi, new_log_probs, new_values = self.model.forward(states, actions)

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
        self.model.train()

        # Reset parameters
        self._reset()
        rewards = []
        acc_reward = 0.0

        # Create the training environment
        self.env = self.create_env(render_mode=None)

        # Create the progress bar
        progressbar = tqdm(total=int(n_timesteps), desc="Training", unit="timesteps")
        old_timestep = 0

        # Reset the environment
        state, info = self.env.reset()
        for timestep in range(int(n_timesteps)):
            # Get the next action
            _, action, log_prob, value = self.model.predict(state, return_all=True)
            action = action.item()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            acc_reward += reward

            # Update the replay buffer by pushing the given transition
            self.replay_buffer.push(state, action, float(reward), next_state, done, log_prob, value)

            # Update the state
            state = next_state

            if self.replay_buffer.filled(self.batch_size * self.steps_per_trajectory):
                # Case: Update the Actor-Critic Network
                self.train()

            if done:
                # Case: End of episode is reached
                if self.render_count.is_interval_reached():
                    progressbar.set_postfix_str(f"{acc_reward:.2f}return")
                    progressbar.update(timestep - old_timestep)
                    old_timestep = timestep

                # Update render count
                self.render_count.increment()

                # Append accumulated rewards to the list of rewards for each episode
                rewards += [acc_reward]

                # Reset the environment
                state, info = self.env.reset()
                acc_reward = 0
                continue

        # Close all necessary objects
        progressbar.close()
        self.env.close()
        return rewards

    def eval(self, n_timesteps: Union[float, int]) -> list[float]:
        self.model.eval()

        # Reset parameters
        self._reset()
        rewards = []
        acc_reward = 0.0

        # Create the progress bar
        progressbar = tqdm(total=int(n_timesteps), desc="Training", unit="timesteps")
        old_timestep = 0

        # Create the environment
        self.env = self.create_env(render_mode="human")
        # self.env = get_env(self.env_type, render_mode="human")

        # Reset the environment
        state, info = self.env.reset()
        for timestep in range(int(n_timesteps)):
            # Get the next action
            action = self.model.predict(state, return_all=False).item()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            acc_reward += reward

            # Update the state
            state = next_state

            if done:
                # Case: End of episode is reached
                if self.render_count.is_interval_reached():
                    progressbar.set_postfix_str(f"{acc_reward:.2f}return")
                    progressbar.update(timestep - old_timestep)
                    old_timestep = timestep

                # Update render count
                self.render_count.increment()

                # Append accumulated rewards to the list of rewards for each episode
                rewards += [acc_reward]

                # Reset the environment
                state, info = self.env.reset()
                acc_reward = 0
                continue

        # Close all necessary objects
        progressbar.close()
        self.env.close()
        return rewards

    def __str__(self) -> str:
        header = f"{self.__class__.__name__}("
        env = f"env={self.env_type},"
        model = f"model={self.model},"
        replay_buffer = f"replay_buffer={self.replay_buffer},"
        optimizer = f"optimizer={self.optimizer},"
        steps_per_trajectory = f"steps_per_trajectory={self.steps_per_trajectory}"
        clip_ratio = f"clip_ratio={self.clip_ratio}"
        gamma = f"gamma={self.gamma},"
        gae_lambda = f"gae_lambda={self.gae_lambda}"
        vf_coef = f"vf_coef={self.vf_coef}"
        ent_coef = f"ent_coef={self.ent_coef}"
        render_freq = f"render_freq={self.render_freq},"
        gradient_steps = f"gradient_steps={self.gradient_steps},"
        end = ")"
        return "\n".join(
            [
                header, env, model, replay_buffer, optimizer, steps_per_trajectory, clip_ratio,
                gamma, gae_lambda, vf_coef, ent_coef, render_freq, gradient_steps, end
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> dict:
        return {
            "env_type": self.env_type,
            "model": self.model,
            "replay_buffer": self.replay_buffer,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "steps_per_trajectory": self.steps_per_trajectory,
            "clip_ratio": self.clip_ratio,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "render_freq": self.render_freq,
            "gradient_steps": self.gradient_steps,
        }

    def __setstate__(self, state: dict):
        self.env_type = state["env_type"]
        self.env = None
        self.model = state["model"]
        self.replay_buffer = state["replay_buffer"]
        self.optimizer_type = state["optimizer_type"]
        self.optimizer_kwargs = state["optimizer_kwargs"]
        self.optimizer = OptimizerEnum(self.optimizer_type).to(
            params=self.model.parameters(),
            **self.optimizer_kwargs,
        )
        self.lr_scheduler_type = state["lr_scheduler_type"]
        self.lr_scheduler_kwargs = state["lr_scheduler_kwargs"]
        self.lr_scheduler = LRSchedulerEnum(self.lr_scheduler_type).to(
            optimizer=self.optimizer,
            **self.lr_scheduler_kwargs,
        )
        self.steps_per_trajectory = state["steps_per_trajectory"]
        self.clip_ratio = state["clip_ratio"]
        self.gamma = state["gamma"]
        self.gae_lambda = state["gae_lambda"]
        self.vf_coef = state["vf_coef"]
        self.ent_coef = state["ent_coef"]
        self.render_freq = state["render_freq"]
        self.gradient_steps = state["gradient_steps"]
