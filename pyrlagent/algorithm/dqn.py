import copy
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from PyRLAgent.algorithm.abstract_algorithm import Algorithm
from PyRLAgent.common.policy.abstract_policy import DeterministicPolicy
from PyRLAgent.enum.buffer import BufferEnum
from PyRLAgent.enum.loss import LossEnum
from PyRLAgent.enum.lr_scheduler import LRSchedulerEnum
from PyRLAgent.enum.optimizer import OptimizerEnum
from PyRLAgent.enum.policy import PolicyEnum
from PyRLAgent.enum.strategy import StrategyEnum
from PyRLAgent.enum.wrapper import GymWrapperEnum
from PyRLAgent.util.environment import get_env
from PyRLAgent.util.interval_counter import IntervalCounter


class DQN(Algorithm):
    """
    Deep Q-Network (DQN) agent for reinforcement learning.

    The corresponding paper can be found here:
    https://arxiv.org/abs/1312.5602

    The DQN agent uses a neural network-based policy to approximate the Q-function and employs techniques
    such as experience replay and target networks for stable Q-learning.

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

        strategy_type (str | StrategyEnum):
            The type of exploration strategy that is used for the action selection (exploration-exploitation problem).
            Either the name or the enum itself can be given.

        strategy_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the strategy.

        replay_buffer_type (str | BufferEnum):
            The type of replay buffer used for storing experiences (experience replay).
            Either the name or the enum itself can be given.

        replay_buffer_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the replay buffer.

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

        loss_type (str | LossEnum):
            The type of loss function used for training the policy.
            Either the name or the class itself can be given.

        loss_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the loss function.

        max_gradient_norm (float | None):
            The maximum gradient norm for gradient clipping.
            If the value is None, then no gradient clipping is used.

        steps_per_trajectory (int):
            The batch size for the gradient upgrades.

        tau (float):
            The update rate of the target network (soft update).

        gamma (float):
            The discount factor (necessary for computing the TD-Target).

        target_freq (int):
            The frequency of target network updates.
            After N gradient updates, the target network will be updated.

        gradient_steps (int):
            The number of gradient updates per training step.
            For each training steps, N gradient steps will be performed.
    """

    def __init__(
            self,
            env_type: str,
            env_wrappers: Union[str, list[str], GymWrapperEnum, list[GymWrapperEnum]],
            policy_type: Union[str, Type[DeterministicPolicy]],
            policy_kwargs: Dict[str, Any],
            strategy_type: Union[str, StrategyEnum],
            strategy_kwargs: Dict[str, Any],
            replay_buffer_type: Union[str, BufferEnum],
            replay_buffer_kwargs: Dict[str, Any],
            optimizer_type: Union[str, OptimizerEnum],
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_type: Union[str, LRSchedulerEnum],
            lr_scheduler_kwargs: Dict[str, Any],
            loss_type: Union[str, LossEnum],
            loss_kwargs: Dict[str, Any],
            max_gradient_norm: Optional[float],
            num_envs: int,
            steps_per_trajectory: int,
            tau: float,
            gamma: float,
            target_freq: int,
            gradient_steps: int,
    ):
        self.env_type = env_type
        self.env_wrappers = [env_wrappers] if isinstance(env_wrappers, (str, GymWrapperEnum)) else env_wrappers
        self.env = None

        self.policy_type = policy_type
        self.policy_kwargs = policy_kwargs

        self.strategy_type = strategy_type
        self.strategy_kwargs = strategy_kwargs
        self.q_net = PolicyEnum(self.policy_type).to(
            observation_space=get_env(self.env_type).observation_space,
            action_space=get_env(self.env_type).action_space,
            **self.policy_kwargs,
            strategy_type=self.strategy_type,
            strategy_kwargs=self.strategy_kwargs,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.freeze()

        self.replay_buffer_type = replay_buffer_type
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self.replay_buffer = BufferEnum(self.replay_buffer_type).to(
            **self.replay_buffer_kwargs
        )

        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = OptimizerEnum(self.optimizer_type).to(
            params=self.q_net.parameters(),
            **self.optimizer_kwargs
        )

        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler = LRSchedulerEnum(self.lr_scheduler_type).to(
            optimizer=self.optimizer,
            **self.lr_scheduler_kwargs
        )

        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs
        self.loss_fn = LossEnum(self.loss_type).to()

        self.max_gradient_norm = max_gradient_norm
        self.num_envs = num_envs
        self.steps_per_trajectory = steps_per_trajectory
        self.tau = tau
        self.gamma = gamma

        self.target_freq = target_freq
        self.gradient_steps = gradient_steps

        self.target_count = IntervalCounter(initial_value=0, modulo=self.target_freq)
        self.train_count = IntervalCounter(initial_value=0, modulo=self.num_envs * self.steps_per_trajectory)

    def _apply_gradient_norm(self):
        """
        Applies gradient clipping on the policy weights based on the given maximum gradient norm.
        If the maximum gradient norm is not given, then no gradient norm will be performed.
        """
        if self.max_gradient_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient_norm)

    def _soft_update(self):
        """
        Performs a soft update of the target network parameters.

        The soft update is performed by blending the current target network parameters with the current model
        network parameters based on the given tau.
        """
        for model_params, target_params in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_params.copy_(self.tau * model_params + (1 - self.tau) * target_params)

    def train(self):
        """
        Train the DQN agent using experience stored in the replay buffer.

        This method is responsible for training the DQN agent using experiences collected in the replay buffer.
        It typically involves sampling mini-batches of experiences, computing Q-value targets, and updating
        the Q-network's parameters through gradient descent.

        After updating the Q-network parameters we apply gradient clipping and soft update of the target network.
        """
        for _ in range(self.gradient_steps):
            # Get the trajectories
            samples = self.replay_buffer.sample(self.steps_per_trajectory)

            # Reset the gradients to zero
            self.optimizer.zero_grad()

            # Compute the DQN Loss function
            loss, loss_info = self.compute_loss(
                states=samples.state,
                actions=samples.action,
                rewards=samples.reward,
                next_states=samples.next_state,
                dones=samples.done
            )

            # Perform the backpropagation
            loss.backward()

            # Clip the gradients
            self._apply_gradient_norm()

            # Perform the update step
            self.optimizer.step()

            # Perform a soft update
            if self.target_count.is_interval_reached():
                self._soft_update()

            # Update the counter
            self.target_count.increment()

        # Perform a learning rate scheduler update
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            Q_next = self.target_q_net.forward(next_states)
            targets = rewards + ~dones * self.gamma * Q_next.max(dim=-1)[0]

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        predicted = Q.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()

        return self.loss_fn(
            predicted.reshape(self.num_envs * self.steps_per_trajectory, -1),
            targets.reshape(self.num_envs * self.steps_per_trajectory, -1),
            **self.loss_kwargs
        ), {}

    def fit(self, n_timesteps: Union[float, int]) -> list[float]:
        # Reset parameters
        self.q_net.train()
        self.target_count.reset()
        self.train_count.reset()
        rewards = []
        acc_reward = 0.0
        progressbar = tqdm(total=int(n_timesteps))

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
            action = self.q_net.predict(state, deterministic=False)
            action = action.numpy()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = np.logical_or(terminated, truncated)
            acc_reward += np.mean(reward)

            # Update the exploration strategy with the given transition
            self.q_net.update_strategy(state, action, reward, next_state, done)

            # Update the replay buffer by pushing the given transition
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Update the state
            state = next_state

            if self.train_count.is_interval_reached() and self.replay_buffer.filled(self.steps_per_trajectory):
                # Case: Update the Q-Network
                self.train()

            # Update the progressbar
            progressbar.update(self.num_envs)

            # Update the counter
            self.train_count.increment(self.num_envs)

        # Close all necessary objects
        self.env.close()
        progressbar.close()
        return rewards

    def eval(self, n_timesteps: Union[float, int]):
        # Reset parameters
        self.q_net.eval()
        self.target_count.reset()
        self.train_count.reset()
        rewards = []
        acc_reward = 0.0

        # Create the environment
        self.env = GymWrapperEnum.create_env(name=self.env_type, wrappers=self.env_wrappers, render_mode="human")

        # Reset the environment
        state, info = self.env.reset()
        for timestep in tqdm(range(int(n_timesteps))):
            # Get the next action
            action = self.q_net.predict(state, deterministic=True).item()

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
        q_net = f"q_net={self.q_net},"
        target_q_net = f"target_q_net={self.target_q_net},"
        replay_buffer = f"replay_buffer={self.replay_buffer},"
        optimizer = f"optimizer={self.optimizer},"
        loss_fn = f"loss_fn=,{self.loss_fn}"
        max_gradient_norm = f"max_gradient_norm={self.max_gradient_norm},"
        num_envs = f"num_envs={self.num_envs},"
        steps_per_trajectory = f"steps_per_trajectory={self.steps_per_trajectory},"
        tau = f"tau={self.tau},"
        gamma = f"gamma={self.gamma},"
        target_freq = f"target_freq={self.target_freq},"
        gradient_steps = f"gradient_steps={self.gradient_steps},"
        end = ")"
        return "\n".join(
            [
                header, env, q_net, target_q_net, replay_buffer, optimizer, loss_fn, max_gradient_norm, num_envs,
                steps_per_trajectory, tau, gamma, target_freq, gradient_steps, end
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> dict:
        return {
            "env_type": self.env_type,
            "env_wrappers": self.env_wrappers,
            "q_net": self.q_net,
            "target_q_net": self.target_q_net,
            "replay_buffer_type": self.replay_buffer_type,
            "replay_buffer": self.replay_buffer,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "loss_type": self.loss_type,
            "loss_kwargs": self.loss_kwargs,
            "loss_fn": self.loss_fn,
            "max_gradient_norm": self.max_gradient_norm,
            "num_envs": self.num_envs,
            "steps_per_trajectory": self.steps_per_trajectory,
            "tau": self.tau,
            "gamma": self.gamma,
            "target_freq": self.target_freq,
            "gradient_steps": self.gradient_steps,
            "target_count": self.target_count,
            "train_count": self.train_count,
        }

    def __setstate__(self, state: dict):
        self.env_type = state["env_type"]
        self.env_wrappers = state["env_wrappers"]
        self.q_net = state["q_net"]
        self.target_q_net = state["target_q_net"]
        self.target_q_net.freeze()
        self.replay_type = state["replay_buffer_type"]
        self.replay_buffer = state["replay_buffer"]
        self.optimizer_type = state["optimizer_type"]
        self.optimizer_kwargs = state["optimizer_kwargs"]
        self.optimizer = self.optimizer = OptimizerEnum(self.optimizer_type).to(
            params=self.q_net.parameters(),
            **self.optimizer_kwargs
        )
        self.lr_scheduler_type = state["lr_scheduler_type"]
        self.lr_scheduler_kwargs = state["lr_scheduler_kwargs"]
        self.lr_scheduler = LRSchedulerEnum(self.lr_scheduler_type).to(
            optimizer=self.optimizer,
            **self.lr_scheduler_kwargs
        )
        self.loss_type = state["loss_type"]
        self.loss_kwargs = state["loss_kwargs"]
        self.loss_fn = state["loss_fn"]

        self.max_gradient_norm = state["max_gradient_norm"]
        self.num_envs = state["num_envs"]
        self.steps_per_trajectory = state["steps_per_trajectory"]
        self.tau = state["tau"]
        self.gamma = state["gamma"]
        self.target_freq = state["target_freq"]
        self.gradient_steps = state["gradient_steps"]
        self.target_count = state["target_count"]
        self.train_count = state["train_count"]
