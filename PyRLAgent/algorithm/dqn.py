import copy
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, LRScheduler, StepLR

from PyRLAgent.algorithm.abstract_algorithm import Algorithm
from PyRLAgent.algorithm.policy import QNetwork
from PyRLAgent.common.buffer.abstract_buffer import Buffer
from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.common.policy.abstract_policy import Policy
from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.util.environment import get_env
from PyRLAgent.util.interval_counter import IntervalCounter
from PyRLAgent.util.mapping import get_value


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
            Either the name or the class itself can be given.

        policy_type (str | Type[Policy]):
            The type of the policy, that we want to use.
            Either the name or the class itself can be given.

        policy_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the policy.

        strategy_type (str | Type[Strategy]):
            The type of exploration strategy that is used for the action selection (exploration-exploitation problem).
            Either the name or the class itself can be given.

        strategy_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the strategy.

        replay_buffer_type (str | Type[Buffer]):
            The type of replay buffer used for storing experiences (experience replay).
            Either the name or the class itself can be given.

        replay_buffer_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the replay buffer.

        optimizer_type (str | Type[Optimizer]):
            The type of optimizer used for training the policy.
            Either the name or the class itself can be given.

        optimizer_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the optimizer.

        loss_type (str | Type["F"]):
            The type of loss function used for training the policy.
            Either the name or the class itself can be given.

        loss_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the loss function.

        learning_starts (int):
            The number of steps before starting Q-learning updates.

        max_gradient_norm (float | None):
            The maximum gradient norm for gradient clipping.
            If the value is None, then no gradient clipping is used.

        batch_size (int):
            The batch size for the gradient upgrades.

        tau (float):
            The update rate of the target network (soft update).

        gamma (float):
            The discount factor (necessary for computing the TD-Target).

        target_freq (int):
            The frequency of target network updates.
            After N gradient updates, the target network will be updated.

        train_freq (int):
            The number of steps per gradient update.

        render_freq (int):
            The frequency of rendering the environment.
            After N episodes the environment gets rendered.

        gradient_steps (int):
            The number of gradient updates per training step.
    """

    def __init__(
            self,
            env_type: str,
            policy_type: Union[str, Type[Policy]],
            policy_kwargs: Dict[str, Any],
            strategy_type: Union[str, Type[Strategy]],
            strategy_kwargs: Dict[str, Any],
            replay_buffer_type: Union[str, Type[Buffer]],
            replay_buffer_kwargs: Dict[str, Any],
            optimizer_type: Union[str, Type[Optimizer]],
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_type: Union[str, Type[LRScheduler]],
            lr_scheduler_kwargs: Dict[str, Any],
            loss_type: Union[str, Type["F"]],
            loss_kwargs: Dict[str, Any],
            learning_starts: int,
            max_gradient_norm: Optional[float],
            batch_size: int,
            tau: float,
            gamma: float,
            target_freq: int,
            train_freq: int,
            render_freq: int,
            gradient_steps: int,
    ):
        self.env_type = env_type
        self.env, self.render_env = get_env(env_type, return_render=True)
        self.q_net = get_value(self.policy_mapping, policy_type)(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **policy_kwargs,
            strategy_type=strategy_type,
            strategy_kwargs=strategy_kwargs,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.freeze()

        self.replay_buffer_type = replay_buffer_type
        self.replay_buffer = get_value(self.replay_buffer_mapping, self.replay_buffer_type)(**replay_buffer_kwargs)

        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = get_value(self.optimizer_mapping, self.optimizer_type)(params=self.q_net.parameters(),
                                                                                **self.optimizer_kwargs)

        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler = get_value(self.lr_scheduler_mapping, self.lr_scheduler_type)(optimizer=self.optimizer,
                                                                                         **self.lr_scheduler_kwargs)

        self.loss_fn = get_value(self.loss_fn_mapping, loss_type)
        self.loss_kwargs = loss_kwargs

        self.max_gradient_norm = max_gradient_norm
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.target_freq = target_freq
        self.train_freq = train_freq
        self.render_freq = render_freq
        self.gradient_steps = gradient_steps

        self.target_count = IntervalCounter(initial_value=0, modulo=self.target_freq)
        self.train_count = IntervalCounter(initial_value=0, modulo=self.train_freq)
        self.render_count = IntervalCounter(initial_value=0, modulo=self.render_freq)

    @property
    def policy_mapping(self) -> dict[str, Any]:
        """
        Returns the mapping between keys and policy classes.

        Returns:
            dict[str, Any]:
                The mapping between keys and policy classes
        """
        return {"q-net": QNetwork}

    @property
    def replay_buffer_mapping(self) -> dict[str, Any]:
        """
        Returns the mapping between keys and replay buffer classes.

        Returns:
            dict[str, Any]:
                The mapping between keys and replay buffer classes
        """
        return {"ring": RingBuffer}

    @property
    def optimizer_mapping(self) -> dict[str, Any]:
        """
        Returns the mapping between keys and optimizer classes.

        Returns:
            dict[str, Any]:
                The mapping between keys and optimizer classes
        """
        return {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        }

    @property
    def lr_scheduler_mapping(self) -> dict[str, Any]:
        """
        Returns the mapping between keys and learning rate scheduler classes.

        Returns:
            dict[str, Any]:
                The mapping between keys and learning rate scheduler classes
        """
        return {
            "linear-lr": LinearLR,
            "exp-lr": ExponentialLR,
            "step-lr": StepLR,
        }

    @property
    def loss_fn_mapping(self) -> dict[str, Any]:
        """
        Returns the mapping between keys and loss function classes.

        Returns:
            dict[str, Any]:
                The mapping between keys and loss function classes
        """
        return {
            "mae": F.l1_loss,
            "mse": F.mse_loss,
            "huber": F.huber_loss,
        }

    def _reset(self):
        """
        Resets the environments to the start state and counters to 0.
        This method is necessary before starting with training the agent.
        """
        # Resets the environment
        self.env.reset()
        self.render_env.reset()

        # Reset the counters
        self.target_count.reset()
        self.train_count.reset()
        self.render_count.reset()

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
        if self.target_count.is_interval_reached():
            # Case: Update the target network with a fraction of the current network
            for model_params, target_params in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_params.copy_(self.tau * model_params + (1 - self.tau) * target_params)

        # Update the counter for soft updates
        self.target_count.increment()

    def train(self):
        """
        Train the DQN agent using experience stored in the replay buffer.

        This method is responsible for training the DQN agent using experiences collected in the replay buffer.
        It typically involves sampling mini-batches of experiences, computing Q-value targets, and updating
        the Q-network's parameters through gradient descent.

        After updating the Q-network parameters we apply gradient clipping and soft update of the target network.
        """
        if self.train_count.is_interval_reached():
            # Case: Update the Q-Network
            self.q_net.train(True)
            self.target_q_net.train(False)

            losses = []
            for _ in range(self.gradient_steps):
                # Sample from replay buffer
                samples = self.replay_buffer.sample(self.batch_size)

                # Reset the gradients to zero
                self.optimizer.zero_grad()

                # Compute the DQN Loss function
                loss = self.compute_loss(
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

                losses.append(loss.item())

            # Perform a learning rate scheduler update
            self.lr_scheduler.step()

            # Perform a soft update
            self._soft_update()

            return np.mean(losses)

        # Update the train count
        self.train_count.increment()

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss between the predicted q-values and the q-targets based on the given loss function.

        In DQN (with target network) the update rule is:
            - loss = L(Q_predicted, Q_target), where
            - Q_predicted := Q(s,a)
            - Q_target := r(s,a) + gamma * max_a' Q_target(s', a')

        Args:
            states (torch.Tensor):
                Minibatch of states s

            actions (torch.Tensor):
                Minibatch of actions a

            rewards (torch.Tensor):
                Minibatch of rewards r

            next_states (torch.Tensor):
                Minibatch of next_states s'

            dones (torch.Tensor):
                Minibatch of dones

        Returns:
            torch.Tensor:
                Loss between predicted and target Q-Values
        """

        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            Q_next = self.target_q_net.forward(next_states)
            q_targets = rewards + ~dones * self.gamma * Q_next.max(dim=1)[0]

            # Clip the target q-values
            q_targets = torch.clamp_(q_targets, min=self.target_q_net.Q_min, max=self.target_q_net.Q_max)

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        q_values = Q.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        return self.loss_fn(q_values, q_targets, **self.loss_kwargs)

    def predict(self, observation: np.ndarray, deterministic: bool) -> torch.Tensor:
        """
        Predicts the next action a given the observation by interacting with the environment.

        Args:
            observation (np.ndarray):
                Current observation getting by interacting with the given environment.

            deterministic (bool):
                Either the deterministic exploration strategy (:= True) or non-deterministic
                exploration strategy (:= False) should be used for selecting the action.

        Returns:
            torch.Tensor:
                Selected action as Pytorch Tensor
        """
        return self.q_net.predict(observation, deterministic)

    def fit(self, n_timesteps: Union[float, int]) -> list[float]:
        """
        Trains the agent for a specified number of timesteps.

        This function represents the main training loop, where the agent interacts with the environment for n_timestep
        steps.

        Args:
            n_timesteps (Union[float, int]):
                The number of steps to train the agent.

        Returns:
            list[float]:
                Accumulated rewards for each episode during training
        """
        # Reset parameters
        self._reset()
        rewards = []
        acc_reward = 0.0

        # Reset the environment
        if self.render_count.is_interval_reached():
            curr_env = self.render_env
        else:
            curr_env = self.env
        state, info = curr_env.reset()

        for timestep in range(int(n_timesteps)):
            # Get the next action
            action = self.q_net.predict(state, deterministic=False).item()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = curr_env.step(action)
            done = terminated or truncated
            acc_reward += reward

            # Update the exploration strategy with the given transition
            self.q_net.update_strategy(state, action, float(reward), next_state, done)

            # Update the replay buffer by pushing the given transition
            self.replay_buffer.push(state, action, float(reward), next_state, done)

            # Update the state
            state = next_state

            if self.replay_buffer.filled(self.learning_starts):
                # Case: Update the Q-Network
                self.train()

            if done:
                # Case: End of episode is reached
                # TODO: Use of logger instead of print-statements
                if self.render_count.is_interval_reached():
                    print(f"Timestep {timestep}, Reward: {acc_reward}")

                # Update render count
                self.render_count.increment()

                # Append accumulated rewards to the list of rewards for each episode
                rewards += [acc_reward]

                # Determine the environment for the next episode
                if self.render_count.is_interval_reached():
                    curr_env = self.render_env
                else:
                    curr_env = self.env

                # Reset the environment
                observation, info = curr_env.reset()
                state = observation
                acc_reward = 0
                continue

        return rewards

    def eval(self, n_timesteps: Union[float, int]):
        """
        The main evaluation loop, where the agent is tested for N episodes with an episode length of L.
        So that means no backpropagation (upgrading weights with gradients) are performed here.

        Args:
            n_timesteps (Union[float, int]):
                The number of steps to train the agent.

        Returns:
            list[float]:
                Accumulated rewards for each episode during training
        """
        # Reset parameters
        self._reset()
        rewards = []
        acc_reward = 0.0

        # Reset the environment
        if self.render_count.is_interval_reached():
            curr_env = self.render_env
        else:
            curr_env = self.env
        state, info = curr_env.reset()

        for timestep in range(int(n_timesteps)):
            # Get the next action
            action = self.q_net.predict(state, deterministic=True).item()

            # Do a step on the environment
            next_state, reward, terminated, truncated, info = curr_env.step(action)
            done = terminated or truncated
            acc_reward += reward

            # Update the exploration strategy with the given transition
            self.q_net.update_strategy(state, action, float(reward), next_state, done)

            # Update the state
            state = next_state

            if done:
                # Case: End of episode is reached
                # TODO: Use of logger instead of print-statements
                if self.render_count.is_interval_reached():
                    print(f"Timestep {timestep}, Reward: {acc_reward}")

                # Update render count
                self.render_count.increment()

                # Append accumulated rewards to the list of rewards for each episode
                rewards += [acc_reward]

                # Determine the environment for the next episode
                if self.render_count.is_interval_reached():
                    curr_env = self.render_env
                else:
                    curr_env = self.env

                # Reset the environment
                observation, info = curr_env.reset()
                state = observation
                acc_reward = 0
                continue

        return rewards

    def __str__(self) -> str:
        header = f"{self.__class__.__name__}("
        env_line = f"env={self.env_type},"
        q_net_line = f"q_net={self.q_net},"
        target_q_net_line = f"target_q_net={self.target_q_net},"
        replay_buffer_line = f"replay_buffer={self.replay_buffer},"
        optimizer_line = f"optimizer={self.optimizer},"
        loss_fn_line = f"loss_fn={self.loss_fn},"
        learning_starts_line = f"learning_starts={self.learning_starts},"
        max_gradient_norm_line = f"max_gradient_norm={self.max_gradient_norm},"
        batch_size_line = f"batch_size={self.batch_size},"
        tau_line = f"tau={self.tau},"
        gamma_line = f"gamma={self.gamma},"
        target_freq_line = f"target_freq={self.target_freq},"
        train_freq_line = f"train_freq={self.train_freq},"
        render_freq_line = f"render_freq={self.render_freq},"
        gradient_steps_line = f"gradient_steps={self.gradient_steps},"
        end = ")"
        return "\n".join([header, env_line, q_net_line, target_q_net_line, replay_buffer_line, optimizer_line,
                          loss_fn_line, learning_starts_line, max_gradient_norm_line, batch_size_line, tau_line,
                          gamma_line, target_freq_line, train_freq_line, render_freq_line, gradient_steps_line, end])

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> dict:
        return {
            "env_type": self.env_type,
            "q_net": self.q_net,
            "target_q_net": self.target_q_net,
            "replay_buffer_type": self.replay_buffer_type,
            "replay_buffer": self.replay_buffer,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "loss_fn": self.loss_fn,
            "loss_kwargs": self.loss_kwargs,
            "max_gradient_norm": self.max_gradient_norm,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "target_freq": self.target_freq,
            "train_freq": self.train_freq,
            "render_freq": self.render_freq,
            "gradient_steps": self.gradient_steps,
            "target_count": self.target_count,
            "train_count": self.train_count,
            "render_count": self.render_count,
        }

    def __setstate__(self, state: dict):
        self.env_type = state["env_type"]
        self.env, self.render_env = get_env(self.env_type, return_render=True)
        self.q_net = state["q_net"]
        self.target_q_net = state["target_q_net"]
        self.target_q_net.freeze()
        self.replay_type = state["replay_buffer_type"]
        self.replay_buffer = state["replay_buffer"]
        self.optimizer_type = state["optimizer_type"]
        self.optimizer_kwargs = state["optimizer_kwargs"]
        self.optimizer = get_value(self.optimizer_mapping, self.optimizer_type)(params=self.q_net.parameters(),
                                                                                **self.optimizer_kwargs)
        self.lr_scheduler_type = state["lr_scheduler_type"]
        self.lr_scheduler_kwargs = state["lr_scheduler_kwargs"]
        self.lr_scheduler = get_value(self.lr_scheduler_mapping, self.lr_scheduler_type)(optimizer=self.optimizer,
                                                                                         **self.lr_scheduler_kwargs)
        self.loss_fn = state["loss_fn"]
        self.loss_kwargs = state["loss_kwargs"]
        self.loss_fn = state["loss_fn"]
        self.loss_kwargs = state["loss_kwargs"]
        self.max_gradient_norm = state["max_gradient_norm"]
        self.learning_starts = state["learning_starts"]
        self.batch_size = state["batch_size"]
        self.tau = state["tau"]
        self.gamma = state["gamma"]
        self.target_freq = state["target_freq"]
        self.train_freq = state["train_freq"]
        self.render_freq = state["render_freq"]
        self.gradient_steps = state["gradient_steps"]
        self.target_count = state["target_count"]
        self.train_count = state["train_count"]
        self.render_count = state["render_count"]
